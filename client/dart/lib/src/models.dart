import 'exceptions.dart';

/// A question sent to `/v1/systemone`.
class SystemOneQuestion {
  const SystemOneQuestion({
    required this.type,
    this.instructions,
    this.criteria,
  });

  /// Creates a choice question. Criteria may be a label map or a list.
  const SystemOneQuestion.choice({
    required Object criteria,
    Object? instructions,
  }) : this(type: 'choice', instructions: instructions, criteria: criteria);

  /// Creates a score question with an ordered criteria list.
  const SystemOneQuestion.score({
    required List<Object?> criteria,
    Object? instructions,
  }) : this(type: 'score', instructions: instructions, criteria: criteria);

  /// Creates a yes/no question. Criteria may contain `false` and `true`.
  const SystemOneQuestion.noul({
    Object? instructions,
    Map<String, Object?>? criteria,
  }) : this(type: 'noul', instructions: instructions, criteria: criteria);

  final String type;
  final Object? instructions;
  final Object? criteria;

  Map<String, Object?> toJson() {
    if (type != 'choice' && type != 'score' && type != 'noul') {
      throw ArgumentError.value(type, 'type', 'must be choice, score, or noul');
    }
    if (instructions != null) {
      _validateJsonValue(instructions, fieldName: 'instructions');
      if (instructions is! String &&
          instructions is! Map &&
          instructions is! List) {
        throw ArgumentError.value(
          instructions,
          'instructions',
          'must be a string, object, or array',
        );
      }
    }

    switch (type) {
      case 'choice':
        if (criteria is! Map && criteria is! List) {
          throw ArgumentError.value(
            criteria,
            'criteria',
            'choice criteria must be an object or array',
          );
        }
      case 'score':
        if (criteria is! List) {
          throw ArgumentError.value(
            criteria,
            'criteria',
            'score criteria must be an array',
          );
        }
      case 'noul':
        if (criteria != null && criteria is! Map) {
          throw ArgumentError.value(
            criteria,
            'criteria',
            'noul criteria must be an object',
          );
        }
    }
    if (criteria != null) {
      _validateJsonValue(criteria, fieldName: 'criteria');
    }

    final result = <String, Object?>{'type': type};
    if (instructions != null) result['instructions'] = instructions;
    if (criteria != null) result['criteria'] = criteria;
    return result;
  }
}

/// The common base for typed and future answer variants.
abstract class SystemOneAnswer {
  SystemOneAnswer({required this.type, required Map<String, Object?> raw})
    : raw = Map.unmodifiable(raw);

  factory SystemOneAnswer.fromJson(Object? json) {
    final map = _jsonMap(json, context: 'answer');
    final type = _requiredString(map, 'type', context: 'answer');
    switch (type) {
      case 'choice':
        return SystemOneChoiceAnswer._fromMap(map);
      case 'score':
        return SystemOneScoreAnswer._fromMap(map);
      case 'noul':
        return SystemOneNoulAnswer._fromMap(map);
      default:
        return SystemOneUnknownAnswer(type: type, raw: map);
    }
  }

  final String type;
  final Map<String, Object?> raw;
}

/// A selected choice and its probability distribution.
class SystemOneChoiceAnswer extends SystemOneAnswer {
  SystemOneChoiceAnswer({
    required this.choice,
    required this.confidence,
    required this.probabilities,
    this.action,
    required super.raw,
  }) : super(type: 'choice');

  factory SystemOneChoiceAnswer._fromMap(Map<String, Object?> map) {
    return SystemOneChoiceAnswer(
      choice: _requiredString(map, 'choice', context: 'choice answer'),
      confidence: _requiredDouble(map, 'confidence', context: 'choice answer'),
      probabilities: _numberMap(
        map['probabilities'],
        context: 'choice answer probabilities',
      ),
      action: _optionalAction(map['action'], context: 'choice answer'),
      raw: map,
    );
  }

  final String choice;
  final double confidence;
  final Map<String, double> probabilities;
  final SystemOneAction? action;
}

/// A probability-weighted score and its rubric.
class SystemOneScoreAnswer extends SystemOneAnswer {
  SystemOneScoreAnswer({
    required this.score,
    required this.confidence,
    required this.legend,
    required this.probabilities,
    this.action,
    required super.raw,
  }) : super(type: 'score');

  factory SystemOneScoreAnswer._fromMap(Map<String, Object?> map) {
    return SystemOneScoreAnswer(
      score: _requiredDouble(map, 'score', context: 'score answer'),
      confidence: _requiredDouble(map, 'confidence', context: 'score answer'),
      legend: _jsonMap(map['legend'], context: 'score answer legend'),
      probabilities: _numberMap(
        map['probabilities'],
        context: 'score answer probabilities',
      ),
      action: _optionalAction(map['action'], context: 'score answer'),
      raw: map,
    );
  }

  final double score;
  final double confidence;
  final Map<String, Object?> legend;
  final Map<String, double> probabilities;
  final SystemOneAction? action;
}

/// A yes/no probability. Some standard Jev responses omit confidence.
class SystemOneNoulAnswer extends SystemOneAnswer {
  SystemOneNoulAnswer({
    required this.noul,
    this.confidence,
    this.action,
    required super.raw,
  }) : super(type: 'noul');

  factory SystemOneNoulAnswer._fromMap(Map<String, Object?> map) {
    return SystemOneNoulAnswer(
      noul: _requiredDouble(map, 'noul', context: 'noul answer'),
      confidence: _optionalDouble(map['confidence'], context: 'noul answer'),
      action: _optionalAction(map['action'], context: 'noul answer'),
      raw: map,
    );
  }

  final double noul;
  final double? confidence;
  final SystemOneAction? action;
}

/// An answer type introduced by a newer server version.
class SystemOneUnknownAnswer extends SystemOneAnswer {
  SystemOneUnknownAnswer({required super.type, required super.raw});
}

/// Optional action metadata returned by the local System One runtime.
class SystemOneAction {
  SystemOneAction({
    required this.actProbability,
    required Map<String, Object?> raw,
  }) : raw = Map.unmodifiable(raw);

  factory SystemOneAction.fromJson(Object? json) {
    final map = _jsonMap(json, context: 'action');
    return SystemOneAction(
      actProbability: _optionalDouble(
        map['act_probability'],
        context: 'action',
      ),
      raw: map,
    );
  }

  final double? actProbability;
  final Map<String, Object?> raw;
}

/// Token usage reported by the server.
class SystemOneUsage {
  const SystemOneUsage({required this.inputTokens, required this.outputTokens});

  factory SystemOneUsage.fromJson(Object? json) {
    final map = _jsonMap(json, context: 'usage');
    return SystemOneUsage(
      inputTokens: _requiredInt(map, 'input_tokens', context: 'usage'),
      outputTokens: _requiredInt(map, 'output_tokens', context: 'usage'),
    );
  }

  final int inputTokens;
  final int outputTokens;
}

/// A complete `/v1/systemone` response.
class SystemOneResult {
  SystemOneResult({
    required this.model,
    required Map<String, SystemOneAnswer> answers,
    required this.usage,
    required Map<String, Object?> raw,
  }) : answers = Map.unmodifiable(answers),
       raw = Map.unmodifiable(raw);

  factory SystemOneResult.fromJson(Object? json) {
    final map = _jsonMap(json, context: 'system one response');
    final answerMap = _jsonMap(
      map['answers'],
      context: 'system one response answers',
    );
    final answers = <String, SystemOneAnswer>{};
    for (final entry in answerMap.entries) {
      answers[entry.key] = SystemOneAnswer.fromJson(entry.value);
    }
    return SystemOneResult(
      model: _requiredString(map, 'model', context: 'system one response'),
      answers: answers,
      usage: SystemOneUsage.fromJson(map['usage']),
      raw: map,
    );
  }

  final String model;
  final Map<String, SystemOneAnswer> answers;
  final SystemOneUsage usage;
  final Map<String, Object?> raw;
}

/// Runtime diagnostics reported by the local model service.
class SystemOneRuntimeInfo {
  SystemOneRuntimeInfo({required Map<String, Object?> raw})
    : runtime = _optionalString(raw['runtime'], context: 'runtime'),
      source = _optionalString(raw['source'], context: 'runtime'),
      version = _optionalString(raw['version'], context: 'runtime'),
      path = _optionalString(raw['path'], context: 'runtime'),
      platform = _optionalString(raw['platform'], context: 'runtime'),
      architecture = _optionalString(raw['architecture'], context: 'runtime'),
      backend = _optionalString(raw['backend'], context: 'runtime'),
      compatible = _optionalBool(raw['compatible'], context: 'runtime'),
      error = _optionalString(raw['error'], context: 'runtime'),
      raw = Map.unmodifiable(raw);

  factory SystemOneRuntimeInfo.fromJson(Object? json) {
    return SystemOneRuntimeInfo(raw: _jsonMap(json, context: 'runtime'));
  }

  final String? runtime;
  final String? source;
  final String? version;
  final String? path;
  final String? platform;
  final String? architecture;
  final String? backend;
  final bool? compatible;
  final String? error;
  final Map<String, Object?> raw;
}

/// Health information from `/health`.
class SystemOneHealth {
  SystemOneHealth({
    required this.status,
    required this.process,
    required this.model,
    required this.modelId,
    required this.runtime,
    required this.error,
    required Map<String, Object?> raw,
  }) : raw = Map.unmodifiable(raw);

  factory SystemOneHealth.fromJson(Object? json) {
    final map = _jsonMap(json, context: 'health response');
    return SystemOneHealth(
      status: _requiredString(map, 'status', context: 'health response'),
      process: _requiredString(map, 'process', context: 'health response'),
      model: _requiredString(map, 'model', context: 'health response'),
      modelId: _optionalString(map['model_id'], context: 'health response'),
      runtime: map['runtime'] == null
          ? null
          : SystemOneRuntimeInfo.fromJson(map['runtime']),
      error: _optionalString(map['error'], context: 'health response'),
      raw: map,
    );
  }

  final String status;
  final String process;
  final String model;
  final String? modelId;
  final SystemOneRuntimeInfo? runtime;
  final String? error;
  final Map<String, Object?> raw;

  bool get isHealthy => status == 'ok';
}

/// One model entry from either the official `models` list or the local `data`
/// list.
class SystemOneModelCard {
  SystemOneModelCard({
    required this.id,
    this.name,
    this.description,
    this.releaseDate,
    this.loaded,
    this.runtime,
    this.error,
    required Map<String, Object?> raw,
  }) : raw = Map.unmodifiable(raw);

  factory SystemOneModelCard.fromJson(Object? json) {
    final map = _jsonMap(json, context: 'model card');
    final id =
        _optionalString(map['id'], context: 'model card') ??
        _optionalString(map['name'], context: 'model card');
    if (id == null || id.isEmpty) {
      throw SystemOneResponseFormatException(
        message: 'model card must contain a non-empty id or name',
      );
    }
    return SystemOneModelCard(
      id: id,
      name: _optionalString(map['name'], context: 'model card'),
      description: _optionalString(map['description'], context: 'model card'),
      releaseDate: _optionalString(map['release_date'], context: 'model card'),
      loaded: _optionalBool(map['loaded'], context: 'model card'),
      runtime: map['runtime'] == null
          ? null
          : SystemOneRuntimeInfo.fromJson(map['runtime']),
      error: _optionalString(map['error'], context: 'model card'),
      raw: map,
    );
  }

  final String id;
  final String? name;
  final String? description;
  final String? releaseDate;
  final bool? loaded;
  final SystemOneRuntimeInfo? runtime;
  final String? error;
  final Map<String, Object?> raw;
}

/// Model discovery response. Both official `models` and local `data` shapes
/// are accepted; [models] is the canonical view, preferring `models` when
/// both shapes are present.
class SystemOneModelsResponse {
  SystemOneModelsResponse({
    required List<SystemOneModelCard> models,
    required List<SystemOneModelCard> data,
    required Map<String, Object?> raw,
  }) : models = List.unmodifiable(models),
       data = List.unmodifiable(data),
       raw = Map.unmodifiable(raw);

  factory SystemOneModelsResponse.fromJson(Object? json) {
    final map = _jsonMap(json, context: 'models response');
    final official = _modelList(
      map['models'],
      context: 'models response.models',
    );
    final local = _modelList(map['data'], context: 'models response.data');
    if (!map.containsKey('models') && !map.containsKey('data')) {
      throw SystemOneResponseFormatException(
        message: 'models response must contain a models or data list',
      );
    }
    return SystemOneModelsResponse(
      models: official.isNotEmpty ? official : local,
      data: local,
      raw: map,
    );
  }

  final List<SystemOneModelCard> models;
  final List<SystemOneModelCard> data;
  final Map<String, Object?> raw;
}

Map<String, Object?> _jsonMap(Object? value, {required String context}) {
  if (value is! Map) {
    throw SystemOneResponseFormatException(
      message: '$context must be a JSON object',
    );
  }
  final result = <String, Object?>{};
  for (final entry in value.entries) {
    if (entry.key is! String) {
      throw SystemOneResponseFormatException(
        message: '$context contains a non-string object key',
      );
    }
    result[entry.key as String] = entry.value;
  }
  return Map.unmodifiable(result);
}

List<SystemOneModelCard> _modelList(Object? value, {required String context}) {
  if (value == null) return <SystemOneModelCard>[];
  if (value is! List) {
    throw SystemOneResponseFormatException(
      message: '$context must be an array',
    );
  }
  return value
      .map((item) => SystemOneModelCard.fromJson(item))
      .toList(growable: false);
}

String _requiredString(
  Map<String, Object?> map,
  String key, {
  required String context,
}) {
  final value = map[key];
  if (value is String) return value;
  throw SystemOneResponseFormatException(
    message: '$context.$key must be a string',
    fieldPath: '$context.$key',
  );
}

String? _optionalString(Object? value, {required String context}) {
  if (value == null) return null;
  if (value is String) return value;
  throw SystemOneResponseFormatException(
    message: '$context value must be a string',
  );
}

bool? _optionalBool(Object? value, {required String context}) {
  if (value == null) return null;
  if (value is bool) return value;
  throw SystemOneResponseFormatException(
    message: '$context value must be boolean',
  );
}

double _requiredDouble(
  Map<String, Object?> map,
  String key, {
  required String context,
}) {
  final result = _optionalDouble(map[key], context: '$context.$key');
  if (result != null) return result;
  throw SystemOneResponseFormatException(
    message: '$context.$key must be a number',
    fieldPath: '$context.$key',
  );
}

double? _optionalDouble(Object? value, {required String context}) {
  if (value == null) return null;
  if (value is num) return value.toDouble();
  throw SystemOneResponseFormatException(message: '$context must be a number');
}

int _requiredInt(
  Map<String, Object?> map,
  String key, {
  required String context,
}) {
  final value = map[key];
  if (value is int) return value;
  if (value is num && value.isFinite && value == value.roundToDouble()) {
    return value.toInt();
  }
  throw SystemOneResponseFormatException(
    message: '$context.$key must be an integer',
    fieldPath: '$context.$key',
  );
}

Map<String, double> _numberMap(Object? value, {required String context}) {
  final map = _jsonMap(value, context: context);
  final result = <String, double>{};
  for (final entry in map.entries) {
    if (entry.value is! num) {
      throw SystemOneResponseFormatException(
        message: '$context.${entry.key} must be a number',
      );
    }
    result[entry.key] = (entry.value as num).toDouble();
  }
  return Map.unmodifiable(result);
}

SystemOneAction? _optionalAction(Object? value, {required String context}) {
  if (value == null) return null;
  final map = _jsonMap(value, context: '$context action');
  return SystemOneAction(
    actProbability: _optionalDouble(
      map['act_probability'],
      context: '$context action',
    ),
    raw: map,
  );
}

void _validateJsonValue(Object? value, {required String fieldName}) {
  if (value == null || value is String || value is bool || value is num) {
    return;
  }
  if (value is List) {
    for (var i = 0; i < value.length; i++) {
      _validateJsonValue(value[i], fieldName: '$fieldName[$i]');
    }
    return;
  }
  if (value is Map) {
    for (final entry in value.entries) {
      if (entry.key is! String) {
        throw ArgumentError('$fieldName contains a non-string map key');
      }
      _validateJsonValue(entry.value, fieldName: '$fieldName.${entry.key}');
    }
    return;
  }
  throw ArgumentError.value(
    value,
    fieldName,
    'must contain only JSON-compatible values',
  );
}
