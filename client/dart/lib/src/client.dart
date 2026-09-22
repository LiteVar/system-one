import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

import 'exceptions.dart';
import 'models.dart';
import 'retry.dart';

typedef _ResponseParser<T> = T Function(Object? json);

/// A pure Dart client for System One / Jev-style HTTP APIs.
class SystemOneClient {
  SystemOneClient({
    String baseUrl = 'http://127.0.0.1:8080',
    this.apiKey,
    String defaultModel = 'laya-multilingual',
    SystemOneRetryPolicy? retry,
    Duration timeout = const Duration(seconds: 10),
    Map<String, String>? defaultHeaders,
    http.Client? httpClient,
    bool closeClient = false,
  }) : _baseUri = _parseBaseUrl(baseUrl),
       defaultModel = _validateModel(defaultModel),
       timeout = _validateTimeout(timeout),
       defaultHeaders = Map.unmodifiable(defaultHeaders ?? const {}),
       _httpClient = httpClient ?? http.Client(),
       _closeHttpClient = httpClient == null || closeClient,
       _retry = retry ?? SystemOneRetryPolicy.defaults();

  final Uri _baseUri;
  final http.Client _httpClient;
  final bool _closeHttpClient;
  final SystemOneRetryPolicy _retry;
  bool _closed = false;

  final String? apiKey;
  final String defaultModel;
  final Duration timeout;
  final Map<String, String> defaultHeaders;

  /// The normalized base URL used by this client.
  Uri get baseUri => _baseUri;

  /// Evaluates one or more named questions against [state].
  Future<SystemOneResult> systemOne({
    required Object? state,
    required Map<String, SystemOneQuestion> questions,
    String? model,
    Duration? timeout,
    SystemOneRetryPolicy? retry,
    Map<String, String>? headers,
    Future<void>? cancellation,
  }) async {
    _ensureOpen();
    _validateState(state);
    if (questions.isEmpty) {
      throw ArgumentError.value(
        questions,
        'questions',
        'must contain at least one question',
      );
    }

    final selectedModel = model ?? defaultModel;
    if (selectedModel.isEmpty) {
      throw ArgumentError.value(model, 'model', 'must not be empty');
    }
    final serializedQuestions = <String, Object?>{};
    for (final entry in questions.entries) {
      if (entry.key.isEmpty) {
        throw ArgumentError.value(
          entry.key,
          'questions',
          'question ids must not be empty',
        );
      }
      serializedQuestions[entry.key] = entry.value.toJson();
    }

    final body = <String, Object?>{
      'model': selectedModel,
      'state': state,
      'questions': serializedQuestions,
    };
    return _request(
      method: 'POST',
      path: '/v1/systemone',
      body: body,
      timeout: timeout,
      retry: retry,
      headers: headers,
      cancellation: cancellation,
      parse: SystemOneResult.fromJson,
    );
  }

  /// Checks process and model availability without using HTTP status alone.
  Future<SystemOneHealth> health({
    Duration? timeout,
    SystemOneRetryPolicy? retry,
    Map<String, String>? headers,
    Future<void>? cancellation,
  }) {
    return _request(
      method: 'GET',
      path: '/health',
      timeout: timeout,
      retry: retry,
      headers: headers,
      cancellation: cancellation,
      parse: SystemOneHealth.fromJson,
    );
  }

  /// Lists model metadata from either the standard or local response shape.
  Future<SystemOneModelsResponse> models({
    Duration? timeout,
    SystemOneRetryPolicy? retry,
    Map<String, String>? headers,
    Future<void>? cancellation,
  }) {
    return _request(
      method: 'GET',
      path: '/v1/models',
      timeout: timeout,
      retry: retry,
      headers: headers,
      cancellation: cancellation,
      parse: SystemOneModelsResponse.fromJson,
    );
  }

  /// Closes the owned transport. Closing is idempotent.
  void close() {
    if (_closed) return;
    _closed = true;
    if (_closeHttpClient) _httpClient.close();
  }

  Future<T> _request<T>({
    required String method,
    required String path,
    Object? body,
    required Duration? timeout,
    required SystemOneRetryPolicy? retry,
    required Map<String, String>? headers,
    required Future<void>? cancellation,
    required _ResponseParser<T> parse,
  }) async {
    _ensureOpen();
    final requestTimeout = _validateTimeout(timeout ?? this.timeout);
    final retryPolicy = retry ?? _retry;
    final uri = _endpoint(path);
    final encodedBody = body == null ? null : _encodeJson(body);
    final requestHeaders = _mergeHeaders(headers, hasBody: encodedBody != null);
    var retryNumber = 0;

    while (true) {
      try {
        final response = await _sendOnce(
          method: method,
          uri: uri,
          body: encodedBody,
          headers: requestHeaders,
          timeout: requestTimeout,
          cancellation: cancellation,
        );
        if (response.statusCode < 200 || response.statusCode >= 300) {
          final error = _apiException(response);
          if (retryNumber < retryPolicy.maxRetries &&
              retryPolicy.shouldRetryStatus(response.statusCode)) {
            retryNumber++;
            await _waitBeforeRetry(
              retryPolicy,
              retryNumber: retryNumber,
              headers: response.headers,
              cancellation: cancellation,
              uri: uri,
            );
            continue;
          }
          throw error;
        }

        return _parseResponse(response.body, uri, parse);
      } on SystemOneAbortException {
        rethrow;
      } on SystemOneTimeoutException {
        if (retryNumber < retryPolicy.maxRetries && retryPolicy.retryTimeouts) {
          retryNumber++;
          await _waitBeforeRetry(
            retryPolicy,
            retryNumber: retryNumber,
            cancellation: cancellation,
            uri: uri,
          );
          continue;
        }
        rethrow;
      } on http.ClientException catch (error, stackTrace) {
        final wrapped = SystemOneConnectionException(
          message: 'HTTP transport failed while requesting $uri: $error',
          uri: uri,
          cause: error,
          stackTrace: stackTrace,
        );
        if (retryNumber < retryPolicy.maxRetries &&
            retryPolicy.retryConnectionErrors) {
          retryNumber++;
          await _waitBeforeRetry(
            retryPolicy,
            retryNumber: retryNumber,
            cancellation: cancellation,
            uri: uri,
          );
          continue;
        }
        throw wrapped;
      } on SystemOneConnectionException {
        if (retryNumber < retryPolicy.maxRetries &&
            retryPolicy.retryConnectionErrors) {
          retryNumber++;
          await _waitBeforeRetry(
            retryPolicy,
            retryNumber: retryNumber,
            cancellation: cancellation,
            uri: uri,
          );
          continue;
        }
        rethrow;
      }
    }
  }

  Future<http.Response> _sendOnce({
    required String method,
    required Uri uri,
    required String? body,
    required Map<String, String> headers,
    required Duration timeout,
    required Future<void>? cancellation,
  }) async {
    final request = http.Request(method, uri);
    request.headers.addAll(headers);
    if (body != null) request.body = body;

    try {
      final future = _sendAndRead(request, cancellation);
      return await future.timeout(timeout);
    } on TimeoutException catch (error, stackTrace) {
      throw SystemOneTimeoutException(
        timeout: timeout,
        uri: uri,
        cause: error,
        stackTrace: stackTrace,
      );
    }
  }

  Future<http.Response> _sendAndRead(
    http.Request request,
    Future<void>? cancellation,
  ) async {
    final streamed = await _awaitCancellable(
      _httpClient.send(request),
      cancellation,
      request.url,
    );
    return _awaitCancellable(
      http.Response.fromStream(streamed),
      cancellation,
      request.url,
    );
  }

  Future<T> _awaitCancellable<T>(
    Future<T> operation,
    Future<void>? cancellation,
    Uri uri,
  ) {
    if (cancellation == null) return operation;

    final completer = Completer<T>();
    operation.then(
      (value) {
        if (!completer.isCompleted) completer.complete(value);
      },
      onError: (Object error, StackTrace stackTrace) {
        if (!completer.isCompleted) {
          completer.completeError(error, stackTrace);
        }
      },
    );
    cancellation.then<void>(
      (_) {
        if (!completer.isCompleted) {
          completer.completeError(SystemOneAbortException(uri: uri));
        }
      },
      onError: (Object _, StackTrace _) {
        if (!completer.isCompleted) {
          completer.completeError(SystemOneAbortException(uri: uri));
        }
      },
    );
    return completer.future;
  }

  Future<void> _waitBeforeRetry(
    SystemOneRetryPolicy policy, {
    required int retryNumber,
    required Future<void>? cancellation,
    required Uri uri,
    Map<String, String>? headers,
  }) async {
    final delay = policy.delayFor(retryNumber: retryNumber, headers: headers);
    if (delay == Duration.zero) {
      await _awaitCancellable(Future<void>.value(), cancellation, uri);
      return;
    }
    await _awaitCancellable(Future<void>.delayed(delay), cancellation, uri);
  }

  T _parseResponse<T>(String body, Uri uri, _ResponseParser<T> parse) {
    try {
      return parse(jsonDecode(body));
    } on SystemOneException {
      rethrow;
    } on Object catch (error, stackTrace) {
      throw SystemOneResponseFormatException(
        message: 'Invalid JSON response from $uri: $error',
        uri: uri,
        cause: error,
        stackTrace: stackTrace,
      );
    }
  }

  SystemOneApiException _apiException(http.Response response) {
    final parsed = _tryJsonMap(response.body);
    var code = 'http_error';
    var message =
        'HTTP ${response.statusCode} returned from ${response.request?.url ?? _baseUri}';

    final errorValue = parsed?['error'];
    if (errorValue is Map) {
      final errorMap = _tryJsonMapValue(errorValue);
      final parsedCode = errorMap?['code'];
      final parsedMessage = errorMap?['message'];
      if (parsedCode is String && parsedCode.isNotEmpty) code = parsedCode;
      if (parsedMessage is String && parsedMessage.isNotEmpty) {
        message = parsedMessage;
      }
    }

    final detail = parsed?['detail'];
    if (detail is List && detail.isNotEmpty) {
      final messages = <String>[];
      for (final item in detail) {
        final itemMap = _tryJsonMapValue(item);
        final itemMessage = itemMap?['msg'];
        if (itemMessage is String && itemMessage.isNotEmpty) {
          messages.add(itemMessage);
        }
        final itemType = itemMap?['type'];
        if (code == 'http_error' && itemType is String && itemType.isNotEmpty) {
          code = itemType;
        }
      }
      if (messages.isNotEmpty) message = messages.join('; ');
    } else if (detail is String && detail.isNotEmpty) {
      message = detail;
    }

    return SystemOneApiException(
      statusCode: response.statusCode,
      code: code,
      message: message,
      responseHeaders: Map.unmodifiable(response.headers),
      rawResponseBody: response.body,
      requestId:
          _header(response.headers, 'x-request-id') ??
          _header(response.headers, 'request-id'),
      errorJson: parsed,
    );
  }

  Map<String, Object?>? _tryJsonMap(String body) {
    try {
      return _tryJsonMapValue(jsonDecode(body));
    } on Object {
      return null;
    }
  }

  Map<String, Object?>? _tryJsonMapValue(Object? value) {
    if (value is! Map) return null;
    final result = <String, Object?>{};
    for (final entry in value.entries) {
      if (entry.key is! String) return null;
      result[entry.key as String] = entry.value;
    }
    return result;
  }

  Map<String, String> _mergeHeaders(
    Map<String, String>? headers, {
    required bool hasBody,
  }) {
    final result = <String, String>{};
    void add(Map<String, String> values) {
      for (final entry in values.entries) {
        result.removeWhere(
          (key, _) => key.toLowerCase() == entry.key.toLowerCase(),
        );
        result[entry.key] = entry.value;
      }
    }

    add(defaultHeaders);
    if (headers != null) add(headers);

    result.removeWhere((key, _) => key.toLowerCase() == 'authorization');
    if (apiKey != null && apiKey!.isNotEmpty) {
      result['Authorization'] = 'Bearer ${apiKey!}';
    }
    if (hasBody && !_containsHeader(result, 'content-type')) {
      result['Content-Type'] = 'application/json';
    }
    if (!_containsHeader(result, 'accept')) {
      result['Accept'] = 'application/json';
    }
    return Map.unmodifiable(result);
  }

  Uri _endpoint(String path) {
    final basePath = _baseUri.path.replaceFirst(RegExp(r'/+$'), '');
    return _baseUri.replace(path: '$basePath$path');
  }

  String _encodeJson(Object body) {
    try {
      return jsonEncode(body);
    } on Object catch (error) {
      throw ArgumentError('Request body is not JSON encodable: $error');
    }
  }

  void _validateState(Object? state) {
    if (state is String || state is Map || state is List) return;
    throw ArgumentError.value(
      state,
      'state',
      'must be a string, object, or array',
    );
  }

  void _ensureOpen() {
    if (_closed) {
      throw StateError('SystemOneClient has been closed');
    }
  }
}

Uri _parseBaseUrl(String value) {
  final uri = Uri.tryParse(value.trim());
  if (uri == null || !uri.hasScheme || uri.host.isEmpty) {
    throw ArgumentError.value(value, 'baseUrl', 'must be an absolute HTTP URL');
  }
  if (uri.scheme != 'http' && uri.scheme != 'https') {
    throw ArgumentError.value(value, 'baseUrl', 'must use http or https');
  }
  if (uri.userInfo.isNotEmpty ||
      uri.query.isNotEmpty ||
      uri.fragment.isNotEmpty) {
    throw ArgumentError.value(
      value,
      'baseUrl',
      'must not contain credentials, query parameters, or a fragment',
    );
  }
  return uri;
}

String _validateModel(String value) {
  if (value.isEmpty) throw ArgumentError.value(value, 'defaultModel');
  return value;
}

Duration _validateTimeout(Duration value) {
  if (value <= Duration.zero) {
    throw ArgumentError.value(value, 'timeout', 'must be positive');
  }
  return value;
}

bool _containsHeader(Map<String, String> headers, String name) {
  final lower = name.toLowerCase();
  return headers.keys.any((key) => key.toLowerCase() == lower);
}

String? _header(Map<String, String> headers, String name) {
  final lower = name.toLowerCase();
  for (final entry in headers.entries) {
    if (entry.key.toLowerCase() == lower) return entry.value;
  }
  return null;
}
