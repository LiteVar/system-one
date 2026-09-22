import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:system_one/system_one.dart';
import 'package:test/test.dart';

void main() {
  group('request serialization', () {
    late _TestServer server;

    setUp(() async {
      server = _TestServer((request) async {
        await _writeJson(request, _resultBody());
      });
      await server.start();
    });

    tearDown(() => server.close());

    test('serializes all question types and structured values', () async {
      final client = SystemOneClient(
        baseUrl: server.baseUrl.toString(),
        retry: _noRetry(),
      );
      addTearDown(client.close);

      await client.systemOne(
        state: <String, Object?>{
          'message': 'hello',
          'items': <Object?>[1, true],
        },
        questions: {
          'choice_map': SystemOneQuestion.choice(
            instructions: <String, Object?>{'prompt': 'pick'},
            criteria: <String, Object?>{'one': 'first', 'two': null},
          ),
          'choice_list': SystemOneQuestion.choice(
            instructions: <Object?>['pick', 'one'],
            criteria: <Object?>[
              'one',
              <String, Object?>{'name': 'two'},
            ],
          ),
          'score': SystemOneQuestion.score(
            instructions: 'rate it',
            criteria: <Object?>['low', 'medium', 'high'],
          ),
          'noul': SystemOneQuestion.noul(
            criteria: <String, Object?>{'false': 'no', 'true': 'yes'},
          ),
        },
      );

      expect(server.requests, 1);
      expect(server.lastBody['model'], 'laya-multilingual');
      expect(server.lastBody['state'], isA<Map>());
      expect(server.lastBody['questions']['choice_map']['criteria'], {
        'one': 'first',
        'two': null,
      });
      expect(server.lastBody['questions']['choice_list']['criteria'], [
        'one',
        {'name': 'two'},
      ]);
      expect(server.lastBody['questions']['score']['criteria'], [
        'low',
        'medium',
        'high',
      ]);
      expect(server.lastBody['questions']['noul']['criteria'], {
        'false': 'no',
        'true': 'yes',
      });
    });

    test('supports string and array state roots', () async {
      final client = SystemOneClient(
        baseUrl: server.baseUrl.toString(),
        retry: _noRetry(),
      );
      addTearDown(client.close);
      final questions = <String, SystemOneQuestion>{
        'q': SystemOneQuestion.noul(),
      };

      await client.systemOne(state: 'hello', questions: questions);
      expect(server.lastBody['state'], 'hello');
      await client.systemOne(
        state: <Object?>['hello', 1],
        questions: questions,
      );
      expect(server.lastBody['state'], ['hello', 1]);
    });
  });

  test('parses choice, score, noul, action, and unknown fields', () {
    final result = SystemOneResult.fromJson({
      'model': 'laya-multilingual',
      'answers': {
        'intent': {
          'type': 'choice',
          'choice': 'execute',
          'probabilities': {'execute': 0.8, 'reject': 0.2},
          'confidence': 0.72,
          'action': {'act_probability': 0.91, 'future_action_field': true},
          'future_answer_field': {'kept': true},
        },
        'risk': {
          'type': 'score',
          'score': 1.2,
          'legend': {'0': 'low', '1': 'medium', '2': 'high'},
          'probabilities': {'0': 0.6, '1': 0.3, '2': 0.1},
          'confidence': 0.54,
          'action': {'act_probability': 0.42},
        },
        'approved': {
          'type': 'noul',
          'noul': 0.87,
          'confidence': 0.87,
          'action': {'act_probability': 0.75},
        },
      },
      'usage': {'input_tokens': 100, 'output_tokens': 0},
      'future_root_field': 'kept',
    });

    final choice = result.answers['intent']! as SystemOneChoiceAnswer;
    final score = result.answers['risk']! as SystemOneScoreAnswer;
    final noul = result.answers['approved']! as SystemOneNoulAnswer;
    expect(choice.choice, 'execute');
    expect(choice.action!.actProbability, 0.91);
    expect(choice.raw['future_answer_field'], {'kept': true});
    expect(score.legend['2'], 'high');
    expect(score.action!.actProbability, 0.42);
    expect(noul.noul, 0.87);
    expect(noul.confidence, 0.87);
    expect(noul.action!.actProbability, 0.75);
    expect(result.raw['future_root_field'], 'kept');
  });

  test('parses a standard Jev noul answer without local extensions', () {
    final result = SystemOneResult.fromJson({
      'model': 'jev-1.13.0',
      'answers': {
        'approved': {'type': 'noul', 'noul': 0.98},
      },
      'usage': {'input_tokens': 3, 'output_tokens': 0},
    });

    final answer = result.answers['approved']! as SystemOneNoulAnswer;
    expect(answer.noul, 0.98);
    expect(answer.confidence, isNull);
    expect(answer.action, isNull);
  });

  test('parses healthy and degraded health responses', () async {
    final responses = <Map<String, Object?>>[
      {
        'status': 'ok',
        'process': 'alive',
        'model': 'loaded',
        'model_id': 'laya-multilingual',
        'runtime': {
          'runtime': 'llama.cpp',
          'source': 'managed',
          'version': 'b10964',
          'path': '/tmp/libllama.dylib',
          'platform': 'macos',
          'architecture': 'arm64',
          'backend': 'metal',
          'compatible': true,
          'error': null,
          'future': 'kept',
        },
        'error': null,
      },
      {
        'status': 'degraded',
        'process': 'alive',
        'model': 'unavailable',
        'model_id': 'laya-multilingual',
        'runtime': null,
        'error': 'missing model',
      },
    ];
    var index = 0;
    final server = _TestServer((request) async {
      await _writeJson(request, responses[index++]);
    });
    await server.start();
    addTearDown(server.close);
    final client = SystemOneClient(
      baseUrl: server.baseUrl.toString(),
      retry: _noRetry(),
    );
    addTearDown(client.close);

    final healthy = await client.health();
    expect(healthy.isHealthy, isTrue);
    expect(healthy.runtime!.backend, 'metal');
    expect(healthy.runtime!.raw['future'], 'kept');

    final degraded = await client.health();
    expect(degraded.isHealthy, isFalse);
    expect(degraded.status, 'degraded');
    expect(degraded.error, 'missing model');
  });

  test('parses standard and local models response shapes', () async {
    final payloads = <Map<String, Object?>>[
      {
        'models': [
          {
            'name': 'jev-latest',
            'description': 'General-purpose model',
            'release_date': '2026-09-15',
          },
        ],
      },
      {
        'data': [
          {
            'id': 'laya-multilingual',
            'loaded': true,
            'runtime': {'runtime': 'llama.cpp', 'backend': 'cpu'},
            'error': null,
          },
        ],
      },
    ];
    var index = 0;
    final server = _TestServer((request) async {
      await _writeJson(request, payloads[index++]);
    });
    await server.start();
    addTearDown(server.close);
    final client = SystemOneClient(
      baseUrl: server.baseUrl.toString(),
      retry: _noRetry(),
    );
    addTearDown(client.close);

    final standard = await client.models();
    expect(standard.models.single.id, 'jev-latest');
    expect(standard.models.single.releaseDate, '2026-09-15');
    expect(standard.data, isEmpty);

    final local = await client.models();
    expect(local.models.single.id, 'laya-multilingual');
    expect(local.data.single.loaded, isTrue);
    expect(local.data.single.runtime!.backend, 'cpu');
  });

  test('parses local and standard API errors', () async {
    final errors = <({int status, Object body, String code})>[
      (
        status: 400,
        body: {
          'error': {'code': 'invalid_json', 'message': 'invalid JSON'},
        },
        code: 'invalid_json',
      ),
      (
        status: 422,
        body: {
          'detail': [
            {
              'loc': ['body', 'questions'],
              'msg': 'Field required',
              'type': 'missing',
            },
          ],
        },
        code: 'missing',
      ),
      (
        status: 503,
        body: {
          'error': {
            'code': 'model_unavailable',
            'message': 'model unavailable',
          },
        },
        code: 'model_unavailable',
      ),
      (
        status: 500,
        body: {
          'error': {'code': 'internal_error', 'message': 'internal'},
        },
        code: 'internal_error',
      ),
    ];
    var index = 0;
    final server = _TestServer((request) async {
      final error = errors[index++];
      await _writeJson(request, error.body, status: error.status);
    });
    await server.start();
    addTearDown(server.close);
    final client = SystemOneClient(
      baseUrl: server.baseUrl.toString(),
      retry: _noRetry(),
    );
    addTearDown(client.close);

    for (final expected in errors) {
      try {
        await client.health();
        fail('expected ${expected.status}');
      } on SystemOneApiException catch (error) {
        expect(error.statusCode, expected.status);
        expect(error.code, expected.code);
        expect(error.responseHeaders, containsPair('x-test-request', 'yes'));
        expect(error.rawResponseBody, isNotEmpty);
      }
    }
  });

  test('retries 503 and honors retry-after', () async {
    var requests = 0;
    final server = _TestServer((request) async {
      requests++;
      if (requests == 1) {
        await _writeJson(
          request,
          {
            'error': {'code': 'model_unavailable', 'message': 'try again'},
          },
          status: 503,
          headers: {'retry-after': '0'},
        );
      } else {
        await _writeJson(request, _healthBody());
      }
    });
    await server.start();
    addTearDown(server.close);
    final client = SystemOneClient(
      baseUrl: server.baseUrl.toString(),
      retry: SystemOneRetryPolicy(
        maxRetries: 1,
        backoffInitial: Duration.zero,
        backoffMax: Duration.zero,
        backoffJitter: 0,
      ),
    );
    addTearDown(client.close);

    expect((await client.health()).status, 'ok');
    expect(requests, 2);
  });

  test('supports timeout and cancellation', () async {
    final server = _TestServer((request) async {
      await Future<void>.delayed(const Duration(milliseconds: 120));
      try {
        await _writeJson(request, _healthBody());
      } on Object {
        // The client may have timed out or cancelled the socket.
      }
    });
    await server.start();
    addTearDown(server.close);

    final timeoutClient = SystemOneClient(
      baseUrl: server.baseUrl.toString(),
      timeout: const Duration(milliseconds: 20),
      retry: _noRetry(),
    );
    addTearDown(timeoutClient.close);
    expect(timeoutClient.health(), throwsA(isA<SystemOneTimeoutException>()));

    final cancellationClient = SystemOneClient(
      baseUrl: server.baseUrl.toString(),
      retry: _noRetry(),
    );
    addTearDown(cancellationClient.close);
    expect(
      cancellationClient.health(
        cancellation: Future<void>.delayed(const Duration(milliseconds: 20)),
      ),
      throwsA(isA<SystemOneAbortException>()),
    );
  });

  test('does not send an authorization header for an empty API key', () async {
    final server = _TestServer((request) async {
      await _writeJson(request, _healthBody());
    });
    await server.start();
    addTearDown(server.close);
    final client = SystemOneClient(
      baseUrl: '${server.baseUrl}/prefix/',
      apiKey: '',
      retry: _noRetry(),
    );
    addTearDown(client.close);

    await client.health(headers: {'X-Test': 'yes'});
    expect(server.lastHeaders.keys, isNot(contains('authorization')));
    expect(server.lastPath, '/prefix/health');
  });

  test('uses bearer auth and respects injected client ownership', () async {
    final server = _TestServer((request) async {
      await _writeJson(request, _healthBody());
    });
    await server.start();
    addTearDown(server.close);

    final transport = _TrackingClient();
    final client = SystemOneClient(
      baseUrl: server.baseUrl.toString(),
      apiKey: 'secret',
      httpClient: transport,
      closeClient: false,
      retry: _noRetry(),
    );
    await client.health();
    expect(server.lastHeaders['authorization'], 'Bearer secret');
    client.close();
    expect(transport.wasClosed, isFalse);
    transport.close();
    expect(transport.wasClosed, isTrue);
  });
}

SystemOneRetryPolicy _noRetry() => SystemOneRetryPolicy(
  maxRetries: 0,
  backoffInitial: Duration.zero,
  backoffMax: Duration.zero,
  backoffJitter: 0,
);

Map<String, Object?> _resultBody() => {
  'model': 'laya-multilingual',
  'answers': {
    'q': {'type': 'noul', 'noul': 0.5},
  },
  'usage': {'input_tokens': 1, 'output_tokens': 0},
};

Map<String, Object?> _healthBody() => {
  'status': 'ok',
  'process': 'alive',
  'model': 'loaded',
  'model_id': 'laya-multilingual',
  'runtime': null,
  'error': null,
};

Future<void> _writeJson(
  HttpRequest request,
  Object body, {
  int status = 200,
  Map<String, String> headers = const {},
}) async {
  request.response
    ..statusCode = status
    ..headers.contentType = ContentType.json
    ..headers.add('x-test-request', 'yes');
  for (final entry in headers.entries) {
    request.response.headers.set(entry.key, entry.value);
  }
  request.response.write(jsonEncode(body));
  await request.response.close();
}

class _TestServer {
  _TestServer(this.handler);

  final Future<void> Function(HttpRequest request) handler;
  late HttpServer _server;
  int requests = 0;
  Map<String, dynamic> lastBody = <String, dynamic>{};
  Map<String, String> lastHeaders = <String, String>{};
  String lastPath = '';

  Uri get baseUrl => Uri.parse('http://127.0.0.1:${_server.port}');

  Future<void> start() async {
    _server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    _server.listen((request) async {
      requests++;
      lastPath = request.uri.path;
      final capturedHeaders = <String, String>{};
      request.headers.forEach((key, values) {
        capturedHeaders[key] = values.join(',');
      });
      lastHeaders = capturedHeaders;
      if (request.method == 'POST') {
        final text = await utf8.decoder.bind(request).join();
        lastBody = jsonDecode(text) as Map<String, dynamic>;
      }
      await handler(request);
    });
  }

  Future<void> close() => _server.close(force: true);
}

class _TrackingClient extends http.BaseClient {
  final http.Client _delegate = http.Client();
  bool wasClosed = false;

  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) {
    return _delegate.send(request);
  }

  @override
  void close() {
    wasClosed = true;
    _delegate.close();
  }
}
