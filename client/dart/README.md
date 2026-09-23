# system_one

A pure Dart HTTP client for the local System One runtime and Jev-style
`/v1/systemone` APIs. It does not depend on Flutter or `jev_dart`, and it does
not contain a model, runtime library, tokenizer, or model weights.

The client keeps the wire format business-neutral: requests use `state`,
`model`, and `questions`, while the Dart API uses `SystemOne*` names. It can
read both the standard `models` response and the local runtime's `data`
response, and it preserves raw JSON for forward compatibility.

## Install

```yaml
dependencies:
  system_one: ^0.1.1
```

Import the package with:

```dart
import 'package:system_one/system_one.dart';
```

## Start the local runtime

From the repository root:

```sh
cargo run -- model pull laya-multilingual
cargo run -- serve
```

The default local endpoint is `http://127.0.0.1:8080`. The Dart package never
downloads or bundles the model assets.

## Evaluate questions

```dart
import 'package:system_one/system_one.dart';

Future<void> main() async {
  final client = SystemOneClient();
  try {
    final result = await client.systemOne(
      state: {
        'message': 'The user wants to restart nginx.',
        'attempts': 1,
      },
      questions: {
        'intent': SystemOneQuestion.choice(
          instructions: 'What should happen next?',
          criteria: {
            'execute': 'Execute the restart',
            'confirm': 'Ask for confirmation',
            'reject': 'Reject the request',
          },
        ),
        'approved': SystemOneQuestion.noul(
          instructions: 'Is the requested action approved?',
        ),
      },
    );

    final intent = result.answers['intent'] as SystemOneChoiceAnswer;
    print(intent.choice);
    print(intent.probabilities);
  } finally {
    client.close();
  }
}
```

`SystemOneQuestion.choice` accepts either a label-to-description map or a
list. `score` accepts an ordered list, and `noul` accepts optional `false` and
`true` criteria. State and instructions may be strings, objects, or arrays.

## Health and models

```dart
final health = await client.health();
if (!health.isHealthy) {
  print('runtime is ${health.status}: ${health.error}');
}

final models = await client.models();
for (final model in models.models) {
  print('${model.id}: loaded=${model.loaded}');
}
```

Health status is meaningful even when the HTTP status is 200: a model that is
not available is reported as `degraded`.

## Timeout, retry, and cancellation

Timeout applies to each HTTP attempt. `maxRetries` counts retries after the
initial attempt. By default, connection errors, timeouts, 408, 429, and 5xx
responses are retried; deterministic 400/422 request errors are not.

```dart
final client = SystemOneClient(
  timeout: const Duration(seconds: 5),
  retry: SystemOneRetryPolicy(
    maxRetries: 3,
    backoffInitial: const Duration(milliseconds: 200),
    respectRetryAfter: true,
  ),
);

final cancellation = Future<void>.delayed(const Duration(seconds: 2));
try {
  await client.health(cancellation: cancellation);
} on SystemOneAbortException {
  print('cancelled');
} finally {
  client.close();
}
```

With the generic `package:http` transport, cancellation stops waiting and
prevents further retries. A custom transport may still finish an underlying
request that was already in flight.

## Standard Jev-compatible endpoint

The same client can target a standard Jev endpoint by changing the base URL,
model, and API key:

```dart
final client = SystemOneClient(
  baseUrl: 'https://api.typesafe.ai',
  apiKey: const String.fromEnvironment('TYPESAFE_API_KEY'),
  defaultModel: 'jev-latest',
);
```

When `apiKey` is omitted or empty, the client does not send an
`Authorization` header. The local System One service therefore works without
authentication, while a hosted Jev endpoint can use bearer authentication.

## Public types

The main public types are `SystemOneClient`, `SystemOneQuestion`,
`SystemOneResult`, `SystemOneChoiceAnswer`, `SystemOneScoreAnswer`,
`SystemOneNoulAnswer`, `SystemOneHealth`, `SystemOneModelsResponse`,
`SystemOneModelCard`, `SystemOneRuntimeInfo`, `SystemOneRetryPolicy`, and the
`SystemOne*Exception` classes.
