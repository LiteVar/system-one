import 'package:system_one/system_one.dart';

Future<void> main() async {
  final client = SystemOneClient();
  try {
    final result = await client.systemOne(
      state: 'The user wants to restart nginx.',
      questions: {
        'intent': SystemOneQuestion.choice(
          instructions: 'What should happen next?',
          criteria: {
            'execute': 'Execute the restart',
            'confirm': 'Ask for confirmation',
            'reject': 'Reject the request',
          },
        ),
      },
    );
    print(result.raw);
  } finally {
    client.close();
  }
}
