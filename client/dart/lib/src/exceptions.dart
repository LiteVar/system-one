/// Base class for failures raised by the System One client.
abstract class SystemOneException implements Exception {
  SystemOneException(this.message, {this.cause, this.stackTrace});

  final String message;
  final Object? cause;
  final StackTrace? stackTrace;

  @override
  String toString() => '$runtimeType: $message';
}

/// An HTTP response with a non-success status code.
class SystemOneApiException extends SystemOneException {
  SystemOneApiException({
    required this.statusCode,
    required this.code,
    required String message,
    required this.responseHeaders,
    required this.rawResponseBody,
    this.requestId,
    this.errorJson,
    Object? cause,
    StackTrace? stackTrace,
  }) : super(message, cause: cause, stackTrace: stackTrace);

  final int statusCode;
  final String code;
  final Map<String, String> responseHeaders;
  final String rawResponseBody;
  final String? requestId;
  final Map<String, Object?>? errorJson;

  /// Alias that makes the raw body name explicit at call sites.
  String get rawBody => rawResponseBody;

  @override
  String toString() {
    final request = requestId == null ? '' : ' request_id=$requestId';
    return '$runtimeType($statusCode, $code$request): $message';
  }
}

/// The server could not be reached or the transport failed before a response.
class SystemOneConnectionException extends SystemOneException {
  SystemOneConnectionException({
    required String message,
    this.uri,
    Object? cause,
    StackTrace? stackTrace,
  }) : super(message, cause: cause, stackTrace: stackTrace);

  final Uri? uri;
}

/// An HTTP attempt exceeded its configured timeout.
class SystemOneTimeoutException extends SystemOneConnectionException {
  SystemOneTimeoutException({
    required this.timeout,
    required Uri uri,
    super.cause,
    super.stackTrace,
  }) : super(
         message: 'Request to $uri timed out after ${timeout.inMilliseconds}ms',
         uri: uri,
       );

  final Duration timeout;
}

/// The caller's cancellation future completed while an operation was pending.
class SystemOneAbortException extends SystemOneException {
  SystemOneAbortException({this.uri, String? message})
    : super(message ?? 'The System One request was cancelled');

  final Uri? uri;
}

/// A successful HTTP response did not match the expected JSON contract.
class SystemOneResponseFormatException extends SystemOneException {
  SystemOneResponseFormatException({
    required String message,
    this.uri,
    this.fieldPath,
    Object? cause,
    StackTrace? stackTrace,
  }) : super(message, cause: cause, stackTrace: stackTrace);

  final Uri? uri;
  final String? fieldPath;
}
