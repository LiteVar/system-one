import 'dart:math';

/// Controls retries for transient transport and HTTP failures.
class SystemOneRetryPolicy {
  SystemOneRetryPolicy({
    this.maxRetries = 2,
    this.backoffInitial = const Duration(milliseconds: 200),
    this.backoffMax = const Duration(seconds: 2),
    this.backoffJitter = 0.2,
    Set<int>? httpStatuses,
    this.respectRetryAfter = true,
    this.maxRetryAfter = const Duration(seconds: 30),
    this.retryConnectionErrors = true,
    this.retryTimeouts = true,
  }) : httpStatuses = Set.unmodifiable(httpStatuses ?? _defaultHttpStatuses) {
    if (maxRetries < 0) {
      throw ArgumentError.value(
        maxRetries,
        'maxRetries',
        'must be non-negative',
      );
    }
    if (backoffInitial.isNegative) {
      throw ArgumentError.value(
        backoffInitial,
        'backoffInitial',
        'must not be negative',
      );
    }
    if (backoffMax.isNegative || backoffMax < backoffInitial) {
      throw ArgumentError.value(
        backoffMax,
        'backoffMax',
        'must be non-negative and at least backoffInitial',
      );
    }
    if (backoffJitter < 0 || backoffJitter > 1) {
      throw ArgumentError.value(
        backoffJitter,
        'backoffJitter',
        'must be between 0 and 1',
      );
    }
    if (maxRetryAfter.isNegative) {
      throw ArgumentError.value(
        maxRetryAfter,
        'maxRetryAfter',
        'must not be negative',
      );
    }
  }

  factory SystemOneRetryPolicy.defaults() => SystemOneRetryPolicy();

  static const Set<int> _defaultHttpStatuses = <int>{
    408,
    429,
    500,
    501,
    502,
    503,
    504,
    505,
    506,
    507,
    508,
    509,
    510,
    511,
  };

  final int maxRetries;
  final Duration backoffInitial;
  final Duration backoffMax;
  final double backoffJitter;
  final Set<int> httpStatuses;
  final bool respectRetryAfter;
  final Duration maxRetryAfter;
  final bool retryConnectionErrors;
  final bool retryTimeouts;

  bool shouldRetryStatus(int statusCode) => httpStatuses.contains(statusCode);

  /// Computes the delay before [retryNumber], where the first retry is 1.
  ///
  /// `Retry-After` is treated as a minimum delay and is capped by
  /// [maxRetryAfter]. Jitter is a positive proportional addition to the
  /// exponential backoff, so a zero jitter value is deterministic in tests.
  Duration delayFor({
    required int retryNumber,
    Map<String, String>? headers,
    DateTime? now,
    Random? random,
  }) {
    if (retryNumber < 1) {
      throw ArgumentError.value(
        retryNumber,
        'retryNumber',
        'must be at least 1',
      );
    }

    var milliseconds = backoffInitial.inMilliseconds.toDouble();
    for (var i = 1; i < retryNumber; i++) {
      milliseconds = min(
        backoffMax.inMilliseconds.toDouble(),
        milliseconds * 2,
      );
    }
    milliseconds = min(backoffMax.inMilliseconds.toDouble(), milliseconds);

    if (backoffJitter > 0 && milliseconds > 0) {
      final source = random ?? Random();
      milliseconds += milliseconds * backoffJitter * source.nextDouble();
      milliseconds = min(backoffMax.inMilliseconds.toDouble(), milliseconds);
    }

    final retryAfter = respectRetryAfter
        ? _parseRetryAfter(headers, now: now ?? DateTime.now())
        : null;
    if (retryAfter != null) {
      final capped = min(
        retryAfter.inMilliseconds,
        maxRetryAfter.inMilliseconds,
      );
      milliseconds = max(milliseconds, capped.toDouble());
    }
    return Duration(milliseconds: milliseconds.round());
  }

  Duration? _parseRetryAfter(
    Map<String, String>? headers, {
    required DateTime now,
  }) {
    if (headers == null) return null;
    String? value;
    for (final entry in headers.entries) {
      if (entry.key.toLowerCase() == 'retry-after') {
        value = entry.value.trim();
        break;
      }
    }
    if (value == null || value.isEmpty) return null;

    final seconds = int.tryParse(value);
    if (seconds != null && seconds >= 0) {
      return Duration(seconds: seconds);
    }

    final date = _parseHttpDate(value);
    if (date == null) return null;
    final difference = date.difference(now);
    return difference.isNegative ? Duration.zero : difference;
  }

  DateTime? _parseHttpDate(String value) {
    final match = RegExp(
      r'^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+(\d{2})\s+'
      r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+'
      r'(\d{4})\s+(\d{2}):(\d{2}):(\d{2})\s+GMT$',
    ).firstMatch(value);
    if (match == null) return null;

    const months = <String, int>{
      'Jan': 1,
      'Feb': 2,
      'Mar': 3,
      'Apr': 4,
      'May': 5,
      'Jun': 6,
      'Jul': 7,
      'Aug': 8,
      'Sep': 9,
      'Oct': 10,
      'Nov': 11,
      'Dec': 12,
    };
    try {
      return DateTime.utc(
        int.parse(match.group(3)!),
        months[match.group(2)]!,
        int.parse(match.group(1)!),
        int.parse(match.group(4)!),
        int.parse(match.group(5)!),
        int.parse(match.group(6)!),
      );
    } on FormatException {
      return null;
    }
  }
}
