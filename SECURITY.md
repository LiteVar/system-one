# Security policy

## Supported versions

| Version | Supported |
| --- | --- |
| `0.1.x` | Yes, during the alpha period |

## Reporting a vulnerability

Please do not disclose exploitable vulnerabilities in a public issue. After this
repository is published on GitHub, use a private GitHub Security Advisory. Until
then, send a private report to the project maintainers and include:

- affected version or commit;
- operating system and architecture;
- exact reproduction steps or a minimal proof of concept;
- impact and any suggested mitigation.

The default server binds to `127.0.0.1`. If it is deliberately exposed on another
interface, put it behind the application's authentication, TLS, rate limiting, and
network access controls. System One does not provide authentication in the local
runtime.

Model and runtime downloads are verified against pinned SHA-256 values. Do not
replace those values or run untrusted runtime libraries in a privileged process.
