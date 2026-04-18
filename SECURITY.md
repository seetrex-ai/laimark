# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

- **Email**: research@seetrex.com
- **Expected response**: within 72 hours
- **Please do NOT** open a public GitHub issue for security vulnerabilities.

## Untrusted Code Execution

The training and verification pipeline executes model-generated Python code. The sandbox is a Python subprocess with a five-second timeout. It works for HumanEval-style code but does not block the executed code from reading or writing the filesystem, opening network sockets, importing arbitrary modules, or reading environment variables (including API keys loaded at process start).

Run the pipeline inside a container or disposable VM. On a machine with sensitive files, cloud credentials, or a usable outbound network, the sandbox is not enough on its own.
