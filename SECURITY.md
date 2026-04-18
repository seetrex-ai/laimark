# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

- **Email**: jesus@tabares.eu
- **Expected response**: Within 72 hours
- **Please do NOT** open a public GitHub issue for security vulnerabilities

## Untrusted Code Execution

This repository executes model-generated Python code as part of the training and verification pipeline. The sandbox is a subprocess with a 5-second timeout, which is sufficient for HumanEval-style code but is **not** a hardened isolation boundary against deliberately adversarial payloads. Run the pipeline inside a Docker container, a disposable VM, or a sandboxed user account on a machine without sensitive data or privileged network access.
