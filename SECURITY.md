# Security Policy

## Scope

NeuroWeave is a research benchmark and portfolio project. It does not handle
user authentication, personal data, payments, or sensitive information.

## Reporting a vulnerability

If you find a security issue (e.g., a credential accidentally committed,
a dependency with a known CVE, or an XSS in the frontend):

1. **Do not open a public issue.**
2. Contact me directly:
   - GitHub: [@Titanium-xd](https://github.com/Titanium-xd)
   - Discord: titanium.dc
   - LinkedIn: [Parva Trivedi](https://www.linkedin.com/in/parva-trivedi/)

I will acknowledge the report within 72 hours and coordinate a fix or
disclosure timeline with you.

## Supported versions

Only the latest commit on `main` is actively maintained.

## Known non-issues

- The `lovable-error-reporting.ts` file is a no-op outside the Lovable
  editor environment. It does not transmit data in production.
- No API keys or tokens are committed. The `.env` file is gitignored.
  The `.env.example` contains only placeholder strings.
