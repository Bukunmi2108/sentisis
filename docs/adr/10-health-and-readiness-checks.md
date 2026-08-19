# 10 — Health and readiness checks

**Status:** Accepted

## Context

A single `/health` endpoint is the obvious thing to build. But the gateway on my host needs a
**readiness** check before it sends traffic to a container it has just started, and one endpoint
can't honestly answer both questions.

## Decision

Provide both:

- **`GET /health`** — is the process up and serving? Touches no model. Always cheap. The
  conventional liveness endpoint, behaving the way that name implies.
- **`GET /ready`** — is the model loaded and working? Runs one small test prediction. This is what
  the gateway checks before routing traffic.
