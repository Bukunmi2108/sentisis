# 08 — Where it runs

**Status:** Accepted

## Context

Everything this project needs in order to be usable already fits in a container that runs
locally. A hosted URL is an extra I chose to add, so it has to earn the time it takes away from
the API, the tests, and the container.

## Decision

One container, on the VPS I already run, behind its shared HTTPS gateway. FastAPI serves both the
API and `web/index.html`, mounted after the API routes so it doesn't shadow them. Sleep-when-idle
is off. Deployment is the last thing I do, and the first thing cut if time runs short.
