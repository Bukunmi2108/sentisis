# 08 — Where it runs

**Status:** Accepted and deployed at https://sentisis.duckdns.org

## Context

Everything this project needs in order to be usable already fits in a container that runs
locally. A hosted URL is an extra I chose to add, so it has to earn the time it takes away from
the API, the tests, and the container.

## Decision

Two containers on the VPS I already run, behind its shared HTTPS gateway: the API and a Redis
cache. FastAPI serves both the API and `web/index.html`, mounted after the API routes so it
doesn't shadow them.

Sleep-when-idle is on. Both containers share one Sablier group, so the cache wakes with the
application rather than leaving it to run without one. A cold start measured 21 seconds, inside
the gateway's two-minute wait.

Edge liveness answers on `/healthz`, a path the application does not define, so uptime probes
never wake a sleeping container and the application's own `/health` and `/ready` reach it
unchanged.

Deployment is the last thing I do, and the first thing cut if time runs short.

## Why sleep after saying it would be off

I first wrote this note assuming a permanently running container, because a demo that takes 21
seconds to answer its first request is a worse demo. The host runs several projects, and this one
will sit idle most of the time, so paying for it continuously to save 21 seconds on the rare
first request is the wrong trade. The gateway blocks and waits rather than failing, so a caller
sees a slow response, not an error.
