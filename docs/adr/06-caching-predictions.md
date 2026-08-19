# 06 — Caching predictions

**Status:** Accepted. Cache hit ratio is exposed on `/metrics`.

## Context

Repeat predictions should be fast, and this API gets a very particular kind of traffic: the
demo page and anyone trying it out send the same handful of example strings over and over.

Two facts about how it runs shape the choice. It serves under uvicorn with more than one worker,
and it gets restarted on every deploy.

## Decision

Redis, running as a service in `docker-compose.yml`. Entries are keyed on the **cleaned** text
and carry a TTL.

If Redis is unreachable the API logs it and runs the model anyway. A cache failure never fails a
request.

## Why

Redis gives every worker one shared cache.
It also survives restarts. Redis stays warm across restarts.