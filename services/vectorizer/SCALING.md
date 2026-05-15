# Scaling the vectorizer service

How to think about — and operate — horizontal scaling for the vectorizer (and any future worker that follows the same pattern: re-ranker, etc.).

## TL;DR

The workers are stateless and compete over a shared `background_tasks` queue via an atomic claim RPC. "Scaling up" means running more replicas of the same container; the code does not change. The only real questions are *what platform runs them* and *what signal triggers more replicas*.

---

## Why scaling is "just run more containers"

The workers are designed to be stateless and competing:

- A new task triggers a Realtime notification → *all* subscribed workers wake up.
- They race into `claim_background_task` — the RPC is atomic (SKIP LOCKED under the hood), so exactly one wins. The losers go back to sleep with no harm done.
- No coordinator, no leader election, no service discovery.

So "spawn another worker" literally means: run another container with valid env vars (or its own JWT) pointed at the same Supabase. It joins the pool, starts pulling work, and the queue distributes naturally.

## Three stages of "spawning more"

### Stage 1 — manual (PoC)

On the NUC, the simplest crude lever:

```bash
docker compose up -d --scale vectorizer=5
```

Five worker replicas in the same compose project. Scale up/down at any time without restarting the others. Every replica holds a Realtime websocket and PostgREST HTTPS connections — don't go to 100 replicas without thinking about Supabase's connection limits.

### Stage 2 — orchestrator (production)

Move the container to a platform that can adjust replica count programmatically. Sensible choices, roughly ordered by operational comfort:

- **Fly.io machines** — simplest, autoscaling by metric or HTTP load.
- **AWS ECS Fargate** — managed containers, autoscaling on CloudWatch metrics.
- **Google Cloud Run jobs** — scales to zero, pay per CPU-second, good for spiky workloads.
- **Kubernetes** — most flexible, most ceremony.

All let you set min/max replica count and a scaling policy. The interesting question is *what signal* they scale on.

### Stage 3 — autoscaling on the right signal

For ML workers, the textbook signals (CPU, memory, request rate) are mostly *wrong* — workers are either pinned at 100% during a job or completely idle. What you actually want to scale on is **queue depth**:

```sql
SELECT count(*) FROM background_tasks
WHERE kind = 'vectorize' AND status = 'pending';
```

Rules of the form *"target: each worker has ≤ 2 pending tasks waiting on it"* give good utilization without bottleneck.

The cleanest tool for this is **KEDA** (Kubernetes Event-Driven Autoscaling) — built-in Postgres scaler that polls a query and adjusts deployment replica count. Can scale **to zero** when the queue is empty (no idle cost) and back up the moment a task lands. Outside Kubernetes, you'd build the equivalent with a small lambda/edge function that polls the queue and adjusts ECS/Cloud Run replicas.

## Constraints to know before scaling aggressively

1. **Supabase connection budget.** Each worker holds a Realtime websocket plus PostgREST HTTPS connections. On managed Supabase this is a real ceiling (varies by tier). Self-hosted: bound by Postgres `max_connections`. Measure before deciding on an upper limit.
2. **Compute substrate.** CPU-only workers (MiniLM and similar) scale freely on cheap instances. Larger models or GPU-bound work is bounded by GPU availability — scaling beyond GPU count doesn't help.
3. **Cold starts.** Spinning up a fresh worker takes 10–60 seconds (container pull + model load). For bursty workloads, keep `min_replicas ≥ 1` so the first task doesn't wait for a cold boot.
4. **Per-job granularity.** A "job" is currently "vectorize this project's whole corpus" — adding workers helps across *projects*, not within one. To parallelize *inside* a project (e.g. split 100k pubs across 10 workers), chunk the work at job-creation time. Not needed at PoC scale.

## Why the code already supports all of this

The worker is built with these properties in mind:

- **Stateless processing** — no in-memory state about specific tasks; restart-safe.
- **Atomic claim** — `claim_background_task` ensures no double-processing under concurrency.
- **Filtered Realtime subscription** — workers only wake for their own `kind`, so different worker types can run in the same Supabase without interfering.
- **Graceful SIGTERM** — finishes the current task before exiting, so the orchestrator can scale down safely.
- **Idempotent upserts** — `publication_embeddings` rows are keyed on `(publication_id, model_name)`, so retries don't duplicate.

No code changes are needed to scale — only deployment decisions.

## Practical recommendation

- **Now (PoC)**: `docker compose up --scale vectorizer=N` with N picked manually based on load. Genuinely fine.
- **Cloud production**: Kubernetes + KEDA polling `background_tasks` is the most architecturally aligned answer (queue-aware, scale-to-zero, no extra infrastructure to babysit). Cloud Run jobs is the simplest non-K8s equivalent.
- **Long-term**: separate worker pools by `kind` (e.g. `vectorize` vs `rerank`) so each can scale on its own queue depth and run on its own compute substrate (CPU vs GPU).
