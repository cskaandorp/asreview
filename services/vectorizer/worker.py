#!/usr/bin/env python3
"""Dense embedding vectorization worker (Supabase-mediated).

Uses `mixedbread-ai/mxbai-embed-large-v1` — ASReview's ELAS h3 feature
extractor, the cheapest of the blessed dense embedders (335M params, 1024-dim
output). Each publication is encoded independently, so adding new publications
later only requires embedding the new ones; existing embeddings stay valid.

Talks to Supabase over HTTPS:
  - Realtime WebSocket subscription to background_tasks INSERTs (push).
  - PostgREST RPC `claim_background_task` to atomically claim work.
  - PostgREST table API to read publications and upsert embeddings.

No direct Postgres connection. Auth is a JWT carrying role=worker_external.

Required env vars:
  SUPABASE_URL          e.g. https://supa.elixlab.nl
  SUPABASE_KEY          JWT signed with role=worker_external (or service_role)

Optional env vars:
  WORKER_KIND           background_tasks.kind to handle (default: 'vectorize')
  MODEL_NAME            stored in publication_embeddings.model_name (default: 'mxbai')
  HF_MODEL_ID           Hugging Face repo id (default: mixedbread-ai/mxbai-embed-large-v1)
  ENCODE_BATCH_SIZE     batch size for SentenceTransformer.encode (default: 32)
  SAFETY_POLL_SECONDS   safety-net poll interval (default: 30)
  SIMULATED_WORK_SECONDS  optional sleep before mark_completed, for testing (default: 0)
  DRY_RUN               if 'true', encode but skip upsert (default: false)
"""

import asyncio
import os
import signal
import sys
import traceback
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from supabase import acreate_client, AsyncClient

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

WORKER_KIND = os.environ.get("WORKER_KIND", "vectorize")
MODEL_NAME = os.environ.get("MODEL_NAME", "mxbai")
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "mixedbread-ai/mxbai-embed-large-v1")
ENCODE_BATCH_SIZE = int(os.environ.get("ENCODE_BATCH_SIZE", 32))
SAFETY_POLL_SECONDS = int(os.environ.get("SAFETY_POLL_SECONDS", 30))
SIMULATED_WORK_SECONDS = int(os.environ.get("SIMULATED_WORK_SECONDS", 0))
DRY_RUN = os.environ.get("DRY_RUN", "false").lower() == "true"
UPSERT_BATCH_SIZE = 200

_wake = asyncio.Event()
_shutdown = asyncio.Event()

# Load the embedding model once at module level. Blocks Python startup until
# the weights are loaded (~10s if cached, ~1–3 min on first download).
print(f"loading model {HF_MODEL_ID} ...", flush=True)
_embedder = SentenceTransformer(HF_MODEL_ID)
EMBED_DIM = _embedder.get_embedding_dimension()
print(f"model loaded; embed dim={EMBED_DIM}", flush=True)


def to_pgvector_literal(vec: np.ndarray) -> str:
    """1-D float array -> pgvector dense literal '[v1,v2,...]'."""
    return "[" + ",".join(f"{float(v):.6f}" for v in vec) + "]"


def encode(pubs: list[dict]) -> np.ndarray:
    texts = [
        " ".join(t for t in (p.get("title"), p.get("abstract")) if t)
        for p in pubs
    ]
    # normalize_embeddings=True mirrors ASReview's `{"normalize": True}` config
    # for h3; it makes cosine similarity equivalent to inner product.
    return _embedder.encode(
        texts,
        batch_size=ENCODE_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


async def claim_task(sb: AsyncClient) -> dict | None:
    res = await sb.rpc("claim_background_task", {"p_kind": WORKER_KIND}).execute()
    data = res.data
    if not data:
        return None
    # PostgREST returns a single row as an object, but some versions wrap in a list.
    return data[0] if isinstance(data, list) else data


async def fetch_publications(sb: AsyncClient, project_id: str) -> list[dict]:
    page_size = 1000
    out: list[dict] = []
    start = 0
    while True:
        res = (
            await sb.table("publications")
            .select("id,title,abstract")
            .eq("project_id", project_id)
            .order("id")
            .range(start, start + page_size - 1)
            .execute()
        )
        rows = res.data or []
        out.extend(rows)
        if len(rows) < page_size:
            return out
        start += page_size


async def upsert_embeddings(
    sb: AsyncClient,
    pubs: list[dict],
    embeddings: np.ndarray,
    model_name: str,
    dim: int,
) -> None:
    # Schema is dense-only since the drop-sparse migration; columns are
    # publication_id, model_name, dim, dense_vec.
    rows = [
        {
            "publication_id": p["id"],
            "model_name": model_name,
            "dim": dim,
            "dense_vec": to_pgvector_literal(embeddings[i]),
        }
        for i, p in enumerate(pubs)
    ]
    for i in range(0, len(rows), UPSERT_BATCH_SIZE):
        chunk = rows[i : i + UPSERT_BATCH_SIZE]
        await (
            sb.table("publication_embeddings")
            .upsert(chunk, on_conflict="publication_id,model_name")
            .execute()
        )


async def mark_completed(sb: AsyncClient, task_id: str, n_pubs: int) -> None:
    await (
        sb.table("background_tasks")
        .update(
            {
                "status": "completed",
                "finished_at": "now()",
                "result": {"publications": n_pubs, "model": MODEL_NAME, "dim": EMBED_DIM},
            }
        )
        .eq("id", task_id)
        .execute()
    )


async def mark_failed(sb: AsyncClient, task_id: str, err: str) -> None:
    await (
        sb.table("background_tasks")
        .update(
            {
                "status": "failed",
                "finished_at": "now()",
                "error": err[:2000],
            }
        )
        .eq("id", task_id)
        .execute()
    )


async def process_one(sb: AsyncClient) -> bool:
    task = await claim_task(sb)
    if not task:
        return False

    task_id = task["id"]
    project_id = task["project_id"]
    params: dict[str, Any] = task.get("params") or {}
    model_name = params.get("model", MODEL_NAME)

    print(
        f"claimed task={task_id} project={project_id} model={model_name}",
        flush=True,
    )

    try:
        pubs = await fetch_publications(sb, project_id)
        if pubs:
            embeddings = encode(pubs)
            if embeddings.shape != (len(pubs), EMBED_DIM):
                raise RuntimeError(
                    f"unexpected embedding shape {embeddings.shape}, "
                    f"expected ({len(pubs)}, {EMBED_DIM})"
                )
            if DRY_RUN:
                print(
                    f"[dry-run] encoded {len(pubs)} pubs, shape={embeddings.shape} — NOT writing",
                    flush=True,
                )
            else:
                await upsert_embeddings(sb, pubs, embeddings, model_name, EMBED_DIM)
                print(
                    f"upserted {len(pubs)} embeddings (dim={EMBED_DIM})",
                    flush=True,
                )
        if SIMULATED_WORK_SECONDS > 0:
            print(
                f"simulating {SIMULATED_WORK_SECONDS}s of extra work before completing...",
                flush=True,
            )
            await asyncio.sleep(SIMULATED_WORK_SECONDS)
        await mark_completed(sb, task_id, len(pubs))
        print(f"completed task={task_id} ({len(pubs)} pubs)", flush=True)
    except Exception:
        err = traceback.format_exc()
        print(err, file=sys.stderr, flush=True)
        await mark_failed(sb, task_id, err)

    return True


async def main() -> None:
    sb: AsyncClient = await acreate_client(SUPABASE_URL, SUPABASE_KEY)

    def on_insert(_payload: dict) -> None:
        _wake.set()

    channel = sb.channel("background_tasks_inserts")
    channel.on_postgres_changes(
        event="INSERT",
        schema="public",
        table="background_tasks",
        filter=f"kind=eq.{WORKER_KIND}",
        callback=on_insert,
    )
    await channel.subscribe()
    print(
        f"subscribed: INSERT on background_tasks where kind={WORKER_KIND}",
        flush=True,
    )

    # Drain anything queued before we subscribed.
    while not _shutdown.is_set() and await process_one(sb):
        pass

    while not _shutdown.is_set():
        try:
            await asyncio.wait_for(_wake.wait(), timeout=SAFETY_POLL_SECONDS)
        except asyncio.TimeoutError:
            pass  # safety-net tick
        _wake.clear()

        while not _shutdown.is_set() and await process_one(sb):
            pass

    print("shutting down", flush=True)


def _stop() -> None:
    _shutdown.set()
    _wake.set()


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _stop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
