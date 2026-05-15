#!/usr/bin/env python3
"""Re-ranker worker (Supabase-mediated).

On each labeling action a `background_tasks` row of kind='rerank' is queued.
The task's `params.model` carries an ASReview algorithm name (e.g. 'elas_h3').
Each algorithm is a bundle: which embedding model to score against + which
classifier hyperparameters to use. The worker:

  - Subscribes to Realtime INSERTs on background_tasks filtered to its kind.
  - Atomically claims one task via `claim_background_task` RPC.
  - Looks up the algorithm in ALGORITHMS to find its embedding model + params.
  - Pulls all labels (publication_labels) and matching embeddings
    (publication_embeddings, filtered to the algorithm's embedding model).
  - Fits a LinearSVC on the labeled embeddings with balanced sample weights.
  - Ships the trained weight vector to the `score_unlabeled` RPC, which
    writes new scores into publication_rankings inside Postgres.

The 100k unlabeled embeddings never leave the database — only the labeled
subset (a few hundred rows) comes out, and only the weight vector goes back.

Required env vars:
  SUPABASE_URL          e.g. https://supa.elixlab.nl
  SUPABASE_KEY          JWT signed with role=worker_external (or service_role)

Optional env vars:
  WORKER_KIND           default: 'rerank'
  ALGORITHM             default algorithm if params.model is not set: 'elas_h3'
  SAFETY_POLL_SECONDS   safety-net poll interval, default: 30
"""

import asyncio
import json
import os
import signal
import sys
import traceback
from typing import Any

import numpy as np
from sklearn.svm import LinearSVC
from supabase import acreate_client, AsyncClient

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

WORKER_KIND = os.environ.get("WORKER_KIND", "rerank")
DEFAULT_ALGORITHM = os.environ.get("ALGORITHM", "elas_h3")
SAFETY_POLL_SECONDS = int(os.environ.get("SAFETY_POLL_SECONDS", 30))

# ASReview's blessed model bundles. Each name maps to:
#   - embedding_model    which publication_embeddings.model_name rows to read
#                        (and which model_name the resulting publication_rankings
#                        rows are stored under, so the read-side query matches)
#   - classifier_params  kwargs to sklearn.svm.LinearSVC
#   - balance_ratio      sample_weight for label==1 (include); excludes get 1.0
#
# Values mirror asreview/models/models.py — see ELAS h3 / l2 / u4 configs there.
# Add new entries to support more algorithms; the worker is otherwise unchanged.
ALGORITHMS: dict[str, dict[str, Any]] = {
    "elas_h3": {
        "embedding_model": "mxbai",
        "classifier_params": {
            "loss": "squared_hinge",
            "C": 0.067,
            "max_iter": 5000,
        },
        "balance_ratio": 9.724,
    },
    # "elas_l2": {
    #     "embedding_model": "multilingual-e5-large",
    #     "classifier_params": {"loss": "squared_hinge", "C": 0.106, "max_iter": 5000},
    #     "balance_ratio": 9.707,
    # },
}

_wake = asyncio.Event()
_shutdown = asyncio.Event()


def _to_array(v: Any) -> list[float]:
    """Parse pgvector dense_vec from PostgREST output.

    Depending on PostgREST version, vector columns come back as either a JSON
    array (already a list) or a pgvector text literal '[v1,v2,...]' (a string).
    Both forms parse with json.loads.
    """
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        return json.loads(v)
    raise TypeError(f"unexpected dense_vec type: {type(v).__name__}")


async def claim_task(sb: AsyncClient) -> dict | None:
    res = await sb.rpc("claim_background_task", {"p_kind": WORKER_KIND}).execute()
    data = res.data
    if not data:
        return None
    return data[0] if isinstance(data, list) else data


async def fetch_labels(sb: AsyncClient, project_id: str) -> list[dict]:
    """Paginated read of publication_labels for the project."""
    page_size = 1000
    out: list[dict] = []
    start = 0
    while True:
        res = (
            await sb.table("publication_labels")
            .select("publication_id,label")
            .eq("project_id", project_id)
            .range(start, start + page_size - 1)
            .execute()
        )
        rows = res.data or []
        out.extend(rows)
        if len(rows) < page_size:
            return out
        start += page_size


async def fetch_embeddings(
    sb: AsyncClient, pub_ids: list[str], model_name: str
) -> dict[str, list[float]]:
    """Return {publication_id: dense_vec} for the requested pub_ids.

    Chunked into URL-sane batches; PostgREST puts the IN-clause IDs in the
    query string and a few thousand UUIDs can otherwise blow the URL limit.
    """
    out: dict[str, list[float]] = {}
    chunk = 500
    for i in range(0, len(pub_ids), chunk):
        ids = pub_ids[i : i + chunk]
        res = (
            await sb.table("publication_embeddings")
            .select("publication_id,dense_vec")
            .in_("publication_id", ids)
            .eq("model_name", model_name)
            .execute()
        )
        for row in res.data or []:
            out[row["publication_id"]] = _to_array(row["dense_vec"])
    return out


def fit_classifier(
    X: np.ndarray, y: np.ndarray, config: dict[str, Any]
) -> np.ndarray:
    """Fit LinearSVC with balanced sample weights; return its 1-D weight vector."""
    sample_weight = np.where(y == 1, config["balance_ratio"], 1.0)
    clf = LinearSVC(**config["classifier_params"])
    clf.fit(X, y, sample_weight=sample_weight)
    return clf.coef_[0]


async def call_score_unlabeled(
    sb: AsyncClient, project_id: str, model_name: str, weights: np.ndarray
) -> int:
    res = await sb.rpc(
        "score_unlabeled",
        {
            "p_project_id": project_id,
            "p_model": model_name,
            "p_weights": weights.tolist(),
        },
    ).execute()
    data = res.data
    if isinstance(data, list):
        data = data[0] if data else 0
    return int(data) if data is not None else 0


async def mark_completed(
    sb: AsyncClient, task_id: str, result: dict[str, Any]
) -> None:
    await (
        sb.table("background_tasks")
        .update({"status": "completed", "finished_at": "now()", "result": result})
        .eq("id", task_id)
        .execute()
    )


async def mark_failed(sb: AsyncClient, task_id: str, err: str) -> None:
    await (
        sb.table("background_tasks")
        .update({"status": "failed", "finished_at": "now()", "error": err[:2000]})
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
    algorithm = params.get("model", DEFAULT_ALGORITHM)

    config = ALGORITHMS.get(algorithm)
    if config is None:
        msg = (
            f"unknown algorithm {algorithm!r}; "
            f"supported: {sorted(ALGORITHMS)}"
        )
        await mark_failed(sb, task_id, msg)
        print(f"failed task={task_id}: {msg}", flush=True)
        return True
    embedding_model = config["embedding_model"]

    print(
        f"claimed task={task_id} project={project_id} "
        f"algorithm={algorithm} (embeddings={embedding_model})",
        flush=True,
    )

    try:
        labels = await fetch_labels(sb, project_id)
        if not labels:
            await mark_completed(
                sb, task_id, {"skipped": True, "reason": "no labels yet"}
            )
            print(f"skipped task={task_id}: no labels yet", flush=True)
            return True

        # Map publication_id -> 1 ('include') / 0 ('exclude').
        label_map = {
            r["publication_id"]: (1 if r["label"] == "include" else 0)
            for r in labels
        }
        n_include = sum(label_map.values())
        n_exclude = len(label_map) - n_include

        if n_include == 0 or n_exclude == 0:
            await mark_completed(
                sb,
                task_id,
                {
                    "skipped": True,
                    "reason": "need >=1 include and >=1 exclude to fit",
                    "n_include": n_include,
                    "n_exclude": n_exclude,
                },
            )
            print(
                f"skipped task={task_id}: "
                f"insufficient class balance (include={n_include} exclude={n_exclude})",
                flush=True,
            )
            return True

        emb_map = await fetch_embeddings(sb, list(label_map.keys()), embedding_model)
        missing = set(label_map) - set(emb_map)
        if missing:
            print(
                f"warn: {len(missing)}/{len(label_map)} labeled pubs have no "
                f"'{embedding_model}' embedding (vectorizer may be behind). "
                f"Fitting on the rest.",
                flush=True,
            )

        aligned_pub_ids = [pid for pid in label_map if pid in emb_map]
        if not aligned_pub_ids:
            await mark_completed(
                sb,
                task_id,
                {
                    "skipped": True,
                    "reason": "no embeddings available for any labeled pub",
                    "algorithm": algorithm,
                    "embedding_model": embedding_model,
                },
            )
            print(
                f"skipped task={task_id}: no usable embeddings for {embedding_model}",
                flush=True,
            )
            return True

        X = np.array(
            [emb_map[pid] for pid in aligned_pub_ids], dtype=np.float32
        )
        y = np.array(
            [label_map[pid] for pid in aligned_pub_ids], dtype=np.int32
        )

        # Class balance can degrade after the embedding filter; re-check.
        if len(set(y.tolist())) < 2:
            await mark_completed(
                sb,
                task_id,
                {
                    "skipped": True,
                    "reason": "post-embedding-filter, only one class remains",
                    "fit_examples": int(len(y)),
                },
            )
            print(f"skipped task={task_id}: one-class after filter", flush=True)
            return True

        weights = fit_classifier(X, y, config)

        if weights.shape[0] != X.shape[1]:
            raise RuntimeError(
                f"weight vector dim {weights.shape[0]} != embedding dim {X.shape[1]}"
            )

        n_scored = await call_score_unlabeled(
            sb, project_id, embedding_model, weights
        )

        await mark_completed(
            sb,
            task_id,
            {
                "skipped": False,
                "algorithm": algorithm,
                "embedding_model": embedding_model,
                "fit_examples": int(len(y)),
                "n_include": int(y.sum()),
                "n_exclude": int((y == 0).sum()),
                "rows_in_rankings": n_scored,
            },
        )
        print(
            f"completed task={task_id} "
            f"(fit on {len(y)} labels with {algorithm}, scored {n_scored} pubs)",
            flush=True,
        )
    except Exception:
        err = traceback.format_exc()
        print(err, file=sys.stderr, flush=True)
        await mark_failed(sb, task_id, err)

    return True


async def main() -> None:
    sb: AsyncClient = await acreate_client(SUPABASE_URL, SUPABASE_KEY)

    def on_insert(_payload: dict) -> None:
        _wake.set()

    channel = sb.channel("background_tasks_rerank_inserts")
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
