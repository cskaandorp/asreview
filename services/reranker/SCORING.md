# Re-ranking with Postgres RPCs

How the re-ranker computes a new ranking after every label *without* shipping the 100k embedding matrix across the network. The key idea: ship the recipe, not the ingredients.

## TL;DR

A linear classifier reduces to "multiply each feature vector by a weight vector, sum the products." That's a dot product — pgvector has an operator for it (`<#>`). So instead of pulling 100k vectors out of Postgres to score them in Python, we wrap the dot-product query in a Postgres function, send only the trained weight vector (~11 KB), and let Postgres compute all 100k scores in place. Sub-second per re-rank, no matter where the worker runs.

## What is an RPC in Supabase?

An RPC is just a **function that lives inside Postgres**. You can call it from outside (the worker, the Next.js app, anything) over HTTPS — exactly like any other API endpoint. The function's *code* runs inside the database, sitting right next to all the data.

**Kitchen analogy:** the data is 100,000 ingredients in a kitchen.

- **The bad way:** ship all 100,000 ingredients to your house. Cook there. Send leftovers back. Lots of trucks, lots of time.
- **The RPC way:** stay home. Send the chef a small recipe. The chef cooks in the kitchen, right where the ingredients already are. You get a tiny "done" message back.

You send a recipe (a trained weight vector — about 11 KB). Postgres cooks (does the math). The data never moves.

## The recipe (dot product + pgvector)

The ranking math is trivial. A linear SVM (or logistic regression, or any linear model) reduces to: *"multiply the feature vector by the weight vector, sum the products. Higher result = more likely relevant."*

That's a **dot product**. pgvector has an operator for it: `<#>`. So this *one* SQL line scores every row in the table:

```sql
SELECT publication_id, -(dense_vec <#> $1) AS score
FROM publication_embeddings
WHERE model_name = 'mxbai';
```

The `$1` is the weight vector passed in. Postgres rips through 100k rows in milliseconds — it's just floating-point arithmetic on data that's already in memory.

**Why the leading `-`:** pgvector's `<#>` returns the **negative** inner product. That convention is deliberate — it lets pgvector's KNN indexes (which can only ASC-scan) return "closest match first." But it means the raw `<#>` value has the opposite sign of what we want: more relevant = more negative. Since our index is `(project_id, model_name, score DESC)` and the read query sorts DESC for "most relevant first," we negate at write time so the stored score is the actual SVM decision value (higher = more relevant). Sign on, problem off.

## Why a separate `publication_rankings` table

Scores live in their own table, not as a column on `publications`. Trade-offs that drove the decision:

- **Multi-model future-proof.** Adding `e5-large` later doesn't require a schema change — just a new `model_name` value.
- **Update churn is contained.** Re-ranking rewrites 100k rows per label. Doing that on the wide `publications` table (title + abstract + authors + doi + ...) creates fat dead tuples and defeats HOT updates. A slim `publication_rankings` row keeps vacuum work and bloat ~5–10× smaller.
- **Different lifecycles.** Publications = stable content. Rankings = ephemeral, derived, wipeable. Mirroring how `publication_labels` is also separate.
- **Optional history later.** A `computed_at` column lets you keep snapshots if needed.

Suggested shape:

```sql
publication_rankings (
    project_id      UUID,
    publication_id  UUID,
    model_name      TEXT,
    score           DOUBLE PRECISION,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (project_id, publication_id, model_name)
)
```

(Final names and FKs are owned by the web-app side.)

## Wrapping it as an RPC

Define a function once (in Supabase Studio's SQL editor). PostgREST auto-exposes it as an HTTP endpoint with no extra work.

```sql
CREATE OR REPLACE FUNCTION public.score_unlabeled(
    p_project_id uuid,
    p_model      text,
    p_weights    vector              -- unconstrained: works for any model dim
) RETURNS integer
LANGUAGE sql
SECURITY DEFINER
SET search_path = public, extensions, pg_temp
AS $$
    INSERT INTO public.publication_rankings
        (project_id, publication_id, model_name, score, computed_at)
    SELECT p_project_id,
           pe.publication_id,
           p_model,
           -(pe.dense_vec <#> p_weights),   -- raw inner product, higher = better
           now()
      FROM public.publication_embeddings pe
      JOIN public.publications p ON p.id = pe.publication_id
     WHERE p.project_id  = p_project_id
       AND pe.model_name = p_model
       AND pe.publication_id NOT IN (
           SELECT publication_id
             FROM public.publication_labels
            WHERE project_id = p_project_id
       )
    ON CONFLICT (project_id, publication_id, model_name) DO UPDATE
        SET score       = EXCLUDED.score,
            computed_at = EXCLUDED.computed_at;

    SELECT count(*)::int
      FROM public.publication_rankings
     WHERE project_id = p_project_id
       AND model_name = p_model;
$$;

-- Strip the default PUBLIC EXECUTE grant; allow only the worker role.
REVOKE ALL ON FUNCTION public.score_unlabeled(uuid, text, vector) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.score_unlabeled(uuid, text, vector) TO worker_external;
```

Why each piece is there:

- **`SET search_path = public, extensions, pg_temp`** — pgvector's `<#>` operator lives in the `extensions` schema in Supabase's default install. Without `extensions` on the path, `CREATE FUNCTION` fails at parse time with `operator does not exist: vector <#> vector`. The explicit path also closes the standard search-path-injection footgun for `SECURITY DEFINER` functions.
- **`SECURITY DEFINER`** — `publication_rankings` has RLS enabled with no INSERT/UPDATE policies for end users. Running the function as its owner side-steps that cleanly without needing the worker role to have `BYPASSRLS`.
- **`REVOKE ALL ... FROM PUBLIC` then `GRANT EXECUTE ... TO worker_external`** — Postgres' default is "any role can call any function." On a `SECURITY DEFINER` function this is wide open. The revoke-then-grant pattern locks execution down to just the worker.
- **`p.project_id = p_project_id`** — `publication_embeddings` has no `project_id` column (correctly: an embedding describes a publication, not a project–publication pair). The join through `publications` scopes the score to *this* project's pubs.
- **`pe.model_name = p_model`** — only score under the requested model (so re-ranking with `mxbai` doesn't overwrite scores stored under `e5-large`).
- **`NOT IN (publication_labels …)`** — don't score already-labeled pubs; their decision is final.
- **`-(pe.dense_vec <#> p_weights)`** — see the "Why the leading `-`" note in the recipe section above. Stores higher-is-better, matching the index and read query.

## What the worker needs as input

Three things, all of them small:

**1. The task** — from `background_tasks` (via the claim RPC)
- `project_id` — which project to re-rank
- `task_id` — so the worker can mark it completed/failed when done

**2. The labels** — from `publication_labels`
- For that project: rows of `(publication_id, label)` where `label` is the text `'include'` or `'exclude'`
- The worker maps these to `1` / `0` for sklearn before fitting
- This is the model's "training data" — what teaches the SVM what relevance looks like

**3. The embeddings of the labeled publications only** — from `publication_embeddings`
- One 1024-dim vector per labeled publication, for the chosen model (e.g. `mxbai`)
- Joined with the labels above to give the SVM `(vector → label)` pairs

That's it. Notice what is **not** input:

- The 100,000 unlabeled embeddings (they never leave Postgres).
- Titles, abstracts, authors, DOIs (the model only sees vectors and labels — text doesn't matter at this stage).

Typical sizes:

| Input | Rows | Bytes |
|---|---|---|
| Task | 1 | ~200 B |
| Labels | a few hundred to a few thousand | <1 MB |
| Labeled embeddings | same count | ~1–10 MB |

The worker's *output* is even smaller: a single 1024-dim weight vector (~11 KB) shipped to the `score_unlabeled` RPC. Postgres does the rest.

## Edge case: not enough labels yet

A linear SVM cannot be fit if all the labels are the same class — it needs at least one `'include'` *and* one `'exclude'` to learn anything. Early in a project, this is a real situation (e.g. the first reviewer has only marked a few exclusions).

The worker handles this by:

- Marking the task **completed** (not failed) — there's nothing wrong, the user just hasn't given the model enough to work with yet.
- Writing a small note into the task result (e.g. `{"skipped": true, "reason": "need >=1 include and >=1 exclude"}`) so the UI can show "waiting for more labels" instead of presenting a stale ranking.
- Not invoking the RPC. Existing rows in `publication_rankings` (if any) are left untouched.

The next labeling action enqueues another `rerank` task; once both classes are present, the model fits normally.

## Calling it from the worker

The Python side is a one-liner:

```python
result = await sb.rpc("score_unlabeled", {
    "p_project_id": project_id,
    "p_model": "mxbai",
    "p_weights": weights.tolist(),
}).execute()
```

What goes over the wire:

- **Out**: ~11 KB (the 1024-dim weight vector as JSON).
- **In**: ~50 bytes (a row count, or "done").

What happens inside Postgres while the worker waits:

- Computes the dot product against 100k vectors.
- Inserts/updates 100k score rows.
- Returns.

Sub-second on a sensible Postgres instance. **The 100k vectors never leave the database.**

## How the Next.js app reads "what's next"

Once scores are in `publication_rankings`, the "give me the highest-scored unlabeled publication" query is a single join:

```sql
SELECT p.*
  FROM publications p
  JOIN publication_rankings r ON r.publication_id = p.id
 WHERE r.project_id   = $1
   AND r.model_name   = (SELECT embedding_model FROM projects WHERE id = $1)
   AND p.id NOT IN (
       SELECT publication_id FROM publication_labels WHERE project_id = $1
   )
 ORDER BY r.score DESC
 LIMIT 1;
```

The existing `publication_rankings (project_id, model_name, score DESC)` index makes this sub-millisecond. The `publication_labels` PK `(project_id, publication_id)` covers the NOT-IN subquery.

The model name is sourced from `projects.embedding_model` rather than hardcoded so the architecture survives swapping models per project (or splitting embedding model from ranker model in the future). You could equivalently pass it as a separate parameter from the caller — both work; not hardcoding is the point.

### Multi-user safety: reservations

The query above is correct for a single reviewer per project. With two reviewers online, both would receive the same top-scored row, label it twice, and one of them wastes the work. The fix is the **`publication_reservations`** pattern: atomically claim the next pub via `INSERT … SELECT … ON CONFLICT DO NOTHING … RETURNING`, with a short expiry so abandoned reservations get freed automatically.

That pattern is documented on the Next.js side in `active-learning.md` — it lives in that doc rather than here because the reservation lifecycle is a web/UI concern (when to claim, when to release on navigation/timeout/submit), not a ranking concern. The reranker worker itself doesn't need to know about reservations; it only cares about the scoring math.

When the reservations table lands, the production "next pub" query gains a `LEFT JOIN publication_reservations` filter (only unreserved, or reserved by me) and becomes the canonical atomic-claim form. That form is a superset of the query above — same WHERE clauses, same ORDER BY, plus the reservation join and the INSERT-on-RETURNING shape. The reranking architecture stays unchanged.

### This SQL *is* the querier

In ASReview's pipeline, the "querier" is the component that picks which record to show the reviewer next. All four blessed models (`elas_u3`, `elas_u4`, `elas_l2`, `elas_h3`) use `querier="max"` — *"always pick the most-likely-relevant unlabeled record."* That's called **certainty sampling** in the active-learning literature.

The `ORDER BY score DESC LIMIT 1` above is a faithful implementation of `max`. No separate querier service exists in this architecture, and no separate querier service is needed — the SQL layer does it natively, with index support already in place.

**Why certainty sampling and not uncertainty sampling?** Classical active learning typically does *uncertainty* sampling — show the record the model is least sure about, to maximize information gain per label. ASReview's authors argue the reviewer's actual goal in a systematic review isn't to train the best classifier — it's to *find the relevant papers as quickly as possible*. Certainty sampling optimizes for that: it pushes probable includes to the top of the queue so reviewers harvest them early, even if the model itself learns slightly slower.

If you ever wanted to expose other queriers (`uncertainty`, `random`, hybrid `max_uncertainty`), they'd be variations on this same SQL — `ORDER BY abs(score) ASC` for uncertainty, `ORDER BY random()` for random, etc. The whole querier menu lives in the read query, not in the worker.

## Why this is the architectural punchline

The algorithm's brain is split cleanly between two systems doing what they're good at:

- **Python worker** — the smart-but-tiny part: train a linear classifier on a few hundred labeled rows. Pure CPU work on tiny data. (sklearn, ~50 lines.)
- **Postgres** — the dumb-but-bulky part: multiply a vector against 100k other vectors. (One SQL statement.)

Nothing moves around that doesn't need to. The worker is so light that it works the same way whether it's on the NUC, in AWS Lambda, or on a Raspberry Pi.

## Data-movement summary

| Step | Direction | Size |
|---|---|---|
| Pull labels for project | Postgres → worker | <1 MB |
| Pull labeled embeddings (~few hundred rows) | Postgres → worker | ~1–10 MB |
| Fit linear SVM in Python | (in-process) | — |
| Call `score_unlabeled` RPC | worker → Postgres | ~11 KB |
| Postgres scores 100k rows + writes `publication_rankings` | (in-database) | — |
| RPC response | Postgres → worker | ~50 B |

Total per re-rank: a few MB in, ~11 KB out. Compare to the naive "ship the 100k matrix to the worker" approach, which is **~1.1 GB JSON over the wire** per re-rank.
