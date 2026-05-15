#!/usr/bin/env bash
# Run the reranker worker locally with env vars from .env.
#
# Activates ./.venv if present, exports everything in .env into the
# environment, then exec's worker.py.

set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
  echo "error: .env not found in $(pwd)" >&2
  echo "       copy .env.example to .env and fill in SUPABASE_URL and SUPABASE_KEY" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

for var in SUPABASE_URL SUPABASE_KEY; do
  if [[ -z "${!var:-}" || "${!var}" == "replace-me" ]]; then
    echo "error: $var is unset or still 'replace-me' in .env" >&2
    exit 1
  fi
done

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

exec python worker.py
