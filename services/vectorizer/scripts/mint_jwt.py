#!/usr/bin/env python3
"""Mint a long-lived Supabase JWT for the vectorization worker.

One-shot admin tool. Run on your own machine, copy the output to the worker's
.env as SUPABASE_KEY. Do NOT bake the JWT secret into the worker.

Usage:
    SUPABASE_JWT_SECRET='...' python scripts/mint_jwt.py
    SUPABASE_JWT_SECRET='...' python scripts/mint_jwt.py --role service_role
    SUPABASE_JWT_SECRET='...' python scripts/mint_jwt.py --years 1

Where to find SUPABASE_JWT_SECRET:
    Supabase Studio -> Project Settings -> API -> JWT Settings -> JWT Secret

The JWT is signed with HS256 (Supabase's default). No external dependencies;
uses only the Python standard library.
"""

import argparse
import base64
import hashlib
import hmac
import json
import os
import sys
import time


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def mint(secret: str, role: str, years: int) -> str:
    now = int(time.time())
    exp = now + years * 365 * 24 * 3600

    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": "supabase",
        "role": role,
        "iat": now,
        "exp": exp,
    }

    header_b64 = b64url(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = b64url(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header_b64}.{payload_b64}.{b64url(signature)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--role",
        default="worker_external",
        help="Postgres role to embed in the 'role' claim (default: worker_external)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=10,
        help="Expiry in years from now (default: 10)",
    )
    args = parser.parse_args()

    secret = os.environ.get("SUPABASE_JWT_SECRET")
    if not secret:
        print(
            "ERROR: set SUPABASE_JWT_SECRET in the environment.\n"
            "  Find it in Supabase Studio -> Project Settings -> API -> JWT Secret.",
            file=sys.stderr,
        )
        return 1
    if len(secret) < 32:
        print(
            f"WARNING: SUPABASE_JWT_SECRET is only {len(secret)} chars; "
            "Supabase secrets are normally 40+. Double-check you copied the JWT "
            "Secret, not an API key.",
            file=sys.stderr,
        )

    token = mint(secret, args.role, args.years)

    print(token)
    print(
        f"\nClaims: role={args.role}, expires in {args.years} years.\n"
        f"Paste this as SUPABASE_KEY in your worker .env file.\n"
        f"Verify by pasting at https://jwt.io (header/payload only — never paste "
        f"the secret).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
