#!/usr/bin/env python3
"""Verify a JWT's HS256 signature against a secret, locally.

Usage:
    SUPABASE_JWT_SECRET='<secret>' python scripts/verify_jwt.py <jwt>
    SUPABASE_JWT_SECRET='<secret>' python scripts/verify_jwt.py < /tmp/jwt.txt

Prints 'OK' if signature matches, 'MISMATCH' otherwise. Decodes and prints
the claims for inspection.
"""

import base64
import hashlib
import hmac
import json
import os
import sys


def b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def main() -> int:
    secret = os.environ.get("SUPABASE_JWT_SECRET")
    if not secret:
        print("error: set SUPABASE_JWT_SECRET in the environment", file=sys.stderr)
        return 1

    if len(sys.argv) >= 2:
        token = sys.argv[1].strip()
    else:
        token = sys.stdin.read().strip()

    if not token:
        print("error: no JWT provided (pass as arg or stdin)", file=sys.stderr)
        return 1

    parts = token.split(".")
    if len(parts) != 3:
        print(f"error: not a JWT (expected 3 segments, got {len(parts)})", file=sys.stderr)
        return 1

    header_b64, payload_b64, sig_b64 = parts

    try:
        header = json.loads(b64url_decode(header_b64))
        payload = json.loads(b64url_decode(payload_b64))
    except Exception as e:
        print(f"error: cannot decode header/payload: {e}", file=sys.stderr)
        return 1

    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    expected_sig = base64.urlsafe_b64encode(
        hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    ).rstrip(b"=").decode("ascii")

    print(f"header:  {json.dumps(header)}")
    print(f"payload: {json.dumps(payload)}")
    print(f"result:  {'OK' if expected_sig == sig_b64 else 'MISMATCH'}")
    return 0 if expected_sig == sig_b64 else 2


if __name__ == "__main__":
    sys.exit(main())
