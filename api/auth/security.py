"""Lightweight API security helpers: rate limiting, JWT auth, RBAC.

This is deliberately dependency-light (no external auth server) so the demo
runs standalone. Swap `SECRET_KEY` and add a real user store for production.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

import jwt
from fastapi import Header, HTTPException

SECRET_KEY = "change-me-in-production"  # pragma: allowlist secret
ALGORITHM = "HS256"

ROLE_PERMISSIONS = {
    "viewer": {"read"},
    "ml_engineer": {"read", "predict", "retrain"},
    "admin": {"read", "predict", "retrain", "promote", "manage_users"},
}


class RateLimiter:
    """Simple sliding-window rate limiter, in-memory (swap for Redis in multi-instance prod)."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.time()
        q = self._hits[key]
        while q and now - q[0] > self.window_seconds:
            q.popleft()
        if len(q) >= self.max_requests:
            return False
        q.append(now)
        return True


def create_access_token(subject: str, role: str, expires_in_seconds: int = 3600) -> str:
    payload = {"sub": subject, "role": role, "exp": int(time.time()) + expires_in_seconds}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="invalid token")


def require_permission(permission: str):
    """FastAPI dependency factory: require_permission("retrain") etc."""

    def _dependency(authorization: str = Header(default="")):
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        token = authorization.removeprefix("Bearer ").strip()
        claims = decode_token(token)
        role = claims.get("role", "viewer")
        if permission not in ROLE_PERMISSIONS.get(role, set()):
            raise HTTPException(status_code=403, detail=f"role '{role}' lacks permission '{permission}'")
        return claims

    return _dependency
