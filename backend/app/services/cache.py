"""
Simple cache read/write helpers for the api_cache table.

Cache keys should be deterministic hashes of the request parameters so
that the same API call (within TTL) reuses the stored result.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def make_key(prefix: str, **kwargs: Any) -> str:
    """Build a stable cache key from a prefix and keyword args."""
    payload = json.dumps(kwargs, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{prefix}:{digest}"


def cache_get(db: Any, key: str) -> dict | None:
    """
    Return cached JSON (as dict) if the entry exists and has not expired.
    Expired entries are deleted on access.
    Returns None on miss or expiry.
    """
    from app.models import ApiCache

    entry = db.query(ApiCache).filter(ApiCache.key == key).first()
    if entry is None:
        return None

    age_s = (datetime.utcnow() - entry.created_at).total_seconds()
    if age_s > entry.ttl_seconds:
        try:
            db.delete(entry)
            db.commit()
        except Exception:
            db.rollback()
        logger.debug("Cache expired: %s (age=%.0f s)", key, age_s)
        return None

    logger.debug("Cache hit: %s (age=%.0f s)", key, age_s)
    return json.loads(entry.json_data)


def cache_set(
    db: Any,
    key: str,
    data: Any,
    ttl_seconds: int = 86_400,  # 24 h default
) -> None:
    """Upsert a cache entry.  Silently ignores write failures."""
    from app.models import ApiCache

    try:
        entry = db.query(ApiCache).filter(ApiCache.key == key).first()
        blob = json.dumps(data)
        if entry:
            entry.json_data = blob
            entry.created_at = datetime.utcnow()
            entry.ttl_seconds = ttl_seconds
        else:
            db.add(ApiCache(key=key, json_data=blob, ttl_seconds=ttl_seconds))
        db.commit()
        logger.debug("Cache set: %s (ttl=%d s)", key, ttl_seconds)
    except Exception as exc:
        logger.warning("Cache write failed for key '%s': %s", key, exc)
        try:
            db.rollback()
        except Exception:
            pass
