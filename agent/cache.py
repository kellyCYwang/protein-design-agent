"""
Redis Cache for Protein Design Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Two-tier caching system:
  1. Tool call result caching — caches external API responses (UniProt, PDB, arXiv, etc.)
  2. Query-level response caching — caches full agent responses for similar queries

Uses Redis with native TTL expiry. All operations are best-effort: if Redis is
unavailable, the agent works normally without caching.
"""

import hashlib
import json
import logging
import os
import re
import time
import uuid
from typing import Callable

import numpy as np
import redis

logger = logging.getLogger(__name__)

# ============================================
# TTL CONSTANTS (seconds)
# ============================================

TOOL_TTLS: dict[str, int] = {
    "get_ec_number": 30 * 86400,             # 30 days
    "get_enzyme_structure": 14 * 86400,      # 14 days
    "get_uniprot_protein_details": 14 * 86400,  # 14 days
    "search_uniprot_proteins": 7 * 86400,    # 7 days
    "search_arxiv_papers": 1 * 86400,        # 1 day
    "search_preprints": 1 * 86400,           # 1 day
    "search_research_papers": 7 * 86400,     # 7 days (cleared on reindex)
}

QUERY_TTLS: dict[str, int] = {
    "simple": 30 * 86400,    # 30 days
    "detailed": 7 * 86400,   # 7 days
    "research": 3 * 86400,   # 3 days
}


# ============================================
# CACHE KEY HELPERS
# ============================================

def normalize_query(text: str) -> str:
    """Normalize a query string for exact-match cache lookup."""
    return re.sub(r"\s+", " ", text.strip().lower())


def make_tool_cache_key(tool_name: str, **kwargs) -> str:
    """
    Create a deterministic Redis key from tool name and arguments.

    Normalization: strip/lower strings, remove None values, sort keys.
    """
    normalized = {}
    for k, v in sorted(kwargs.items()):
        if v is None:
            continue
        if isinstance(v, str):
            normalized[k] = v.strip().lower()
        else:
            normalized[k] = v
    payload = json.dumps({"tool": tool_name, "kwargs": normalized}, sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"tool:{tool_name}:{digest}"


# ============================================
# REDIS CACHE
# ============================================

class RedisCache:
    """
    Persistent Redis cache for tool results and query responses.

    All public methods are safe to call even when Redis is down — they return
    None / 0 and log a warning instead of raising.
    """

    def __init__(self, url: str = "redis://localhost:6379/0"):
        self._url = url
        try:
            self._r = redis.from_url(url, decode_responses=False)
            self._r.ping()
            self._connected = True
            logger.info("Redis cache connected at %s", url)
        except (redis.ConnectionError, redis.TimeoutError) as exc:
            self._r = None
            self._connected = False
            logger.warning("Redis unavailable (%s) — caching disabled: %s", url, exc)

    @property
    def connected(self) -> bool:
        return self._connected

    def _reconnect(self) -> bool:
        """Try to reconnect if previously disconnected."""
        if self._connected:
            return True
        try:
            self._r = redis.from_url(self._url, decode_responses=False)
            self._r.ping()
            self._connected = True
            logger.info("Redis reconnected")
            return True
        except Exception:
            return False

    # ── Tool cache ───────────────────────────────────

    def get_tool_result(self, tool_name: str, **kwargs) -> str | None:
        """Look up a cached tool result. Returns the result string or None."""
        if not self._connected and not self._reconnect():
            return None
        try:
            key = make_tool_cache_key(tool_name, **kwargs)
            val = self._r.get(key)
            if val is not None:
                self._r.incr(b"stats:tool_hits")
                return val.decode("utf-8")
            self._r.incr(b"stats:tool_misses")
            return None
        except Exception as exc:
            logger.warning("Redis get_tool_result failed: %s", exc)
            self._connected = False
            return None

    def set_tool_result(self, tool_name: str, kwargs: dict, result: str, ttl: int) -> None:
        """Store a tool result with TTL."""
        if not self._connected and not self._reconnect():
            return
        try:
            key = make_tool_cache_key(tool_name, **kwargs)
            self._r.setex(key, ttl, result.encode("utf-8"))
        except Exception as exc:
            logger.warning("Redis set_tool_result failed: %s", exc)
            self._connected = False

    # ── Query cache ──────────────────────────────────

    def get_query_exact(self, normalized: str) -> tuple[str, str] | None:
        """
        Exact-match query lookup.

        Returns (response_text, query_type) or None.
        """
        if not self._connected and not self._reconnect():
            return None
        try:
            exact_key = f"qexact:{hashlib.sha256(normalized.encode()).hexdigest()}"
            cache_id = self._r.get(exact_key.encode())
            if cache_id is None:
                return None
            return self._get_query_data(cache_id.decode())
        except Exception as exc:
            logger.warning("Redis get_query_exact failed: %s", exc)
            self._connected = False
            return None

    def get_query_response(self, cache_id: str) -> tuple[str, str] | None:
        """Get query response by cache_id. Returns (response_text, query_type) or None."""
        return self._get_query_data(cache_id)

    def _get_query_data(self, cache_id: str) -> tuple[str, str] | None:
        """Internal: fetch qdata:{cache_id} and parse JSON."""
        try:
            data = self._r.get(f"qdata:{cache_id}".encode())
            if data is None:
                return None
            d = json.loads(data)
            self._r.incr(b"stats:query_hits")
            return d["response"], d["query_type"]
        except Exception as exc:
            logger.warning("Redis _get_query_data failed: %s", exc)
            return None

    def get_all_embeddings(self) -> list[tuple[str, np.ndarray]]:
        """
        Load all cached query embeddings for similarity search.

        Returns list of (cache_id, embedding_array) tuples.
        """
        if not self._connected and not self._reconnect():
            return []
        try:
            results = []
            cursor = 0
            while True:
                cursor, keys = self._r.scan(cursor, match=b"qembed:*", count=500)
                if keys:
                    values = self._r.mget(keys)
                    for k, v in zip(keys, values):
                        if v is not None:
                            cache_id = k.decode().split(":", 1)[1]
                            emb = np.frombuffer(v, dtype=np.float32).copy()
                            results.append((cache_id, emb))
                if cursor == 0:
                    break
            return results
        except Exception as exc:
            logger.warning("Redis get_all_embeddings failed: %s", exc)
            self._connected = False
            return []

    def set_query_cache(
        self,
        normalized: str,
        embedding: list[float],
        query_type: str,
        response: str,
        tools_used: list[str],
        ttl: int,
    ) -> None:
        """Store a query response with its embedding for future similarity matching."""
        if not self._connected and not self._reconnect():
            return
        try:
            cache_id = uuid.uuid4().hex
            pipe = self._r.pipeline()

            # Exact-match index
            exact_key = f"qexact:{hashlib.sha256(normalized.encode()).hexdigest()}"
            pipe.setex(exact_key.encode(), ttl, cache_id.encode())

            # Response data
            data = json.dumps({
                "response": response,
                "query_type": query_type,
                "tools_used": tools_used,
            })
            pipe.setex(f"qdata:{cache_id}".encode(), ttl, data.encode())

            # Embedding (raw float32 bytes)
            emb_bytes = np.array(embedding, dtype=np.float32).tobytes()
            pipe.setex(f"qembed:{cache_id}".encode(), ttl, emb_bytes)

            pipe.execute()
        except Exception as exc:
            logger.warning("Redis set_query_cache failed: %s", exc)
            self._connected = False

    # ── Maintenance ──────────────────────────────────

    def invalidate_tool(self, tool_name: str) -> int:
        """Delete all cached results for a specific tool. Returns count deleted."""
        if not self._connected and not self._reconnect():
            return 0
        try:
            count = 0
            cursor = 0
            while True:
                cursor, keys = self._r.scan(cursor, match=f"tool:{tool_name}:*".encode(), count=500)
                if keys:
                    count += self._r.delete(*keys)
                if cursor == 0:
                    break
            return count
        except Exception as exc:
            logger.warning("Redis invalidate_tool failed: %s", exc)
            self._connected = False
            return 0

    def invalidate_rag_cache(self) -> int:
        """Clear all search_research_papers tool cache and research query cache entries."""
        count = self.invalidate_tool("search_research_papers")
        # Also invalidate research-type query cache entries.
        # We need to scan qdata:* keys and check query_type, then delete related keys.
        if not self._connected and not self._reconnect():
            return count
        try:
            cursor = 0
            while True:
                cursor, keys = self._r.scan(cursor, match=b"qdata:*", count=500)
                if keys:
                    values = self._r.mget(keys)
                    to_delete = []
                    for k, v in zip(keys, values):
                        if v is None:
                            continue
                        try:
                            d = json.loads(v)
                            if d.get("query_type") == "research":
                                cache_id = k.decode().split(":", 1)[1]
                                to_delete.extend([
                                    k,
                                    f"qembed:{cache_id}".encode(),
                                ])
                        except (json.JSONDecodeError, KeyError):
                            pass
                    if to_delete:
                        count += self._r.delete(*to_delete)
                if cursor == 0:
                    break
            # Also clear qexact keys that pointed to deleted entries (they'll expire via TTL)
            return count
        except Exception as exc:
            logger.warning("Redis invalidate_rag_cache failed: %s", exc)
            self._connected = False
            return count

    def clear_all(self, cache_type: str = "all") -> int:
        """
        Clear cache keys.

        Args:
            cache_type: "tool", "query", or "all"
        """
        if not self._connected and not self._reconnect():
            return 0
        try:
            patterns = []
            if cache_type in ("tool", "all"):
                patterns.append(b"tool:*")
            if cache_type in ("query", "all"):
                patterns.extend([b"qexact:*", b"qdata:*", b"qembed:*"])
            if cache_type == "all":
                patterns.append(b"stats:*")

            count = 0
            for pattern in patterns:
                cursor = 0
                while True:
                    cursor, keys = self._r.scan(cursor, match=pattern, count=500)
                    if keys:
                        count += self._r.delete(*keys)
                    if cursor == 0:
                        break
            return count
        except Exception as exc:
            logger.warning("Redis clear_all failed: %s", exc)
            self._connected = False
            return 0

    def stats(self) -> dict:
        """Get cache statistics."""
        if not self._connected and not self._reconnect():
            return {"connected": False}
        try:
            pipe = self._r.pipeline()
            pipe.get(b"stats:tool_hits")
            pipe.get(b"stats:tool_misses")
            pipe.get(b"stats:query_hits")
            pipe.get(b"stats:query_misses")
            results = pipe.execute()

            def _int(val):
                return int(val) if val else 0

            # Count keys by type
            tool_count = 0
            cursor = 0
            while True:
                cursor, keys = self._r.scan(cursor, match=b"tool:*", count=500)
                tool_count += len(keys)
                if cursor == 0:
                    break

            query_count = 0
            cursor = 0
            while True:
                cursor, keys = self._r.scan(cursor, match=b"qdata:*", count=500)
                query_count += len(keys)
                if cursor == 0:
                    break

            return {
                "connected": True,
                "tool_cache": {
                    "entries": tool_count,
                    "hits": _int(results[0]),
                    "misses": _int(results[1]),
                },
                "query_cache": {
                    "entries": query_count,
                    "hits": _int(results[2]),
                    "misses": _int(results[3]),
                },
            }
        except Exception as exc:
            logger.warning("Redis stats failed: %s", exc)
            self._connected = False
            return {"connected": False, "error": str(exc)}


# ============================================
# QUERY EMBEDDING INDEX
# ============================================

class QueryEmbeddingIndex:
    """
    In-process cache of query embeddings for cosine similarity search.

    Periodically refreshes from Redis to pick up entries written by other workers.
    """

    def __init__(self, cache: RedisCache, refresh_interval: int = 60):
        self._cache = cache
        self._refresh_interval = refresh_interval
        self._embeddings: np.ndarray | None = None
        self._norms: np.ndarray | None = None
        self._cache_ids: list[str] = []
        self._last_refresh: float = 0

    def find_similar(
        self,
        query_embedding: list[float],
        threshold: float = 0.95,
    ) -> tuple[str, str] | None:
        """
        Find a cached response for a semantically similar query.

        Returns (response_text, query_type) or None.
        """
        self._maybe_refresh()

        if self._embeddings is None or len(self._embeddings) == 0:
            return None

        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm < 1e-10:
            return None

        similarities = (self._embeddings @ query_vec) / (self._norms * query_norm + 1e-10)
        best_idx = int(np.argmax(similarities))

        if similarities[best_idx] >= threshold:
            result = self._cache.get_query_response(self._cache_ids[best_idx])
            if result is not None:
                return result

        return None

    def _maybe_refresh(self) -> None:
        now = time.time()
        if now - self._last_refresh < self._refresh_interval:
            return

        rows = self._cache.get_all_embeddings()
        if rows:
            self._cache_ids = [r[0] for r in rows]
            self._embeddings = np.stack([r[1] for r in rows])
            self._norms = np.linalg.norm(self._embeddings, axis=1)
        else:
            self._cache_ids = []
            self._embeddings = None
            self._norms = None
        self._last_refresh = now


# ============================================
# SINGLETON + HELPER
# ============================================

_cache: RedisCache | None = None


def get_cache() -> RedisCache:
    """Get or create the singleton RedisCache instance."""
    global _cache
    if _cache is None:
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        enabled = os.getenv("CACHE_ENABLED", "true").lower() in ("true", "1", "yes")
        if enabled:
            _cache = RedisCache(url)
        else:
            # Return a disconnected instance (all ops will no-op)
            _cache = RedisCache.__new__(RedisCache)
            _cache._r = None
            _cache._connected = False
            _cache._url = url
            logger.info("Caching disabled via CACHE_ENABLED=false")
    return _cache


class QueryCacheLayer:
    """
    High-level interface for query-response caching.

    Combines exact-match lookup with cosine-similarity fallback.
    Requires an embedding function to compute query embeddings.
    """

    def __init__(self, embed_fn: Callable[[str], list[float]] | None = None):
        self._cache = get_cache()
        self._embed_fn = embed_fn
        self._index = QueryEmbeddingIndex(self._cache) if self._cache.connected else None

    @property
    def enabled(self) -> bool:
        return self._cache.connected

    def lookup(self, query: str) -> tuple[str, str] | None:
        """
        Check cache for a matching response.

        Returns (response_text, query_type) or None.
        Tries exact match first, then semantic similarity.
        """
        if not self._cache.connected:
            return None

        normalized = normalize_query(query)

        # 1. Exact match
        result = self._cache.get_query_exact(normalized)
        if result is not None:
            logger.info("Query cache HIT (exact): %s", normalized[:60])
            return result

        # 2. Similarity match (requires embedding function)
        if self._embed_fn is not None and self._index is not None:
            try:
                embedding = self._embed_fn(normalized)
                result = self._index.find_similar(embedding, threshold=0.95)
                if result is not None:
                    logger.info("Query cache HIT (similar): %s", normalized[:60])
                    return result
            except Exception as exc:
                logger.warning("Similarity search failed: %s", exc)

        return None

    def store(
        self,
        query: str,
        query_type: str,
        response: str,
        tools_used: list[str] | None = None,
    ) -> None:
        """Store a query response in the cache."""
        if not self._cache.connected:
            return

        normalized = normalize_query(query)
        ttl = QUERY_TTLS.get(query_type, 7 * 86400)

        embedding: list[float] = []
        if self._embed_fn is not None:
            try:
                embedding = self._embed_fn(normalized)
            except Exception as exc:
                logger.warning("Failed to compute embedding for cache: %s", exc)

        self._cache.set_query_cache(
            normalized=normalized,
            embedding=embedding,
            query_type=query_type,
            response=response,
            tools_used=tools_used or [],
            ttl=ttl,
        )
        logger.info("Query cached (%s, ttl=%dd): %s", query_type, ttl // 86400, normalized[:60])


_query_cache: QueryCacheLayer | None = None


def get_query_cache(embed_fn: Callable[[str], list[float]] | None = None) -> QueryCacheLayer:
    """Get or create the singleton QueryCacheLayer."""
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryCacheLayer(embed_fn)
    return _query_cache


def cached_tool_call(tool_name: str, result_fn: Callable[[], str], **kwargs) -> str:
    """
    Check Redis cache for a tool result. On miss, call result_fn() and store the result.

    Args:
        tool_name: Canonical tool name (must be in TOOL_TTLS)
        result_fn: Zero-arg callable that produces the result string
        **kwargs: The tool's arguments (used for cache key generation)

    Returns:
        The tool result string (from cache or freshly computed)
    """
    cache = get_cache()
    cached = cache.get_tool_result(tool_name, **kwargs)
    if cached is not None:
        return cached
    result = result_fn()
    if result is not None:
        ttl = TOOL_TTLS.get(tool_name, 86400)
        cache.set_tool_result(tool_name, kwargs, result, ttl)
    return result
