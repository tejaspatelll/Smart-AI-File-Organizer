"""Advanced caching system for Smart File Organizer.

This module provides multiple caching strategies with:
- SQLite caching with optimizations
- Redis support for distributed caching
- In-memory caching with LRU eviction
- Hybrid caching strategies
- Cache warming and background cleanup
- Compression and serialization
"""
from __future__ import annotations

import asyncio
import gzip
import json
import pickle
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

# Try to import Redis
try:
    import redis.asyncio as redis
except ImportError:
    redis = None

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    expires_at: Optional[float] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def touch(self):
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1

class CacheStrategy(ABC):
    """Abstract base class for cache strategies."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass

class MemoryCache(CacheStrategy):
    """High-performance in-memory cache with LRU eviction."""
    
    def __init__(self, max_size_mb: int = 512, default_ttl: int = 3600):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.entries: Dict[str, CacheEntry] = {}
        self.current_size = 0
        self.lock = asyncio.Lock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value with LRU tracking."""
        async with self.lock:
            if key not in self.entries:
                self.misses += 1
                return None
            
            entry = self.entries[key]
            if entry.is_expired():
                await self._remove_entry(key)
                self.misses += 1
                return None
            
            entry.touch()
            self.hits += 1
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value with size management."""
        async with self.lock:
            try:
                # Serialize to estimate size
                serialized = pickle.dumps(value)
                size_bytes = len(serialized)
                
                # Remove existing entry if present
                if key in self.entries:
                    await self._remove_entry(key)
                
                # Ensure we have space
                while (self.current_size + size_bytes > self.max_size_bytes and 
                       len(self.entries) > 0):
                    await self._evict_lru()
                
                # Create entry
                expires_at = None
                if ttl:
                    expires_at = time.time() + ttl
                elif self.default_ttl:
                    expires_at = time.time() + self.default_ttl
                
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=time.time(),
                    accessed_at=time.time(),
                    expires_at=expires_at,
                    size_bytes=size_bytes
                )
                
                self.entries[key] = entry
                self.current_size += size_bytes
                return True
                
            except Exception as e:
                logger.error(f"Failed to cache entry {key}: {e}")
                return False
    
    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        async with self.lock:
            if key in self.entries:
                await self._remove_entry(key)
                return True
            return False
    
    async def clear(self) -> bool:
        """Clear all entries."""
        async with self.lock:
            self.entries.clear()
            self.current_size = 0
            return True
    
    async def _remove_entry(self, key: str):
        """Remove entry and update size."""
        if key in self.entries:
            entry = self.entries.pop(key)
            self.current_size -= entry.size_bytes
    
    async def _evict_lru(self):
        """Evict least recently used entry."""
        if not self.entries:
            return
        
        # Find LRU entry
        lru_key = min(self.entries.keys(), 
                     key=lambda k: self.entries[k].accessed_at)
        
        await self._remove_entry(lru_key)
        self.evictions += 1
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "type": "memory",
            "entries": len(self.entries),
            "size_mb": self.current_size / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "hit_rate": hit_rate,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions
        }

class SQLiteCache(CacheStrategy):
    """Enhanced SQLite cache with compression and optimization."""
    
    def __init__(self, db_path: Path, enable_compression: bool = True):
        self.db_path = db_path
        self.enable_compression = enable_compression
        self.conn = None
        self.lock = asyncio.Lock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    async def initialize(self):
        """Initialize SQLite database."""
        self.conn = sqlite3.connect(
            str(self.db_path), 
            check_same_thread=False,
            timeout=30.0
        )
        
        # Optimize SQLite settings
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=10000")
        self.conn.execute("PRAGMA temp_store=memory")
        
        # Create enhanced schema
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                value BLOB NOT NULL,
                created_at REAL NOT NULL,
                accessed_at REAL NOT NULL,
                access_count INTEGER DEFAULT 1,
                expires_at REAL,
                size_bytes INTEGER,
                is_compressed BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Create indexes
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries(expires_at)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_accessed_at ON cache_entries(accessed_at)")
        
        self.conn.commit()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from SQLite cache."""
        if not self.conn:
            await self.initialize()
        
        async with self.lock:
            try:
                cursor = self.conn.execute(
                    "SELECT value, expires_at, is_compressed FROM cache_entries WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if not row:
                    self.misses += 1
                    return None
                
                value_blob, expires_at, is_compressed = row
                
                # Check expiration
                if expires_at and time.time() > expires_at:
                    await self.delete(key)
                    self.misses += 1
                    return None
                
                # Update access statistics
                self.conn.execute(
                    "UPDATE cache_entries SET accessed_at = ?, access_count = access_count + 1 WHERE key = ?",
                    (time.time(), key)
                )
                self.conn.commit()
                
                # Deserialize value
                if is_compressed:
                    value_blob = gzip.decompress(value_blob)
                
                value = pickle.loads(value_blob)
                self.hits += 1
                return value
                
            except Exception as e:
                logger.error(f"Failed to get cache entry {key}: {e}")
                self.misses += 1
                return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in SQLite cache."""
        if not self.conn:
            await self.initialize()
        
        async with self.lock:
            try:
                # Serialize value
                value_blob = pickle.dumps(value)
                is_compressed = False
                
                # Compress if beneficial
                if self.enable_compression and len(value_blob) > 1024:
                    compressed = gzip.compress(value_blob)
                    if len(compressed) < len(value_blob) * 0.9:  # 10% savings threshold
                        value_blob = compressed
                        is_compressed = True
                
                expires_at = None
                if ttl:
                    expires_at = time.time() + ttl
                
                current_time = time.time()
                
                self.conn.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (key, value, created_at, accessed_at, expires_at, size_bytes, is_compressed)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (key, value_blob, current_time, current_time, expires_at, len(value_blob), is_compressed))
                
                self.conn.commit()
                return True
                
            except Exception as e:
                logger.error(f"Failed to set cache entry {key}: {e}")
                return False
    
    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        if not self.conn:
            return False
        
        async with self.lock:
            try:
                self.conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                self.conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to delete cache entry {key}: {e}")
                return False
    
    async def clear(self) -> bool:
        """Clear all entries."""
        if not self.conn:
            return False
        
        async with self.lock:
            try:
                self.conn.execute("DELETE FROM cache_entries")
                self.conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")
                return False
    
    async def cleanup_expired(self) -> int:
        """Remove expired entries."""
        if not self.conn:
            return 0
        
        async with self.lock:
            try:
                cursor = self.conn.execute(
                    "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at < ?",
                    (time.time(),)
                )
                deleted = cursor.rowcount
                self.conn.commit()
                return deleted
            except Exception as e:
                logger.error(f"Failed to cleanup expired entries: {e}")
                return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.conn:
            return {"type": "sqlite", "initialized": False}
        
        try:
            cursor = self.conn.execute("SELECT COUNT(*), SUM(size_bytes), AVG(access_count) FROM cache_entries")
            count, total_size, avg_access = cursor.fetchone()
            
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "type": "sqlite",
                "entries": count or 0,
                "size_mb": (total_size or 0) / (1024 * 1024),
                "avg_access_count": avg_access or 0,
                "hit_rate": hit_rate,
                "hits": self.hits,
                "misses": self.misses,
                "db_path": str(self.db_path)
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"type": "sqlite", "error": str(e)}

class RedisCache(CacheStrategy):
    """Redis-based distributed cache."""
    
    def __init__(self, redis_url: str, prefix: str = "smart_organizer:"):
        self.redis_url = redis_url
        self.prefix = prefix
        self.client = None
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    async def initialize(self):
        """Initialize Redis connection."""
        if not redis:
            raise RuntimeError("Redis not available. Install with: pip install redis")
        
        self.client = redis.from_url(self.redis_url)
        
        # Test connection
        await self.client.ping()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        if not self.client:
            await self.initialize()
        
        try:
            value_bytes = await self.client.get(self.prefix + key)
            if value_bytes is None:
                self.misses += 1
                return None
            
            value = pickle.loads(value_bytes)
            self.hits += 1
            return value
            
        except Exception as e:
            logger.error(f"Failed to get Redis entry {key}: {e}")
            self.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis."""
        if not self.client:
            await self.initialize()
        
        try:
            value_bytes = pickle.dumps(value)
            await self.client.set(self.prefix + key, value_bytes, ex=ttl)
            return True
        except Exception as e:
            logger.error(f"Failed to set Redis entry {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete entry from Redis."""
        if not self.client:
            return False
        
        try:
            result = await self.client.delete(self.prefix + key)
            return result > 0
        except Exception as e:
            logger.error(f"Failed to delete Redis entry {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all entries with prefix."""
        if not self.client:
            return False
        
        try:
            # Get all keys with prefix
            keys = await self.client.keys(self.prefix + "*")
            if keys:
                await self.client.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Failed to clear Redis cache: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics."""
        if not self.client:
            return {"type": "redis", "initialized": False}
        
        try:
            info = await self.client.info("memory")
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "type": "redis",
                "memory_usage_mb": info.get("used_memory", 0) / (1024 * 1024),
                "hit_rate": hit_rate,
                "hits": self.hits,
                "misses": self.misses,
                "redis_url": self.redis_url
            }
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {"type": "redis", "error": str(e)}

class HybridCache(CacheStrategy):
    """Hybrid cache combining memory and persistent storage."""
    
    def __init__(self, memory_cache: MemoryCache, persistent_cache: CacheStrategy):
        self.l1_cache = memory_cache  # Fast memory cache
        self.l2_cache = persistent_cache  # Persistent cache
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from L1 first, then L2."""
        # Try L1 (memory) first
        value = await self.l1_cache.get(key)
        if value is not None:
            return value
        
        # Try L2 (persistent)
        value = await self.l2_cache.get(key)
        if value is not None:
            # Promote to L1
            await self.l1_cache.set(key, value)
            return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set in both L1 and L2."""
        l1_success = await self.l1_cache.set(key, value, ttl)
        l2_success = await self.l2_cache.set(key, value, ttl)
        return l1_success or l2_success
    
    async def delete(self, key: str) -> bool:
        """Delete from both caches."""
        l1_success = await self.l1_cache.delete(key)
        l2_success = await self.l2_cache.delete(key)
        return l1_success or l2_success
    
    async def clear(self) -> bool:
        """Clear both caches."""
        l1_success = await self.l1_cache.clear()
        l2_success = await self.l2_cache.clear()
        return l1_success and l2_success
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        l1_stats = await self.l1_cache.get_stats()
        l2_stats = await self.l2_cache.get_stats()
        
        return {
            "type": "hybrid",
            "l1_cache": l1_stats,
            "l2_cache": l2_stats
        }

class SmartCache:
    """Smart cache manager with automatic strategy selection."""
    
    def __init__(self):
        self.cache: Optional[CacheStrategy] = None
        self.background_tasks: set = set()
    
    async def initialize(self, config):
        """Initialize cache based on configuration."""
        from .config import CacheStrategy as ConfigCacheStrategy
        
        if config.cache.strategy == ConfigCacheStrategy.MEMORY:
            self.cache = MemoryCache(
                max_size_mb=config.cache.memory_limit_mb,
                default_ttl=config.cache.ttl_hours * 3600
            )
        elif config.cache.strategy == ConfigCacheStrategy.SQLITE:
            cache = SQLiteCache(
                db_path=config.get_cache_path(),
                enable_compression=config.cache.enable_compression
            )
            await cache.initialize()
            self.cache = cache
        elif config.cache.strategy == ConfigCacheStrategy.REDIS and config.cache.redis_url:
            cache = RedisCache(config.cache.redis_url)
            await cache.initialize()
            self.cache = cache
        elif config.cache.strategy == ConfigCacheStrategy.HYBRID:
            memory_cache = MemoryCache(config.cache.memory_limit_mb)
            sqlite_cache = SQLiteCache(config.get_cache_path())
            await sqlite_cache.initialize()
            self.cache = HybridCache(memory_cache, sqlite_cache)
        else:
            # Default to SQLite
            cache = SQLiteCache(config.get_cache_path())
            await cache.initialize()
            self.cache = cache
        
        # Start background cleanup if enabled
        if config.cache.background_cleanup:
            self._start_background_cleanup()
    
    def _start_background_cleanup(self):
        """Start background cleanup task."""
        async def cleanup_task():
            while True:
                try:
                    await asyncio.sleep(3600)  # Run every hour
                    if hasattr(self.cache, 'cleanup_expired'):
                        cleaned = await self.cache.cleanup_expired()
                        if cleaned > 0:
                            logger.info(f"Cleaned up {cleaned} expired cache entries")
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
        
        task = asyncio.create_task(cleanup_task())
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.cache:
            return None
        return await self.cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        if not self.cache:
            return False
        return await self.cache.set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.cache:
            return False
        return await self.cache.delete(key)
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        if not self.cache:
            return False
        return await self.cache.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache:
            return {"type": "none", "initialized": False}
        return await self.cache.get_stats()

# Global cache instance
smart_cache = SmartCache()

__all__ = [
    "CacheStrategy",
    "CacheEntry", 
    "MemoryCache",
    "SQLiteCache",
    "RedisCache",
    "HybridCache",
    "SmartCache",
    "smart_cache"
] 