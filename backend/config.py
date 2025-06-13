"""Production-grade configuration management for Smart File Organizer.

This module provides centralized configuration management with environment variables,
validation, and type safety for all backend components.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set
import logging

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AIProvider(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    CLAUDE = "claude"
    OLLAMA = "ollama"

class CacheStrategy(str, Enum):
    MEMORY = "memory"
    SQLITE = "sqlite"
    REDIS = "redis"
    HYBRID = "hybrid"

@dataclass
class AIConfig:
    """AI model configuration."""
    primary_provider: AIProvider = AIProvider.GEMINI
    fallback_providers: List[AIProvider] = field(default_factory=lambda: [AIProvider.GEMINI])
    
    # API Keys
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    claude_api_key: Optional[str] = None
    
    # Model settings
    gemini_model: str = "gemini-2.5-flash-preview-05-20"
    openai_model: str = "gpt-4o-mini"
    claude_model: str = "claude-3-haiku-20240307"
    ollama_model: str = "llama3.1:8b"
    
    # Processing limits
    batch_size: int = 25
    max_concurrent_requests: int = 5
    request_timeout: int = 120
    max_retries: int = 3
    
    # Content analysis
    max_text_length: int = 15000
    enable_content_analysis: bool = True
    enable_similarity_detection: bool = True
    similarity_threshold: float = 0.85
    
    # Advanced features
    enable_learning: bool = True
    enable_auto_tagging: bool = True
    enable_smart_batching: bool = True
    confidence_threshold: float = 0.7

@dataclass
class CacheConfig:
    """Caching configuration."""
    strategy: CacheStrategy = CacheStrategy.SQLITE
    sqlite_path: Path = Path("smart_cache.db")
    redis_url: Optional[str] = None
    memory_limit_mb: int = 512
    ttl_hours: int = 24 * 7  # 1 week
    enable_compression: bool = True
    
    # Cache warming
    preload_common: bool = True
    background_cleanup: bool = True

@dataclass
class PerformanceConfig:
    """Performance and resource management."""
    max_workers: int = 4
    max_memory_mb: int = 2048
    chunk_size: int = 1024 * 1024  # 1MB
    
    # File processing limits
    max_file_size_mb: int = 100
    max_files_per_batch: int = 1000
    max_scan_depth: int = 20
    
    # Async settings
    enable_async: bool = True
    async_batch_size: int = 50
    event_loop_timeout: int = 300

@dataclass
class SecurityConfig:
    """Security and safety configuration."""
    max_path_length: int = 4096
    allowed_extensions: Optional[Set[str]] = None
    blocked_extensions: Set[str] = field(default_factory=lambda: {'.exe', '.scr', '.bat', '.cmd'})
    blocked_directories: Set[str] = field(default_factory=lambda: {'System32', 'Windows', 'Program Files'})
    
    # Safety checks
    enable_backup: bool = True
    backup_important_files: bool = True
    dry_run_by_default: bool = False
    require_confirmation: bool = True

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: LogLevel = LogLevel.INFO
    file_path: Optional[Path] = None
    max_file_size_mb: int = 50
    backup_count: int = 5
    
    # Structured logging
    enable_json_logging: bool = False
    include_stack_trace: bool = True
    log_api_calls: bool = True
    log_performance_metrics: bool = True

@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration."""
    enable_metrics: bool = True
    metrics_file: Optional[Path] = None
    
    # Performance monitoring
    track_processing_time: bool = True
    track_memory_usage: bool = True
    track_api_usage: bool = True
    alert_on_errors: bool = True
    
    # Health checks
    enable_health_endpoint: bool = False
    health_check_interval: int = 60

@dataclass
class SmartOrganizerConfig:
    """Main configuration class containing all settings."""
    ai: AIConfig = field(default_factory=AIConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Application settings
    version: str = "2.0.0"
    debug_mode: bool = False
    
    def __post_init__(self):
        """Load configuration from environment variables."""
        self._load_from_env()
        self._validate_config()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # AI Configuration
        if api_key := os.getenv("GEMINI_API_KEY"):
            self.ai.gemini_api_key = api_key
        if api_key := os.getenv("OPENAI_API_KEY"):
            self.ai.openai_api_key = api_key
        if api_key := os.getenv("CLAUDE_API_KEY"):
            self.ai.claude_api_key = api_key
        
        # Provider selection
        if provider := os.getenv("AI_PROVIDER"):
            try:
                self.ai.primary_provider = AIProvider(provider.lower())
            except ValueError:
                pass
        
        # Cache configuration
        if cache_strategy := os.getenv("CACHE_STRATEGY"):
            try:
                self.cache.strategy = CacheStrategy(cache_strategy.lower())
            except ValueError:
                pass
        
        if redis_url := os.getenv("REDIS_URL"):
            self.cache.redis_url = redis_url
        
        # Performance settings
        if max_workers := os.getenv("MAX_WORKERS"):
            try:
                self.performance.max_workers = int(max_workers)
            except ValueError:
                pass
        
        # Debug mode
        self.debug_mode = os.getenv("DEBUG", "false").lower() == "true"
        
        # Logging
        if log_level := os.getenv("LOG_LEVEL"):
            try:
                self.logging.level = LogLevel(log_level.upper())
            except ValueError:
                pass
    
    def _validate_config(self):
        """Validate configuration values."""
        # Ensure at least one AI provider is available
        available_providers = []
        if self.ai.gemini_api_key:
            available_providers.append(AIProvider.GEMINI)
        if self.ai.openai_api_key:
            available_providers.append(AIProvider.OPENAI)
        if self.ai.claude_api_key:
            available_providers.append(AIProvider.CLAUDE)
        
        if not available_providers:
            # Fallback to local models if no API keys
            available_providers.append(AIProvider.OLLAMA)
        
        # Ensure primary provider is available
        if self.ai.primary_provider not in available_providers:
            self.ai.primary_provider = available_providers[0]
        
        # Validate performance limits
        if self.performance.max_workers < 1:
            self.performance.max_workers = 1
        elif self.performance.max_workers > 32:
            self.performance.max_workers = 32
        
        # Validate batch sizes
        if self.ai.batch_size < 1:
            self.ai.batch_size = 1
        elif self.ai.batch_size > 100:
            self.ai.batch_size = 100
    
    def setup_logging(self) -> logging.Logger:
        """Setup logging based on configuration."""
        logger = logging.getLogger("smart_organizer")
        logger.setLevel(getattr(logging, self.logging.level.value))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        if self.logging.enable_json_logging:
            # Would use structured logging here
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if self.logging.file_path:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                self.logging.file_path,
                maxBytes=self.logging.max_file_size_mb * 1024 * 1024,
                backupCount=self.logging.backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def get_cache_path(self) -> Path:
        """Get the cache database path."""
        if self.cache.sqlite_path.is_absolute():
            return self.cache.sqlite_path
        return Path(__file__).parent / self.cache.sqlite_path

# Global configuration instance
config = SmartOrganizerConfig()
logger = config.setup_logging()

__all__ = [
    "SmartOrganizerConfig",
    "AIConfig", 
    "CacheConfig",
    "PerformanceConfig",
    "SecurityConfig",
    "LoggingConfig",
    "MonitoringConfig",
    "AIProvider",
    "CacheStrategy",
    "LogLevel",
    "config",
    "logger"
] 