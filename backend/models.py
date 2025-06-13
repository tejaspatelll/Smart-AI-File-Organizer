"""Enhanced data models for Smart File Organizer.

This module provides comprehensive data models with validation, serialization,
and advanced features for the file organization system.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

from .config import AIProvider

class FileCategory(str, Enum):
    """Standard file categories."""
    DOCUMENTS = "Documents"
    IMAGES = "Images"
    VIDEOS = "Videos"
    AUDIO = "Audio"
    CODE = "Code"
    ARCHIVES = "Archives"
    APPLICATIONS = "Applications"
    OTHER = "Other"

class ProcessingStatus(str, Enum):
    """File processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class Priority(int, Enum):
    """Organization priority levels."""
    LOW = 1
    MODERATE = 2
    NORMAL = 3
    HIGH = 4
    CRITICAL = 5

@dataclass
class FileMetadata:
    """Enhanced file metadata with validation."""
    # Basic metadata
    name: str
    extension: str
    size_bytes: int
    created_timestamp: float
    modified_timestamp: float
    path: Path
    
    # Content metadata
    content_type: Optional[str] = None
    content_hash: Optional[str] = None
    encoding: Optional[str] = None
    
    # AI-generated metadata
    ai_tags: List[str] = field(default_factory=list)
    ai_description: Optional[str] = None
    ai_confidence: float = 0.0
    ai_provider: Optional[str] = None
    
    # Classification metadata
    detected_language: Optional[str] = None
    text_preview: Optional[str] = None
    image_dimensions: Optional[tuple[int, int]] = None
    duration_seconds: Optional[float] = None
    
    # Security metadata
    is_suspicious: bool = False
    virus_scan_result: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            "created_timestamp": self.created_timestamp,
            "modified_timestamp": self.modified_timestamp,
            "path": str(self.path),
            "content_type": self.content_type,
            "content_hash": self.content_hash,
            "encoding": self.encoding,
            "ai_tags": self.ai_tags,
            "ai_description": self.ai_description,
            "ai_confidence": self.ai_confidence,
            "ai_provider": self.ai_provider,
            "detected_language": self.detected_language,
            "text_preview": self.text_preview,
            "image_dimensions": self.image_dimensions,
            "duration_seconds": self.duration_seconds,
            "is_suspicious": self.is_suspicious,
            "virus_scan_result": self.virus_scan_result
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FileMetadata:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            extension=data["extension"],
            size_bytes=data["size_bytes"],
            created_timestamp=data["created_timestamp"],
            modified_timestamp=data["modified_timestamp"],
            path=Path(data["path"]),
            content_type=data.get("content_type"),
            content_hash=data.get("content_hash"),
            encoding=data.get("encoding"),
            ai_tags=data.get("ai_tags", []),
            ai_description=data.get("ai_description"),
            ai_confidence=data.get("ai_confidence", 0.0),
            ai_provider=data.get("ai_provider"),
            detected_language=data.get("detected_language"),
            text_preview=data.get("text_preview"),
            image_dimensions=tuple(data["image_dimensions"]) if data.get("image_dimensions") else None,
            duration_seconds=data.get("duration_seconds"),
            is_suspicious=data.get("is_suspicious", False),
            virus_scan_result=data.get("virus_scan_result")
        )
    
    @property
    def size_mb(self) -> float:
        """Get file size in MB."""
        return self.size_bytes / (1024 * 1024)
    
    @property
    def created_datetime(self) -> datetime:
        """Get creation datetime."""
        return datetime.fromtimestamp(self.created_timestamp)
    
    @property
    def modified_datetime(self) -> datetime:
        """Get modification datetime."""
        return datetime.fromtimestamp(self.modified_timestamp)
    
    @property
    def age_days(self) -> float:
        """Get file age in days."""
        return (time.time() - self.modified_timestamp) / (24 * 3600)

@dataclass
class FileInfo:
    """Enhanced file information with rich metadata."""
    path: Path
    metadata: FileMetadata
    
    # Classification results
    category: Optional[FileCategory] = None
    suggestion: Optional[str] = None
    confidence: float = 0.0
    
    # Processing status
    status: ProcessingStatus = ProcessingStatus.PENDING
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    
    # Relationships
    duplicates: List[Path] = field(default_factory=list)
    similar_files: List[Path] = field(default_factory=list)
    related_files: List[Path] = field(default_factory=list)
    
    # Advanced features
    tags: Set[str] = field(default_factory=set)
    custom_properties: Dict[str, Any] = field(default_factory=dict)
    
    # Unique identifier
    file_id: str = field(default_factory=lambda: str(uuid4()))
    
    def add_tag(self, tag: str):
        """Add a tag to the file."""
        self.tags.add(tag.lower().strip())
    
    def remove_tag(self, tag: str):
        """Remove a tag from the file."""
        self.tags.discard(tag.lower().strip())
    
    def has_tag(self, tag: str) -> bool:
        """Check if file has a specific tag."""
        return tag.lower().strip() in self.tags
    
    def set_category(self, category: Union[str, FileCategory], confidence: float = 0.0):
        """Set file category with confidence."""
        if isinstance(category, str):
            try:
                self.category = FileCategory(category)
            except ValueError:
                self.category = FileCategory.OTHER
        else:
            self.category = category
        self.confidence = confidence
    
    def mark_as_duplicate(self, original_path: Path):
        """Mark file as duplicate of another."""
        if original_path not in self.duplicates:
            self.duplicates.append(original_path)
        self.add_tag("duplicate")
    
    def is_duplicate(self) -> bool:
        """Check if file is marked as duplicate."""
        return len(self.duplicates) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "file_id": self.file_id,
            "path": str(self.path),
            "metadata": self.metadata.to_dict(),
            "category": self.category.value if self.category else None,
            "suggestion": self.suggestion,
            "confidence": self.confidence,
            "status": self.status.value,
            "processing_time": self.processing_time,
            "error_message": self.error_message,
            "duplicates": [str(p) for p in self.duplicates],
            "similar_files": [str(p) for p in self.similar_files],
            "related_files": [str(p) for p in self.related_files],
            "tags": list(self.tags),
            "custom_properties": self.custom_properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FileInfo:
        """Create from dictionary."""
        instance = cls(
            path=Path(data["path"]),
            metadata=FileMetadata.from_dict(data["metadata"]),
            suggestion=data.get("suggestion"),
            confidence=data.get("confidence", 0.0),
            status=ProcessingStatus(data.get("status", "pending")),
            processing_time=data.get("processing_time"),
            error_message=data.get("error_message"),
            duplicates=[Path(p) for p in data.get("duplicates", [])],
            similar_files=[Path(p) for p in data.get("similar_files", [])],
            related_files=[Path(p) for p in data.get("related_files", [])],
            tags=set(data.get("tags", [])),
            custom_properties=data.get("custom_properties", {}),
            file_id=data.get("file_id", str(uuid4()))
        )
        
        if category := data.get("category"):
            instance.set_category(category, instance.confidence)
        
        return instance

@dataclass
class AIResponse:
    """Response from AI classification."""
    file_path: Path
    category: str
    suggestion: str
    confidence: float
    reasoning: str
    tags: List[str] = field(default_factory=list)
    priority: int = 3
    relationships: List[str] = field(default_factory=list)
    provider: AIProvider = AIProvider.GEMINI
    
    # Metadata
    processing_time: float = 0.0
    tokens_used: int = 0
    cost_estimate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": str(self.file_path),
            "category": self.category,
            "suggestion": self.suggestion,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "tags": self.tags,
            "priority": self.priority,
            "relationships": self.relationships,
            "provider": self.provider.value,
            "processing_time": self.processing_time,
            "tokens_used": self.tokens_used,
            "cost_estimate": self.cost_estimate
        }

@dataclass
class BatchProcessingResult:
    """Result of batch processing operation."""
    results: List[AIResponse]
    success: bool
    processing_time: float = 0.0
    tokens_used: int = 0
    estimated_cost: float = 0.0
    error_message: Optional[str] = None
    provider: AIProvider = AIProvider.GEMINI
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of batch."""
        if not self.results:
            return 0.0
        successful = sum(1 for r in self.results if r.confidence > 0.5)
        return successful / len(self.results)
    
    @property
    def average_confidence(self) -> float:
        """Calculate average confidence."""
        if not self.results:
            return 0.0
        return sum(r.confidence for r in self.results) / len(self.results)

@dataclass
class OrganizationPlan:
    """Comprehensive organization plan."""
    plan_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    
    # Plan items
    moves: List[MovePlanItem] = field(default_factory=list)
    
    # Statistics
    total_files: int = 0
    estimated_time: float = 0.0
    confidence_score: float = 0.0
    
    # User preferences
    user_prompt: Optional[str] = None
    template_type: Optional[str] = None
    
    # Execution tracking
    is_executed: bool = False
    execution_time: Optional[datetime] = None
    execution_results: Dict[str, Any] = field(default_factory=dict)
    
    def add_move(self, move: MovePlanItem):
        """Add a move to the plan."""
        self.moves.append(move)
        self.total_files = len(self.moves)
        self._update_statistics()
    
    def _update_statistics(self):
        """Update plan statistics."""
        if self.moves:
            self.confidence_score = sum(m.confidence for m in self.moves) / len(self.moves)
            self.estimated_time = len(self.moves) * 0.1  # Rough estimate
    
    def get_summary(self) -> Dict[str, Any]:
        """Get plan summary."""
        categories = {}
        priorities = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        for move in self.moves:
            # Extract category from destination
            category = move.destination.parent.name if move.destination.parent else "Other"
            categories[category] = categories.get(category, 0) + 1
            priorities[move.priority] += 1
        
        return {
            "plan_id": self.plan_id,
            "total_files": self.total_files,
            "confidence_score": self.confidence_score,
            "estimated_time": self.estimated_time,
            "categories": categories,
            "priorities": priorities,
            "high_confidence_moves": sum(1 for m in self.moves if m.confidence >= 0.8),
            "duplicate_moves": sum(1 for m in self.moves if m.is_duplicate),
            "is_executed": self.is_executed
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "plan_id": self.plan_id,
            "created_at": self.created_at.isoformat(),
            "moves": [move.to_dict() for move in self.moves],
            "total_files": self.total_files,
            "estimated_time": self.estimated_time,
            "confidence_score": self.confidence_score,
            "user_prompt": self.user_prompt,
            "template_type": self.template_type,
            "is_executed": self.is_executed,
            "execution_time": self.execution_time.isoformat() if self.execution_time else None,
            "execution_results": self.execution_results
        }

@dataclass
class MovePlanItem:
    """Individual move operation in organization plan."""
    source: Path
    destination: Path
    
    # Metadata
    priority: Priority = Priority.NORMAL
    confidence: float = 0.0
    reason: str = ""
    
    # Flags
    is_duplicate: bool = False
    is_backup: bool = False
    requires_review: bool = False
    
    # Relationships
    original_file: Optional[Path] = None
    related_moves: List[str] = field(default_factory=list)  # Move IDs
    
    # Execution tracking
    move_id: str = field(default_factory=lambda: str(uuid4()))
    is_executed: bool = False
    execution_time: Optional[datetime] = None
    execution_error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "move_id": self.move_id,
            "source": str(self.source),
            "destination": str(self.destination),
            "priority": self.priority.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "is_duplicate": self.is_duplicate,
            "is_backup": self.is_backup,
            "requires_review": self.requires_review,
            "original_file": str(self.original_file) if self.original_file else None,
            "related_moves": self.related_moves,
            "is_executed": self.is_executed,
            "execution_time": self.execution_time.isoformat() if self.execution_time else None,
            "execution_error": self.execution_error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MovePlanItem:
        """Create from dictionary."""
        return cls(
            source=Path(data["source"]),
            destination=Path(data["destination"]),
            priority=Priority(data.get("priority", 3)),
            confidence=data.get("confidence", 0.0),
            reason=data.get("reason", ""),
            is_duplicate=data.get("is_duplicate", False),
            is_backup=data.get("is_backup", False),
            requires_review=data.get("requires_review", False),
            original_file=Path(data["original_file"]) if data.get("original_file") else None,
            related_moves=data.get("related_moves", []),
            move_id=data.get("move_id", str(uuid4())),
            is_executed=data.get("is_executed", False),
            execution_time=datetime.fromisoformat(data["execution_time"]) if data.get("execution_time") else None,
            execution_error=data.get("execution_error")
        )

@dataclass
class ProcessingProgress:
    """Track processing progress."""
    total_files: int
    processed_files: int = 0
    current_stage: str = "initializing"
    current_file: Optional[str] = None
    
    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    estimated_completion: Optional[datetime] = None
    
    # Statistics
    success_count: int = 0
    error_count: int = 0
    skip_count: int = 0
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return (datetime.now() - self.start_time).total_seconds()
    
    def update(self, processed: int = None, stage: str = None, current_file: str = None):
        """Update progress."""
        if processed is not None:
            self.processed_files = processed
        if stage is not None:
            self.current_stage = stage
        if current_file is not None:
            self.current_file = current_file
        
        # Update estimated completion
        if self.processed_files > 0 and self.total_files > 0:
            elapsed = self.elapsed_time
            rate = self.processed_files / elapsed
            remaining = self.total_files - self.processed_files
            eta_seconds = remaining / rate if rate > 0 else 0
            self.estimated_completion = datetime.now() + datetime.timedelta(seconds=eta_seconds)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "progress_percentage": self.progress_percentage,
            "current_stage": self.current_stage,
            "current_file": self.current_file,
            "elapsed_time": self.elapsed_time,
            "estimated_completion": self.estimated_completion.isoformat() if self.estimated_completion else None,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "skip_count": self.skip_count
        }

__all__ = [
    "FileCategory",
    "ProcessingStatus", 
    "Priority",
    "FileMetadata",
    "FileInfo",
    "AIResponse",
    "BatchProcessingResult",
    "OrganizationPlan",
    "MovePlanItem",
    "ProcessingProgress"
] 