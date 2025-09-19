"""Advanced AI Provider System for Smart File Organizer.

This module implements a sophisticated multi-provider AI system with:
- Multiple AI service support (Gemini, OpenAI, Claude, Ollama)
- Intelligent fallback mechanisms
- Rate limiting and error handling
- Content-aware processing
- Smart batching optimization
- Learning from user corrections
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import aiohttp
try:
    import backoff  # type: ignore
except Exception:
    # Minimal shim so the app still runs if 'backoff' isn't installed.
    class _BackoffShim:
        def expo(self, *args, **kwargs):  # placeholder backoff strategy
            return None

        def on_exception(self, *args, **kwargs):
            def _decorator(fn):
                return fn
            return _decorator

    backoff = _BackoffShim()  # type: ignore

from .config import config, logger, AIProvider
from .enhanced_cache import smart_cache
from .json_utils import extract_json_array
from .models import FileInfo, AIResponse, BatchProcessingResult

# Conditional imports for different AI providers
try:
    import google.generativeai as genai
    from PIL import Image
except ImportError:
    genai = None
    Image = None

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

@dataclass
class ValidationResult:
    """Result of response validation."""
    is_valid: bool
    confidence_adjustment: float = 0.0
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    quality_score: float = 1.0

class ResponseValidator:
    """Validates AI responses for quality and consistency."""
    
    def __init__(self):
        self.validation_rules = [
            self._validate_confidence_consistency,
            self._validate_suggestion_quality,
            self._validate_reasoning_depth,
            self._validate_tag_relevance,
            self._validate_category_appropriateness
        ]
    
    def validate_response(self, response_item: Dict[str, Any], file_info: FileInfo) -> ValidationResult:
        """Comprehensive response validation."""
        result = ValidationResult(is_valid=True)
        
        for rule in self.validation_rules:
            try:
                rule_result = rule(response_item, file_info)
                if not rule_result.is_valid:
                    result.is_valid = False
                result.issues.extend(rule_result.issues)
                result.suggestions.extend(rule_result.suggestions)
                result.confidence_adjustment += rule_result.confidence_adjustment
                result.quality_score *= rule_result.quality_score
            except Exception as e:
                logger.warning(f"Validation rule failed: {e}")
        
        # Normalize confidence adjustment
        result.confidence_adjustment = max(-0.3, min(0.2, result.confidence_adjustment))
        
        return result
    
    def _validate_confidence_consistency(self, item: Dict[str, Any], file_info: FileInfo) -> ValidationResult:
        """Validate confidence score consistency."""
        result = ValidationResult(is_valid=True)
        confidence = float(item.get('confidence', 0.5))
        reasoning = item.get('reasoning', '')
        
        # Check if confidence matches reasoning quality
        if confidence > 0.8 and len(reasoning) < 100:
            result.issues.append("High confidence with insufficient reasoning")
            result.confidence_adjustment -= 0.15
            result.quality_score *= 0.9
        
        elif confidence < 0.3 and len(reasoning) > 200:
            result.issues.append("Low confidence despite detailed reasoning")
            result.confidence_adjustment += 0.1
        
        return result
    
    def _validate_suggestion_quality(self, item: Dict[str, Any], file_info: FileInfo) -> ValidationResult:
        """Validate folder suggestion quality."""
        result = ValidationResult(is_valid=True)
        suggestion = item.get('suggestion', '')
        
        if not suggestion or suggestion.lower() in ['miscellaneous', 'other', 'unknown']:
            result.issues.append("Generic folder suggestion")
            result.confidence_adjustment -= 0.2
            result.quality_score *= 0.8
        
        return result
    
    def _validate_reasoning_depth(self, item: Dict[str, Any], file_info: FileInfo) -> ValidationResult:
        """Validate reasoning quality and depth."""
        result = ValidationResult(is_valid=True)
        reasoning = item.get('reasoning', '')
        
        if len(reasoning) < 50:
            result.issues.append("Insufficient reasoning detail")
            result.confidence_adjustment -= 0.1
            result.quality_score *= 0.9
        
        return result
    
    def _validate_tag_relevance(self, item: Dict[str, Any], file_info: FileInfo) -> ValidationResult:
        """Validate tag relevance and quality."""
        result = ValidationResult(is_valid=True)
        tags = item.get('tags', [])
        
        if len(tags) < 3:
            result.suggestions.append("Add more relevant tags")
            result.confidence_adjustment -= 0.03
        
        return result
    
    def _validate_category_appropriateness(self, item: Dict[str, Any], file_info: FileInfo) -> ValidationResult:
        """Validate category assignment appropriateness."""
        result = ValidationResult(is_valid=True)
        # Basic validation - can be expanded
        return result

class FeedbackTracker:
    """Tracks user feedback for continuous improvement."""
    
    def __init__(self):
        self.feedback_history = []
        self.pattern_corrections = defaultdict(list)
    
    def record_feedback(self, original_suggestion: str, user_choice: str, file_info: FileInfo, confidence: float):
        """Record user feedback for learning."""
        feedback = {
            'timestamp': time.time(),
            'original_suggestion': original_suggestion,
            'user_choice': user_choice,
            'filename': file_info.path.name,
            'original_confidence': confidence,
            'was_accepted': original_suggestion == user_choice
        }
        self.feedback_history.append(feedback)

class QualityAssessor:
    """Assesses overall quality of AI responses."""
    
    def __init__(self):
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.75,
            'acceptable': 0.6,
            'poor': 0.4
        }
    
    def assess_batch_quality(self, results: List[AIResponse]) -> Dict[str, Any]:
        """Assess quality of a batch of AI responses."""
        if not results:
            return {'overall_quality': 'no_data', 'score': 0.0}
        
        quality_scores = [self._calculate_individual_quality(result) for result in results]
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        quality_level = 'poor'
        for level, threshold in self.quality_thresholds.items():
            if avg_quality >= threshold:
                quality_level = level
                break
        
        return {
            'overall_quality': quality_level,
            'score': avg_quality,
            'average_confidence': sum(r.confidence for r in results) / len(results)
        }
    
    def _calculate_individual_quality(self, result: AIResponse) -> float:
        """Calculate quality score for individual response."""
        score = 0.0
        
        # Confidence appropriateness (40%)
        if 0.7 <= result.confidence <= 0.95:
            score += 0.4
        elif 0.5 <= result.confidence < 0.7:
            score += 0.3
        else:
            score += 0.1
        
        # Reasoning quality (30%)
        if len(result.reasoning) >= 100:
            score += 0.3
        elif len(result.reasoning) >= 50:
            score += 0.2
        else:
            score += 0.1
        
        # Suggestion specificity (30%)
        if '/' in result.suggestion and result.suggestion.lower() not in ['other', 'miscellaneous']:
            score += 0.3
        elif result.suggestion.lower() not in ['other', 'miscellaneous']:
            score += 0.2
        else:
            score += 0.1
        
        return min(1.0, score)

@dataclass
class ProcessingMetrics:
    """Metrics for tracking AI processing performance."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_processing_time: float = 0.0
    avg_confidence: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0
    
    def add_request(self, processing_time: float, success: bool, confidence: float = 0.0, tokens: int = 0, cost: float = 0.0):
        """Add metrics for a single request."""
        self.total_requests += 1
        self.total_processing_time += processing_time
        if success:
            self.successful_requests += 1
            self.avg_confidence = (self.avg_confidence * (self.successful_requests - 1) + confidence) / self.successful_requests
        else:
            self.failed_requests += 1
        self.tokens_used += tokens
        self.cost_usd += cost
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_requests / self.total_requests if self.total_requests > 0 else 0.0
    
    @property
    def avg_processing_time(self) -> float:
        """Calculate average processing time."""
        return self.total_processing_time / self.total_requests if self.total_requests > 0 else 0.0

class AIProviderBase(ABC):
    """Abstract base class for AI providers."""
    
    def __init__(self, provider_type: AIProvider):
        self.provider_type = provider_type
        self.metrics = ProcessingMetrics()
        self.is_available = False
        self.last_error: Optional[str] = None
        self.rate_limiter = self._create_rate_limiter()
    
    def _create_rate_limiter(self):
        """Create rate limiter for this provider."""
        from asyncio import Semaphore
        return Semaphore(config.ai.max_concurrent_requests)
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the AI provider."""
        pass
    
    @abstractmethod
    async def classify_batch(self, files: List[FileInfo], context: Dict[str, Any] = None) -> BatchProcessingResult:
        """Classify a batch of files."""
        pass
    
    @abstractmethod
    def estimate_cost(self, files: List[FileInfo]) -> float:
        """Estimate the cost for processing files."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model being used."""
        pass
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the provider."""
        return {
            "provider": self.provider_type.value,
            "is_available": self.is_available,
            "last_error": self.last_error,
            "success_rate": self.metrics.success_rate,
            "avg_processing_time": self.metrics.avg_processing_time,
            "total_requests": self.metrics.total_requests
        }

class GeminiProvider(AIProviderBase):
    """Google Gemini AI provider with advanced features and validation."""
    
    def __init__(self):
        super().__init__(AIProvider.GEMINI)
        self.model = None
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        # Enhanced validation and feedback systems
        self.response_validator = ResponseValidator()
        self.feedback_tracker = FeedbackTracker()
        self.quality_assessor = QualityAssessor()
    
    async def initialize(self) -> bool:
        """Initialize Gemini provider."""
        if not genai or not config.ai.gemini_api_key:
            logger.warning("Gemini not available: missing dependencies or API key")
            return False
        
        try:
            # Mask the key in logs: show first 4 and last 4 chars
            key = config.ai.gemini_api_key or ""
            masked = f"{key[:4]}...{key[-4:]} (len={len(key)})" if key else "<none>"
            logger.info(f"GeminiProvider using API key: {masked}")
            genai.configure(api_key=config.ai.gemini_api_key)
            # Try preferred model, with fallback to common stable names
            preferred_models = [
                config.ai.gemini_model,
                "gemini-2.5-flash-lite",
                "gemini-2.5-flash-lite",
            ]
            last_err: Optional[Exception] = None
            for model_name in preferred_models:
                try:
                    self.model = genai.GenerativeModel(
                        model_name,
                        safety_settings=self.safety_settings
                    )
                    config.ai.gemini_model = model_name
                    break
                except Exception as e:  # model not found or not enabled
                    last_err = e
                    continue
            if not self.model:
                raise last_err or RuntimeError("Failed to initialize Gemini model")
            # Perform a lightweight validation to catch invalid keys early
            try:
                # Count tokens is inexpensive and validates auth
                _ = self.model.count_tokens(["healthcheck"])  # type: ignore[attr-defined]
            except Exception:
                # Fallback to a minimal generate call for older SDKs
                try:
                    _ = self.model.generate_content(["healthcheck"], generation_config={"temperature": 0.0})
                except Exception as e:
                    self.last_error = str(e)
                    logger.error(f"Failed to initialize Gemini (key validation): {e}")
                    return False

            self.is_available = True
            logger.info(f"Gemini provider initialized successfully with model {config.ai.gemini_model}")
            return True
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to initialize Gemini: {e}")
            return False
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=300
    )
    async def classify_batch(self, files: List[FileInfo], context: Dict[str, Any] = None) -> BatchProcessingResult:
        """Classify files using Gemini with advanced error handling."""
        start_time = time.time()
        
        async with self.rate_limiter:
            try:
                prompt = self._build_advanced_prompt(files, context)
                content_parts = []
                
                # Add context information
                if context:
                    content_parts.append(f"Context: {json.dumps(context, indent=2)}")
                
                content_parts.append(prompt)
                
                # Process files and add content (optional; disabled by default for cost)
                metadata_only = bool((context or {}).get("metadata_only", True))
                if not metadata_only:
                    for file_info in files:
                        content = await self._extract_file_content(file_info)
                        # Only add textual content; skip binary images to avoid MIME issues
                        if isinstance(content, str) and content:
                            content_parts.append(f"--- File: {file_info.path.name} ---")
                            content_parts.append(content)
                
                # Caching key (provider+model+files overview+context)
                overview = "\n".join([f.path.name for f in files])
                cache_key = f"ai:gemini:{config.ai.gemini_model}:{hashlib.sha256((overview + json.dumps(context or {}, sort_keys=True)).encode()).hexdigest()}"

                cached = await smart_cache.get(cache_key)
                if cached:
                    processing_time = time.time() - start_time
                    self.metrics.add_request(processing_time, True, 0.0, cached.get('tokens_used', 0), cached.get('estimated_cost', 0.0))
                    return BatchProcessingResult(**cached)

                # Generate response (prefer deterministic output)
                def _call_gemini():
                    try:
                        return self.model.generate_content(
                            content_parts,
                            generation_config={"temperature": 0.1}
                        )
                    except TypeError:
                        return self.model.generate_content(content_parts)

                response = await asyncio.get_event_loop().run_in_executor(
                    None, _call_gemini
                )
                
                # Parse response
                result = await self._parse_advanced_response(response, files)
                
                # Update metrics
                processing_time = time.time() - start_time
                avg_confidence = sum(r.confidence for r in result.results) / len(result.results) if result.results else 0.0
                self.metrics.add_request(processing_time, True, avg_confidence, result.tokens_used, result.estimated_cost)
                
                # Cache result
                await smart_cache.set(cache_key, result.__dict__, ttl=config.cache.ttl_hours * 3600)

                logger.info(f"Gemini processed {len(files)} files in {processing_time:.2f}s")
                return result
                
            except Exception as e:
                processing_time = time.time() - start_time
                self.metrics.add_request(processing_time, False)
                self.last_error = str(e)
                logger.error(f"Gemini classification failed: {e}")
                raise
    
    def _build_advanced_prompt(self, files: List[FileInfo], context: Dict[str, Any] = None) -> str:
        """Build a sophisticated multi-stage prompt with advanced context awareness."""
        
        # Stage 1: Context Analysis
        file_patterns = self._analyze_file_patterns(files)
        temporal_analysis = self._analyze_temporal_patterns(files)
        content_insights = self._analyze_content_patterns(files)
        
        base_prompt = f"""You are an elite AI file organization specialist with expertise in:
• Content semantic analysis and classification
• User workflow optimization and productivity enhancement  
• Project structure recognition and intelligent grouping
• Temporal pattern analysis for lifecycle management
• Duplicate detection and content similarity assessment

MISSION: Analyze {len(files)} files and create an optimal organization strategy that maximizes user productivity, maintains logical structure, and preserves important relationships.

CONTEXT ANALYSIS:
File Distribution: {file_patterns}
Temporal Patterns: {temporal_analysis}  
Content Insights: {content_insights}

ADVANCED RESPONSE FORMAT: Return a JSON array with enhanced objects containing:
- "filename": exact filename provided
- "category": main category (Documents, Images, Videos, Audio, Code, Archives, Applications, Other)
- "suggestion": intelligent folder path based on multi-factor analysis (FOLDER PATH ONLY; never include the filename)
- "confidence": sophisticated confidence score (0.0-1.0) with detailed breakdown
- "confidence_factors": {{
    "content_analysis": 0.0-1.0,
    "filename_patterns": 0.0-1.0,
    "temporal_relevance": 0.0-1.0,
    "relationship_strength": 0.0-1.0,
    "user_workflow_fit": 0.0-1.0
  }}
- "reasoning": comprehensive multi-factor analysis explanation
- "tags": semantic and functional tags (5-10 relevant tags)
- "priority": intelligent priority (1-5) based on usage patterns and importance
- "relationships": detailed file relationships with relationship types
- "workflow_impact": how this organization affects user productivity
- "alternative_suggestions": 2-3 alternative organization options with scores
- "archival_recommendation": assessment for archival/cleanup (if applicable)
- "duplicate_likelihood": probability this is a duplicate (0.0-1.0)
- "project_association": detected project or domain association
- "seasonal_relevance": time-based organization recommendations
- "folder_only": true (always true; indicates suggestion is not a file path)
- "matched_existing": string|null (the existing folder reused, if any)
- "new_folder": boolean (true only if no suitable existing folder can be reused)
- "rule_hits": string[] (list of rules/policies used to reach the decision)

SOPHISTICATED ANALYSIS FRAMEWORK:
1. CONTENT SEMANTIC ANALYSIS: Extract meaning, purpose, and functional context from file content
2. WORKFLOW OPTIMIZATION: Consider how users typically access and use similar files
3. PROJECT COHERENCE: Identify project boundaries and maintain logical groupings
4. TEMPORAL INTELLIGENCE: Factor in creation dates, modification patterns, and lifecycle stage
5. RELATIONSHIP MAPPING: Detect dependencies, references, and collaborative connections
6. PRODUCTIVITY ENHANCEMENT: Optimize for quick access, reduced cognitive load, and efficient workflows
7. FUTURE-PROOFING: Consider scalability and maintainability of the organization structure
8. INTELLIGENT DEDUPLICATION: Identify true duplicates vs. legitimate copies with different purposes
9. CONTEXTUAL PRIORITIZATION: Weight importance based on recency, size, content value, and usage indicators
10. USER INTENT INFERENCE: Infer user goals and preferences from file patterns and naming conventions

CONFIDENCE SCORING METHODOLOGY:
- 0.95-1.0: Extremely confident - Clear content analysis, strong patterns, obvious categorization
- 0.85-0.94: Highly confident - Good content understanding, clear category, minor ambiguity
- 0.70-0.84: Moderately confident - Reasonable analysis, some uncertainty in specific placement
- 0.50-0.69: Low confidence - Limited content access, ambiguous category, requires review
- 0.0-0.49: Very low confidence - Insufficient data, multiple valid options, needs human input"""
        
        # Add context-specific instructions
        if context:
            if user_prompt := context.get("user_prompt"):
                base_prompt += f"\n\nUSER INSTRUCTIONS:\n{user_prompt}\n"
            
            if template := context.get("template"):
                base_prompt += f"\n\nORGANIZATION TEMPLATE: {template}\n"
            
            if existing_structure := context.get("existing_structure"):
                base_prompt += f"\n\nEXISTING FOLDER STRUCTURE (reuse-first):\n{json.dumps(existing_structure, indent=2)}\n"
            
            # Optional policies
            if policies := context.get("policies"):
                base_prompt += f"\n\nPOLICIES (must be enforced):\n{json.dumps(policies, indent=2)}\n"
        
        # Add file overview
        file_overview = self._create_file_overview(files)
        base_prompt += f"\n\nFILES TO ANALYZE ({len(files)} total):\n{file_overview}\n"
        
        base_prompt += """

CRITICAL ORGANIZATION RULES (must follow):
- Suggestion must be a FOLDER path only, relative to the selected root; never include the filename.
- Prefer reusing existing subfolders from existing_structure; only set new_folder=true if no suitable existing folder exists.
- When reusing an existing folder, set matched_existing to that folder path.
- Use these top-level categories only: Documents, Images, Videos, Audio, Code, Archives, Applications, Other.
- Consolidate similar files in this batch under a common folder when appropriate.
- Respect user policies; record applied policies in rule_hits.
- Respond with ONLY the JSON array - no additional text.

EXAMPLE HIGH-QUALITY RESPONSE:
[
  {{
    "filename": "project_report_2024.pdf",
    "category": "Documents",
    "suggestion": "Projects/2024/Reports",
    "confidence": 0.92,
    "confidence_factors": {{
      "content_analysis": 0.95,
      "filename_patterns": 0.90,
      "temporal_relevance": 0.88,
      "relationship_strength": 0.85,
      "user_workflow_fit": 0.92
    }},
    "reasoning": "PDF document with clear project context.",
    "tags": ["project", "report", "2024"],
    "priority": 4,
    "relationships": ["project_data_2024.xlsx"],
    "folder_only": true,
    "matched_existing": "Documents/Projects/2024/Reports",
    "new_folder": false,
    "rule_hits": ["reuse_existing", "category_documents"],
    "workflow_impact": "Central reference document",
    "alternative_suggestions": [
      {{"path": "Reports/2024", "score": 0.78}},
      {{"path": "Documents/Business/Projects", "score": 0.71}}
    ],
    "archival_recommendation": "Keep active",
    "duplicate_likelihood": 0.15,
    "project_association": "2024 Project Initiative",
    "seasonal_relevance": "Current year"
  }}
]"""
        return base_prompt
    
    def _analyze_file_patterns(self, files: List[FileInfo]) -> str:
        """Analyze file patterns for context."""
        patterns = {}
        extensions = {}
        sizes = {"small": 0, "medium": 0, "large": 0}
        
        for file_info in files:
            # Extension analysis
            try:
                ext = getattr(file_info.metadata, 'extension', None) or 'unknown'
                if isinstance(file_info.metadata, dict):
                    ext = file_info.metadata.get('extension', 'unknown') or 'unknown'
                extensions[ext] = extensions.get(ext, 0) + 1
                
                # Size analysis
                size = getattr(file_info.metadata, 'size_bytes', 0)
                if isinstance(file_info.metadata, dict):
                    size = file_info.metadata.get('size', 0) or 0
                    
                if size < 1024 * 1024:  # < 1MB
                    sizes["small"] += 1
                elif size < 50 * 1024 * 1024:  # < 50MB
                    sizes["medium"] += 1
                else:
                    sizes["large"] += 1
                    
                # Pattern analysis from filename
                name = file_info.path.name.lower()
                if any(word in name for word in ['project', 'work', 'task']):
                    patterns['work_related'] = patterns.get('work_related', 0) + 1
                if any(word in name for word in ['personal', 'private', 'family']):
                    patterns['personal'] = patterns.get('personal', 0) + 1
                if any(word in name for word in ['temp', 'tmp', 'backup', 'copy']):
                    patterns['temporary'] = patterns.get('temporary', 0) + 1
                if any(word in name for word in ['2024', '2023', '2022']):
                    patterns['dated'] = patterns.get('dated', 0) + 1
                    
            except Exception:
                continue
        
        return f"Extensions: {dict(list(extensions.items())[:5])}, Sizes: {sizes}, Patterns: {patterns}"
    
    def _analyze_temporal_patterns(self, files: List[FileInfo]) -> str:
        """Analyze temporal patterns in files."""
        try:
            from datetime import datetime, timedelta
            now = datetime.now()
            recent = old = very_old = 0
            
            for file_info in files:
                try:
                    # Get modification time
                    mod_time = getattr(file_info.metadata, 'modified_timestamp', None)
                    if isinstance(file_info.metadata, dict):
                        mod_time = file_info.metadata.get('modified_timestamp', None)
                    
                    if mod_time:
                        file_date = datetime.fromtimestamp(mod_time)
                        days_old = (now - file_date).days
                        
                        if days_old <= 30:
                            recent += 1
                        elif days_old <= 365:
                            old += 1
                        else:
                            very_old += 1
                except Exception:
                    continue
            
            return f"Recent (≤30d): {recent}, Old (30d-1y): {old}, Very old (>1y): {very_old}"
        except Exception:
            return "Temporal analysis unavailable"
    
    def _analyze_content_patterns(self, files: List[FileInfo]) -> str:
        """Analyze content patterns for context."""
        content_types = {}
        for file_info in files:
            try:
                ext = getattr(file_info.metadata, 'extension', '').lower()
                if isinstance(file_info.metadata, dict):
                    ext = (file_info.metadata.get('extension', '') or '').lower()
                
                if ext in {'.pdf', '.docx', '.doc', '.txt', '.md'}:
                    content_types['documents'] = content_types.get('documents', 0) + 1
                elif ext in {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg'}:
                    content_types['images'] = content_types.get('images', 0) + 1
                elif ext in {'.mp4', '.avi', '.mov', '.mkv'}:
                    content_types['videos'] = content_types.get('videos', 0) + 1
                elif ext in {'.py', '.js', '.html', '.css', '.json'}:
                    content_types['code'] = content_types.get('code', 0) + 1
                elif ext in {'.zip', '.rar', '.tar', '.gz'}:
                    content_types['archives'] = content_types.get('archives', 0) + 1
                else:
                    content_types['other'] = content_types.get('other', 0) + 1
            except Exception:
                continue
        
        return f"Content types: {content_types}"
    
    def _create_file_overview(self, files: List[FileInfo]) -> str:
        """Create an overview of files for better context."""
        overview = []
        for file_info in files:
            try:
                ext = getattr(file_info.metadata, 'extension', None) or 'unknown'
                size = getattr(file_info.metadata, 'size_bytes', None)
                if size is None:
                    # Fallback to dict-style if metadata is a plain dict
                    size = file_info.metadata.get('size', 0) if isinstance(file_info.metadata, dict) else 0
            except Exception:
                ext, size = 'unknown', 0
            overview.append(f"- {file_info.path.name} ({ext}, {size} bytes)")
        return "\n".join(overview)
    
    async def _extract_file_content(self, file_info: FileInfo) -> Optional[str]:
        """Extract content from file for enhanced AI analysis."""
        try:
            ext = None
            # Support dataclass and dict metadata
            if hasattr(file_info.metadata, 'extension'):
                ext = (file_info.metadata.extension or '').lower()
            elif isinstance(file_info.metadata, dict):
                ext = (file_info.metadata.get("extension", "") or '').lower()
            
            # Enhanced content extraction with context awareness
            content = await self._extract_content_by_type(file_info, ext)
            
            if content:
                # Add metadata context to content
                enhanced_content = self._enhance_content_with_metadata(file_info, content, ext)
                return enhanced_content
            
            return None
        except Exception as e:
            logger.warning(f"Failed to extract content from {file_info.path.name}: {e}")
            return None
    
    async def _extract_content_by_type(self, file_info: FileInfo, ext: str) -> Optional[str]:
        """Extract content based on file type with enhanced processing."""
        try:
            # Skip binary image payloads but provide metadata description
            if ext in {'.png', '.jpg', '.jpeg', '.webp', '.heic', '.gif', '.bmp', '.tiff'}:
                return await self._analyze_image_metadata(file_info)
            
            # Text-based files with enhanced processing
            elif ext in {'.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv', '.log'}:
                content = file_info.path.read_text(encoding='utf-8', errors='ignore')
                # Intelligent content truncation
                return await self._intelligently_truncate_content(content, ext)
            
            # Document files with enhanced extraction
            elif ext == '.pdf':
                from .utils import extract_text_from_pdf
                text = extract_text_from_pdf(file_info.path)
                return await self._process_document_content(text, 'pdf')
            
            elif ext == '.docx':
                from .utils import extract_text_from_docx
                text = extract_text_from_docx(file_info.path)
                return await self._process_document_content(text, 'docx')
            
            # Code files with syntax awareness
            elif ext in {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs'}:
                content = file_info.path.read_text(encoding='utf-8', errors='ignore')
                return await self._analyze_code_content(content, ext)
            
            # Configuration and data files
            elif ext in {'.json', '.yaml', '.yml', '.xml', '.cfg', '.ini'}:
                content = file_info.path.read_text(encoding='utf-8', errors='ignore')
                return await self._analyze_config_content(content, ext)
            
            return None
            
        except Exception as e:
            logger.debug(f"Content extraction failed for {file_info.path.name}: {e}")
            return None
    
    async def _analyze_image_metadata(self, file_info: FileInfo) -> str:
        """Analyze image file metadata for AI context."""
        try:
            metadata_info = []
            
            # Get file size and dimensions if available
            if hasattr(file_info.metadata, 'size_bytes'):
                size_mb = file_info.metadata.size_bytes / (1024 * 1024)
                metadata_info.append(f"Size: {size_mb:.1f}MB")
            
            # Try to get image dimensions
            try:
                if Image:
                    with Image.open(file_info.path) as img:
                        metadata_info.append(f"Dimensions: {img.width}x{img.height}")
                        metadata_info.append(f"Format: {img.format}")
                        if hasattr(img, 'mode'):
                            metadata_info.append(f"Mode: {img.mode}")
            except Exception:
                pass
            
            # Analyze filename for context
            filename_analysis = self._analyze_filename_patterns(file_info.path.name)
            if filename_analysis:
                metadata_info.append(f"Filename suggests: {filename_analysis}")
            
            return f"Image file metadata: {', '.join(metadata_info)}" if metadata_info else "Image file"
            
        except Exception:
            return "Image file"
    
    async def _intelligently_truncate_content(self, content: str, ext: str) -> str:
        """Intelligently truncate content while preserving important parts."""
        from .config import config
        max_length = getattr(config.ai, 'max_text_length', 5000)
        
        if len(content) <= max_length:
            return content
        
        # For different file types, preserve different parts
        if ext == '.md':
            # For markdown, preserve headers and first paragraphs
            lines = content.split('\n')
            important_lines = []
            char_count = 0
            
            for line in lines:
                if line.startswith('#') or len(line.strip()) == 0:  # Headers and empty lines
                    important_lines.append(line)
                    char_count += len(line) + 1
                elif char_count < max_length * 0.8:  # First 80% for content
                    important_lines.append(line)
                    char_count += len(line) + 1
                else:
                    break
            
            result = '\n'.join(important_lines)
            if len(result) < len(content):
                result += f"\n\n[Content truncated - original length: {len(content)} chars]"
            return result
        
        elif ext in {'.py', '.js', '.ts'}:
            # For code, preserve imports, classes, and function definitions
            lines = content.split('\n')
            important_lines = []
            char_count = 0
            
            for line in lines:
                stripped = line.strip()
                # Preserve imports, class/function definitions, comments
                if (stripped.startswith(('import ', 'from ', 'class ', 'def ', 'function ', '/*', '//', '#')) or
                    len(stripped) == 0):
                    important_lines.append(line)
                    char_count += len(line) + 1
                elif char_count < max_length * 0.7:
                    important_lines.append(line)
                    char_count += len(line) + 1
                else:
                    break
            
            result = '\n'.join(important_lines)
            if len(result) < len(content):
                result += f"\n\n// [Code truncated - original length: {len(content)} chars]"
            return result
        
        else:
            # Default truncation with smart boundary
            truncated = content[:max_length]
            # Try to end at a sentence or paragraph
            for boundary in ['. ', '\n\n', '\n', ' ']:
                last_boundary = truncated.rfind(boundary)
                if last_boundary > max_length * 0.8:
                    truncated = truncated[:last_boundary + len(boundary)]
                    break
            
            if len(truncated) < len(content):
                truncated += f"\n\n[Content truncated - original length: {len(content)} chars]"
            return truncated
    
    async def _process_document_content(self, text: Optional[str], doc_type: str) -> Optional[str]:
        """Process document content with enhanced analysis."""
        if not text:
            return None
        
        # Clean and structure document content
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                processed_lines.append(line)
        
        processed_text = '\n'.join(processed_lines)
        
        # Add document analysis context
        analysis = []
        if len(processed_lines) > 50:
            analysis.append("lengthy document")
        if any(word in processed_text.lower() for word in ['table of contents', 'executive summary', 'conclusion']):
            analysis.append("formal document structure")
        if any(word in processed_text.lower() for word in ['project', 'report', 'analysis']):
            analysis.append("business/project document")
        
        context = f"[{doc_type.upper()} document" + (f" - {', '.join(analysis)}" if analysis else "") + "]\n"
        
        return context + await self._intelligently_truncate_content(processed_text, f'.{doc_type}')
    
    async def _analyze_code_content(self, content: str, ext: str) -> str:
        """Analyze code content with programming context."""
        lines = content.split('\n')
        analysis = []
        
        # Detect programming patterns
        imports = [line for line in lines if line.strip().startswith(('import ', 'from ', '#include', 'using '))]
        classes = [line for line in lines if 'class ' in line and not line.strip().startswith('#')]
        functions = [line for line in lines if any(keyword in line for keyword in ['def ', 'function ', 'func '])]
        
        if imports:
            analysis.append(f"{len(imports)} imports")
        if classes:
            analysis.append(f"{len(classes)} classes")
        if functions:
            analysis.append(f"{len(functions)} functions")
        
        # Detect frameworks/libraries
        frameworks = []
        content_lower = content.lower()
        if 'react' in content_lower or 'jsx' in content_lower:
            frameworks.append('React')
        if 'django' in content_lower or 'flask' in content_lower:
            frameworks.append('Python web framework')
        if 'express' in content_lower or 'node' in content_lower:
            frameworks.append('Node.js')
        
        context = f"[{ext[1:].upper()} code file"
        if analysis:
            context += f" - {', '.join(analysis)}"
        if frameworks:
            context += f" - uses {', '.join(frameworks)}"
        context += "]\n"
        
        return context + await self._intelligently_truncate_content(content, ext)
    
    async def _analyze_config_content(self, content: str, ext: str) -> str:
        """Analyze configuration file content."""
        analysis = []
        
        try:
            if ext == '.json':
                import json
                data = json.loads(content)
                if isinstance(data, dict):
                    analysis.append(f"{len(data)} config keys")
                    # Detect common config types
                    if 'dependencies' in data or 'devDependencies' in data:
                        analysis.append('package.json')
                    elif 'scripts' in data:
                        analysis.append('build configuration')
            elif ext in {'.yaml', '.yml'}:
                analysis.append('YAML configuration')
            elif ext in {'.cfg', '.ini'}:
                analysis.append('INI/Config file')
        except Exception:
            pass
        
        context = f"[{ext[1:].upper()} configuration"
        if analysis:
            context += f" - {', '.join(analysis)}"
        context += "]\n"
        
        return context + await self._intelligently_truncate_content(content, ext)
    
    def _enhance_content_with_metadata(self, file_info: FileInfo, content: str, ext: str) -> str:
        """Enhance content with metadata context for better AI analysis."""
        metadata_context = []
        
        # File size context
        try:
            size_bytes = getattr(file_info.metadata, 'size_bytes', 0)
            if isinstance(file_info.metadata, dict):
                size_bytes = file_info.metadata.get('size', 0) or 0
            
            if size_bytes:
                if size_bytes > 10 * 1024 * 1024:  # > 10MB
                    metadata_context.append("large file")
                elif size_bytes < 1024:  # < 1KB
                    metadata_context.append("small file")
        except Exception:
            pass
        
        # Temporal context
        try:
            mod_time = getattr(file_info.metadata, 'modified_timestamp', None)
            if isinstance(file_info.metadata, dict):
                mod_time = file_info.metadata.get('modified_timestamp', None)
            
            if mod_time:
                from datetime import datetime, timedelta
                file_date = datetime.fromtimestamp(mod_time)
                days_old = (datetime.now() - file_date).days
                
                if days_old <= 7:
                    metadata_context.append("recently modified")
                elif days_old > 365:
                    metadata_context.append("old file")
        except Exception:
            pass
        
        # Filename analysis
        filename_analysis = self._analyze_filename_patterns(file_info.path.name)
        if filename_analysis:
            metadata_context.append(filename_analysis)
        
        if metadata_context:
            context_header = f"[File context: {', '.join(metadata_context)}]\n\n"
            return context_header + content
        
        return content
    
    def _analyze_filename_patterns(self, filename: str) -> str:
        """Analyze filename patterns for context clues."""
        name_lower = filename.lower()
        patterns = []
        
        # Project indicators
        if any(word in name_lower for word in ['project', 'proj']):
            patterns.append('project-related')
        
        # Document types
        if any(word in name_lower for word in ['report', 'summary', 'analysis']):
            patterns.append('report/analysis')
        elif any(word in name_lower for word in ['meeting', 'notes', 'minutes']):
            patterns.append('meeting/notes')
        elif any(word in name_lower for word in ['presentation', 'slides', 'ppt']):
            patterns.append('presentation')
        
        # Temporal indicators
        if any(year in name_lower for year in ['2024', '2023', '2022', '2021']):
            patterns.append('dated content')
        
        # Status indicators
        if any(word in name_lower for word in ['draft', 'temp', 'tmp']):
            patterns.append('temporary/draft')
        elif any(word in name_lower for word in ['final', 'complete']):
            patterns.append('final version')
        elif any(word in name_lower for word in ['backup', 'copy']):
            patterns.append('backup/copy')
        
        # Work categories
        if any(word in name_lower for word in ['personal', 'private']):
            patterns.append('personal')
        elif any(word in name_lower for word in ['work', 'business', 'corporate']):
            patterns.append('business/work')
        
        return ', '.join(patterns) if patterns else ""
    
    async def _parse_advanced_response(self, response, files: List[FileInfo]) -> BatchProcessingResult:
        """Parse enhanced Gemini response with sophisticated confidence analysis."""
        try:
            text = response.text.strip()
            data = extract_json_array(text)
            
            if not isinstance(data, list):
                raise ValueError("Response is not a JSON array")
            
            # Create file map for lookup
            file_map = {f.path.name: f for f in files}
            results = []
            
            for item in data:
                filename = item.get("filename")
                if filename in file_map:
                    file_info = file_map[filename]
                    
                    # Calculate enhanced confidence score
                    base_confidence = float(item.get("confidence", 0.5))
                    confidence_factors = item.get("confidence_factors", {})
                    
                    # Enhanced validation and confidence adjustment
                    validation_result = self.response_validator.validate_response(item, file_info)
                    
                    # Validate and adjust confidence based on factors
                    adjusted_confidence = self._calculate_adjusted_confidence(
                        base_confidence, confidence_factors, item
                    )
                    
                    # Apply validation adjustments
                    adjusted_confidence += validation_result.confidence_adjustment
                    adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
                    
                    # Log validation issues if any
                    if validation_result.issues:
                        logger.debug(f"Validation issues for {filename}: {validation_result.issues}")
                    if validation_result.suggestions:
                        logger.debug(f"Validation suggestions for {filename}: {validation_result.suggestions}")
                    
                    ai_response = AIResponse(
                        file_path=file_info.path,
                        category=item.get("category", "Other"),
                        suggestion=item.get("suggestion", "Miscellaneous"),
                        confidence=adjusted_confidence,
                        reasoning=item.get("reasoning", "AI classification"),
                        tags=item.get("tags", []),
                        priority=int(item.get("priority", 3)),
                        relationships=item.get("relationships", []),
                        provider=self.provider_type
                    )
                    results.append(ai_response)
                    
                    # Enhanced metadata update with new fields
                    file_info.category = ai_response.category
                    file_info.suggestion = ai_response.suggestion
                    
                    try:
                        if hasattr(file_info.metadata, 'ai_confidence'):
                            file_info.metadata.ai_confidence = ai_response.confidence
                        if hasattr(file_info.metadata, 'ai_provider'):
                            file_info.metadata.ai_provider = self.provider_type.value
                        if hasattr(file_info.metadata, 'ai_description'):
                            file_info.metadata.ai_description = ai_response.reasoning
                        if hasattr(file_info.metadata, 'ai_tags'):
                            file_info.metadata.ai_tags = ai_response.tags
                    except Exception:
                        # If metadata is a dict - add enhanced fields
                        if isinstance(file_info.metadata, dict):
                            file_info.metadata.update({
                                "ai_confidence": ai_response.confidence,
                                "ai_reasoning": ai_response.reasoning,
                                "ai_tags": ai_response.tags,
                                "ai_priority": ai_response.priority,
                                "ai_provider": self.provider_type.value,
                                # Enhanced fields from new response format
                                "confidence_factors": confidence_factors,
                                "workflow_impact": item.get("workflow_impact", ""),
                                "alternative_suggestions": item.get("alternative_suggestions", []),
                                "archival_recommendation": item.get("archival_recommendation", ""),
                                "duplicate_likelihood": float(item.get("duplicate_likelihood", 0.0)),
                                "project_association": item.get("project_association", ""),
                                "seasonal_relevance": item.get("seasonal_relevance", ""),
                            })
            
            # Enhanced token and cost estimation
            estimated_tokens = len(text.split()) * 4
            # Account for enhanced prompt complexity
            estimated_tokens = int(estimated_tokens * 1.5)  # More complex responses
            estimated_cost = estimated_tokens * 0.00001
            
            # Quality assessment
            quality_assessment = self.quality_assessor.assess_batch_quality(results)
            
            logger.info(f"Successfully parsed {len(results)} enhanced AI responses")
            logger.info(f"Average confidence: {sum(r.confidence for r in results) / len(results):.2f}")
            logger.info(f"Quality assessment: {quality_assessment['overall_quality']} (score: {quality_assessment['score']:.2f})")
            
            return BatchProcessingResult(
                results=results,
                success=True,
                processing_time=0.0,  # Will be set by caller
                tokens_used=estimated_tokens,
                estimated_cost=estimated_cost,
                provider=self.provider_type
            )
            
        except Exception as e:
            logger.error(f"Failed to parse enhanced Gemini response: {e}")
            # Try fallback parsing for compatibility
            try:
                return await self._fallback_parse_response(response, files)
            except Exception as fallback_error:
                logger.error(f"Fallback parsing also failed: {fallback_error}")
                return BatchProcessingResult(
                    results=[],
                    success=False,
                    error_message=f"Primary parsing failed: {e}, Fallback failed: {fallback_error}",
                    provider=self.provider_type
                )
    
    def _calculate_adjusted_confidence(self, base_confidence: float, confidence_factors: Dict[str, float], item: Dict[str, Any]) -> float:
        """Calculate sophisticated confidence score based on multiple factors."""
        try:
            # If no confidence factors provided, return base confidence
            if not confidence_factors:
                return base_confidence
            
            # Weight different factors
            factor_weights = {
                "content_analysis": 0.35,
                "filename_patterns": 0.20,
                "temporal_relevance": 0.15,
                "relationship_strength": 0.15,
                "user_workflow_fit": 0.15
            }
            
            # Calculate weighted confidence
            weighted_score = 0.0
            total_weight = 0.0
            
            for factor, weight in factor_weights.items():
                if factor in confidence_factors:
                    factor_score = float(confidence_factors[factor])
                    weighted_score += factor_score * weight
                    total_weight += weight
            
            if total_weight > 0:
                adjusted_confidence = weighted_score / total_weight
                
                # Apply additional adjustments based on response quality
                quality_multiplier = self._assess_response_quality(item)
                adjusted_confidence *= quality_multiplier
                
                # Ensure confidence stays within bounds
                adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
                
                return adjusted_confidence
            
            return base_confidence
            
        except Exception as e:
            logger.warning(f"Failed to calculate adjusted confidence: {e}")
            return base_confidence
    
    def _assess_response_quality(self, item: Dict[str, Any]) -> float:
        """Assess the quality of the AI response to adjust confidence."""
        quality_score = 1.0
        
        # Check reasoning quality
        reasoning = item.get("reasoning", "")
        if len(reasoning) < 50:
            quality_score *= 0.9  # Short reasoning reduces confidence
        elif len(reasoning) > 200:
            quality_score *= 1.05  # Detailed reasoning increases confidence
        
        # Check tag relevance
        tags = item.get("tags", [])
        if len(tags) < 3:
            quality_score *= 0.95  # Few tags may indicate less analysis
        elif len(tags) > 8:
            quality_score *= 1.02  # More tags suggest thorough analysis
        
        # Check for specific quality indicators
        if any(indicator in reasoning.lower() for indicator in [
            "content analysis", "semantic", "workflow", "relationship", "pattern"
        ]):
            quality_score *= 1.03  # Advanced analysis indicators
        
        # Check suggestion specificity
        suggestion = item.get("suggestion", "")
        if "/" in suggestion and len(suggestion.split("/")) >= 2:
            quality_score *= 1.02  # Specific folder path structure
        
        return min(1.1, quality_score)  # Cap at 10% bonus
    
    async def _fallback_parse_response(self, response, files: List[FileInfo]) -> BatchProcessingResult:
        """Fallback parsing for compatibility with simpler response format."""
        try:
            text = response.text.strip()
            data = extract_json_array(text)
            
            file_map = {f.path.name: f for f in files}
            results = []
            
            for item in data:
                filename = item.get("filename")
                if filename in file_map:
                    file_info = file_map[filename]
                    
                    ai_response = AIResponse(
                        file_path=file_info.path,
                        category=item.get("category", "Other"),
                        suggestion=item.get("suggestion", "Miscellaneous"),
                        confidence=float(item.get("confidence", 0.5)),
                        reasoning=item.get("reasoning", "AI classification"),
                        tags=item.get("tags", []),
                        priority=int(item.get("priority", 3)),
                        relationships=item.get("relationships", []),
                        provider=self.provider_type
                    )
                    results.append(ai_response)
                    
                    # Basic metadata update
                    file_info.category = ai_response.category
                    file_info.suggestion = ai_response.suggestion
                    if isinstance(file_info.metadata, dict):
                        file_info.metadata.update({
                            "ai_confidence": ai_response.confidence,
                            "ai_reasoning": ai_response.reasoning,
                            "ai_tags": ai_response.tags,
                            "ai_priority": ai_response.priority,
                            "ai_provider": self.provider_type.value
                        })
            
            estimated_tokens = len(text.split()) * 4
            estimated_cost = estimated_tokens * 0.00001
            
            return BatchProcessingResult(
                results=results,
                success=True,
                tokens_used=estimated_tokens,
                estimated_cost=estimated_cost,
                provider=self.provider_type
            )
            
        except Exception as e:
            raise Exception(f"Fallback parsing failed: {e}")
    
    def estimate_cost(self, files: List[FileInfo]) -> float:
        """Estimate cost for processing files."""
        # Rough estimation based on content
        estimated_tokens = 0
        for file_info in files:
            estimated_tokens += 100  # Base tokens per file
            size_val = 0
            try:
                size_val = getattr(file_info.metadata, 'size_bytes', None)
                if size_val is None and isinstance(file_info.metadata, dict):
                    size_val = file_info.metadata.get("size", 0)
            except Exception:
                size_val = 0
            if (size_val or 0) > 10000:
                estimated_tokens += 500
        
        return estimated_tokens * 0.00001  # Rough cost per token
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get enhanced Gemini model information."""
        return {
            "provider": "Google Gemini (Enhanced)",
            "model": config.ai.gemini_model,
            "supports_images": True,
            "supports_documents": True,
            "max_tokens": 1000000,
            "cost_per_1k_tokens": 0.01,
            "enhancements": {
                "sophisticated_prompting": True,
                "multi_factor_confidence": True,
                "content_aware_analysis": True,
                "pattern_recognition": True,
                "validation_system": True,
                "quality_assessment": True,
                "feedback_learning": True
            },
            "features": {
                "context_analysis": ["file_patterns", "temporal_patterns", "content_insights"],
                "confidence_factors": ["content_analysis", "filename_patterns", "temporal_relevance", "relationship_strength", "user_workflow_fit"],
                "validation_rules": ["confidence_consistency", "suggestion_quality", "reasoning_depth", "tag_relevance", "category_appropriateness"],
                "quality_metrics": ["individual_scoring", "batch_assessment", "distribution_analysis"]
            }
        }

class OpenAIProvider(AIProviderBase):
    """OpenAI provider with GPT-4 capabilities."""
    
    def __init__(self):
        super().__init__(AIProvider.OPENAI)
        self.client = None
    
    async def initialize(self) -> bool:
        """Initialize OpenAI provider."""
        if not openai or not config.ai.openai_api_key:
            logger.warning("OpenAI not available: missing dependencies or API key")
            return False
        
        try:
            self.client = openai.AsyncOpenAI(api_key=config.ai.openai_api_key)
            self.is_available = True
            logger.info(f"OpenAI provider initialized successfully with model {config.ai.openai_model}")
            return True
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to initialize OpenAI: {e}")
            return False
    
    async def classify_batch(self, files: List[FileInfo], context: Dict[str, Any] = None) -> BatchProcessingResult:
        """Classify files using OpenAI."""
        start_time = time.time()
        
        async with self.rate_limiter:
            try:
                prompt = self._build_openai_prompt(files, context)
                
                response = await self.client.chat.completions.create(
                    model=config.ai.openai_model,
                    messages=[
                        {"role": "system", "content": "You are an expert file organization assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=4000
                )
                
                result = await self._parse_openai_response(response, files)
                
                # Update metrics
                processing_time = time.time() - start_time
                tokens_used = response.usage.total_tokens if response.usage else 0
                cost = tokens_used * 0.00001  # Rough estimate
                avg_confidence = sum(r.confidence for r in result.results) / len(result.results) if result.results else 0.0
                
                self.metrics.add_request(processing_time, True, avg_confidence, tokens_used, cost)
                
                logger.info(f"OpenAI processed {len(files)} files in {processing_time:.2f}s")
                return result
                
            except Exception as e:
                processing_time = time.time() - start_time
                self.metrics.add_request(processing_time, False)
                self.last_error = str(e)
                logger.error(f"OpenAI classification failed: {e}")
                raise
    
    def _build_openai_prompt(self, files: List[FileInfo], context: Dict[str, Any] = None) -> str:
        """Build prompt for OpenAI."""
        # Similar to Gemini but concise
        prompt = """Analyze the following files and provide intelligent organization suggestions.

Return a JSON array of objects with:
- filename
- category (Documents, Images, Videos, Audio, Code, Archives, Applications, Other)
- suggestion (FOLDER PATH ONLY, relative; never include the filename)
- confidence (0.0-1.0)
- reasoning (brief)
- tags (3-8)
- priority (1-5)
- folder_only: true
- matched_existing: string|null
- new_folder: boolean
- rule_hits: string[]
"""
        
        # Policies and structure
        if context and context.get("existing_structure"):
            prompt += "\nEXISTING STRUCTURE (reuse-first):\n" + json.dumps(context["existing_structure"], indent=2) + "\n"
        if context and context.get("policies"):
            prompt += "\nPOLICIES (must enforce):\n" + json.dumps(context["policies"], indent=2) + "\n"

        prompt += "\nFILES TO ANALYZE:\n"
        for file_info in files:
            try:
                ext = getattr(file_info.metadata, 'extension', None)
                if ext is None and isinstance(file_info.metadata, dict):
                    ext = file_info.metadata.get('extension', 'unknown')
            except Exception:
                ext = 'unknown'
            prompt += f"- {file_info.path.name} ({ext})\n"
        
        if context and context.get("user_prompt"):
            prompt += f"\nUser instructions: {context['user_prompt']}\n"
        
        prompt += """

RULES (must follow):
- suggestion must be a folder path only; never include the filename
- prefer reusing existing folders; set matched_existing when used; set new_folder only if needed
- use only the listed categories
- respond with only the JSON array
"""
        return prompt
    
    async def _parse_openai_response(self, response, files: List[FileInfo]) -> BatchProcessingResult:
        """Parse OpenAI response."""
        try:
            content = response.choices[0].message.content.strip()
            data = extract_json_array(content)
            
            # Process results similar to Gemini
            file_map = {f.path.name: f for f in files}
            results = []
            
            for item in data:
                filename = item.get("filename")
                if filename in file_map:
                    file_info = file_map[filename]
                    
                    ai_response = AIResponse(
                        file_path=file_info.path,
                        category=item.get("category", "Other"),
                        suggestion=item.get("suggestion", "Miscellaneous"),
                        confidence=float(item.get("confidence", 0.5)),
                        reasoning=item.get("reasoning", "AI classification"),
                        tags=item.get("tags", []),
                        priority=int(item.get("priority", 3)),
                        relationships=item.get("relationships", []),
                        provider=self.provider_type
                    )
                    results.append(ai_response)
                    
                    # Update file info
                    file_info.category = ai_response.category
                    file_info.suggestion = ai_response.suggestion
                    file_info.metadata.update({
                        "ai_confidence": ai_response.confidence,
                        "ai_reasoning": ai_response.reasoning,
                        "ai_tags": ai_response.tags,
                        "ai_priority": ai_response.priority,
                        "ai_provider": self.provider_type.value
                    })
            
            tokens_used = response.usage.total_tokens if response.usage else 0
            cost = tokens_used * 0.00001
            
            return BatchProcessingResult(
                results=results,
                success=True,
                tokens_used=tokens_used,
                estimated_cost=cost,
                provider=self.provider_type
            )
            
        except Exception as e:
            logger.error(f"Failed to parse OpenAI response: {e}")
            return BatchProcessingResult(
                results=[],
                success=False,
                error_message=str(e),
                provider=self.provider_type
            )
    
    def estimate_cost(self, files: List[FileInfo]) -> float:
        """Estimate OpenAI processing cost."""
        estimated_tokens = len(files) * 200  # Rough estimate
        return estimated_tokens * 0.00001
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information."""
        return {
            "provider": "OpenAI",
            "model": config.ai.openai_model,
            "supports_images": False,
            "supports_documents": True,
            "max_tokens": 128000,
            "cost_per_1k_tokens": 0.0001
        }

class MultiProviderAI:
    """Intelligent multi-provider AI system with fallback and optimization."""
    
    def __init__(self):
        self.providers: Dict[AIProvider, AIProviderBase] = {}
        self.primary_provider: Optional[AIProviderBase] = None
        self.fallback_providers: List[AIProviderBase] = []
        self.learning_data: Dict[str, Any] = {}
    
    async def initialize(self):
        """Initialize all available providers."""
        # Initialize shared cache once
        try:
            await smart_cache.initialize(config)
        except Exception as e:
            logger.warning(f"Cache initialization failed: {e}")

        # Initialize Gemini
        if config.ai.gemini_api_key:
            gemini = GeminiProvider()
            if await gemini.initialize():
                self.providers[AIProvider.GEMINI] = gemini
        
        # Initialize OpenAI
        if config.ai.openai_api_key:
            openai_provider = OpenAIProvider()
            if await openai_provider.initialize():
                self.providers[AIProvider.OPENAI] = openai_provider
        
        # TODO: Add Claude and Ollama providers
        
        # Set primary and fallback providers
        if config.ai.primary_provider in self.providers:
            self.primary_provider = self.providers[config.ai.primary_provider]
        elif self.providers:
            self.primary_provider = next(iter(self.providers.values()))
        
        self.fallback_providers = [p for p in self.providers.values() if p != self.primary_provider]
        
        logger.info(f"Initialized {len(self.providers)} AI providers")
        logger.info(f"Primary provider: {self.primary_provider.provider_type.value if self.primary_provider else 'None'}")
    
    async def classify_files(self, files: List[FileInfo], context: Dict[str, Any] = None) -> BatchProcessingResult:
        """Classify files using the best available provider."""
        if not self.primary_provider:
            raise RuntimeError("No AI providers available")
        
        # Try primary provider first
        try:
            result = await self.primary_provider.classify_batch(files, context)
            if result.success:
                return result
        except Exception as e:
            logger.warning(f"Primary provider {self.primary_provider.provider_type.value} failed: {e}")
        
        # Try fallback providers
        for provider in self.fallback_providers:
            try:
                logger.info(f"Trying fallback provider: {provider.provider_type.value}")
                result = await provider.classify_batch(files, context)
                if result.success:
                    return result
            except Exception as e:
                logger.warning(f"Fallback provider {provider.provider_type.value} failed: {e}")
        
        # If all providers fail, return empty result
        logger.error("All AI providers failed")
        return BatchProcessingResult(
            results=[],
            success=False,
            error_message="All AI providers failed",
            provider=AIProvider.GEMINI  # Default
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all providers."""
        return {
            "primary_provider": self.primary_provider.provider_type.value if self.primary_provider else None,
            "available_providers": [p.provider_type.value for p in self.providers.values() if p.is_available],
            "provider_stats": {p.provider_type.value: p.get_health_status() for p in self.providers.values()}
        }
    
    def estimate_total_cost(self, files: List[FileInfo]) -> Dict[str, float]:
        """Estimate costs across all providers."""
        return {
            provider.provider_type.value: provider.estimate_cost(files) 
            for provider in self.providers.values()
        }

# Global AI system instance
ai_system = MultiProviderAI()

__all__ = [
    "AIProviderBase",
    "GeminiProvider", 
    "OpenAIProvider",
    "MultiProviderAI",
    "ProcessingMetrics",
    "ai_system"
] 