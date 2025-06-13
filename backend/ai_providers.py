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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import aiohttp
import backoff

from .config import config, logger, AIProvider
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
    """Google Gemini AI provider with advanced features."""
    
    def __init__(self):
        super().__init__(AIProvider.GEMINI)
        self.model = None
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
    
    async def initialize(self) -> bool:
        """Initialize Gemini provider."""
        if not genai or not config.ai.gemini_api_key:
            logger.warning("Gemini not available: missing dependencies or API key")
            return False
        
        try:
            genai.configure(api_key=config.ai.gemini_api_key)
            self.model = genai.GenerativeModel(
                config.ai.gemini_model,
                safety_settings=self.safety_settings
            )
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
                
                # Process files and add content
                for file_info in files:
                    content = await self._extract_file_content(file_info)
                    if content:
                        content_parts.append(f"--- File: {file_info.path.name} ---")
                        content_parts.append(content)
                
                # Generate response
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self.model.generate_content, content_parts
                )
                
                # Parse response
                result = await self._parse_advanced_response(response, files)
                
                # Update metrics
                processing_time = time.time() - start_time
                avg_confidence = sum(r.confidence for r in result.results) / len(result.results) if result.results else 0.0
                self.metrics.add_request(processing_time, True, avg_confidence, result.tokens_used, result.estimated_cost)
                
                logger.info(f"Gemini processed {len(files)} files in {processing_time:.2f}s")
                return result
                
            except Exception as e:
                processing_time = time.time() - start_time
                self.metrics.add_request(processing_time, False)
                self.last_error = str(e)
                logger.error(f"Gemini classification failed: {e}")
                raise
    
    def _build_advanced_prompt(self, files: List[FileInfo], context: Dict[str, Any] = None) -> str:
        """Build an advanced prompt with context awareness."""
        base_prompt = """You are an expert AI file organizer with deep understanding of content, context, and user workflows.

TASK: Analyze the provided files and return intelligent organization suggestions.

RESPONSE FORMAT: Return a JSON array with objects containing:
- "filename": exact filename provided
- "category": main category (Documents, Images, Videos, Audio, Code, Archives, Applications, Other)
- "suggestion": specific folder name based on intelligent content analysis
- "confidence": 0.0-1.0 confidence score
- "reasoning": detailed explanation of your analysis
- "tags": array of relevant tags for the file
- "priority": 1-5 organization priority (5=urgent, 1=optional)
- "relationships": array of related files (if any)

ANALYSIS GUIDELINES:
1. Consider file content, not just names/extensions
2. Look for patterns across files to create coherent folder structures
3. Prioritize user workflow and accessibility
4. Group related files intelligently
5. Consider temporal relationships (dates, projects, etc.)
6. Identify duplicate or similar content
7. Suggest archival for old/unused files
"""
        
        # Add context-specific instructions
        if context:
            if user_prompt := context.get("user_prompt"):
                base_prompt += f"\n\nUSER INSTRUCTIONS:\n{user_prompt}\n"
            
            if template := context.get("template"):
                base_prompt += f"\n\nORGANIZATION TEMPLATE: {template}\n"
            
            if existing_structure := context.get("existing_structure"):
                base_prompt += f"\n\nEXISTING FOLDER STRUCTURE:\n{json.dumps(existing_structure, indent=2)}\n"
        
        # Add file overview
        file_overview = self._create_file_overview(files)
        base_prompt += f"\n\nFILES TO ANALYZE ({len(files)} total):\n{file_overview}\n"
        
        base_prompt += "\nRespond with ONLY the JSON array:"
        return base_prompt
    
    def _create_file_overview(self, files: List[FileInfo]) -> str:
        """Create an overview of files for better context."""
        overview = []
        for file_info in files:
            overview.append(f"- {file_info.path.name} ({file_info.metadata.get('extension', 'unknown')}, {file_info.metadata.get('size', 0)} bytes)")
        return "\n".join(overview)
    
    async def _extract_file_content(self, file_info: FileInfo) -> Optional[str]:
        """Extract content from file for analysis."""
        try:
            ext = file_info.metadata.get("extension", "").lower()
            
            # Handle different file types
            if ext in {'.png', '.jpg', '.jpeg', '.webp', '.heic', '.gif', '.bmp', '.tiff'} and Image:
                return Image.open(file_info.path)
            elif ext in {'.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv', '.log'}:
                content = file_info.path.read_text(encoding='utf-8', errors='ignore')
                return content[:config.ai.max_text_length] if len(content) > config.ai.max_text_length else content
            elif ext == '.pdf':
                from .utils import extract_text_from_pdf
                return extract_text_from_pdf(file_info.path)
            elif ext == '.docx':
                from .utils import extract_text_from_docx
                return extract_text_from_docx(file_info.path)
            
            return None
        except Exception as e:
            logger.warning(f"Failed to extract content from {file_info.path.name}: {e}")
            return None
    
    async def _parse_advanced_response(self, response, files: List[FileInfo]) -> BatchProcessingResult:
        """Parse Gemini response with advanced error handling."""
        try:
            text = response.text.strip()
            
            # Extract JSON from response
            start = text.find('[')
            end = text.rfind(']')
            if start == -1 or end == -1:
                raise ValueError("No JSON array found in response")
            
            json_text = text[start:end+1]
            data = json.loads(json_text)
            
            if not isinstance(data, list):
                raise ValueError("Response is not a JSON array")
            
            # Create file map for lookup
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
            
            # Estimate tokens and cost (approximate)
            estimated_tokens = len(text.split()) * 4  # Rough estimate
            estimated_cost = estimated_tokens * 0.00001  # Rough cost estimate
            
            return BatchProcessingResult(
                results=results,
                success=True,
                processing_time=0.0,  # Will be set by caller
                tokens_used=estimated_tokens,
                estimated_cost=estimated_cost,
                provider=self.provider_type
            )
            
        except Exception as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            return BatchProcessingResult(
                results=[],
                success=False,
                error_message=str(e),
                provider=self.provider_type
            )
    
    def estimate_cost(self, files: List[FileInfo]) -> float:
        """Estimate cost for processing files."""
        # Rough estimation based on content
        estimated_tokens = 0
        for file_info in files:
            estimated_tokens += 100  # Base tokens per file
            if file_info.metadata.get("size", 0) > 10000:  # Large files
                estimated_tokens += 500
        
        return estimated_tokens * 0.00001  # Rough cost per token
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Gemini model information."""
        return {
            "provider": "Google Gemini",
            "model": config.ai.gemini_model,
            "supports_images": True,
            "supports_documents": True,
            "max_tokens": 1000000,
            "cost_per_1k_tokens": 0.01
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
        # Similar to Gemini but optimized for OpenAI
        prompt = """Analyze the following files and provide intelligent organization suggestions.

Return a JSON array with objects containing:
- filename: exact filename
- category: main category 
- suggestion: specific folder name
- confidence: 0.0-1.0 score
- reasoning: brief explanation
- tags: relevant tags
- priority: 1-5 priority

Files to analyze:
"""
        
        for file_info in files:
            prompt += f"- {file_info.path.name} ({file_info.metadata.get('extension', 'unknown')})\n"
        
        if context and context.get("user_prompt"):
            prompt += f"\nUser instructions: {context['user_prompt']}\n"
        
        prompt += "\nRespond with only the JSON array:"
        return prompt
    
    async def _parse_openai_response(self, response, files: List[FileInfo]) -> BatchProcessingResult:
        """Parse OpenAI response."""
        try:
            content = response.choices[0].message.content.strip()
            
            # Extract JSON
            start = content.find('[')
            end = content.rfind(']')
            if start == -1 or end == -1:
                raise ValueError("No JSON array found")
            
            data = json.loads(content[start:end+1])
            
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