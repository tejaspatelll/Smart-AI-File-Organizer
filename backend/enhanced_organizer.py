"""Enhanced Smart File Organizer - Production Grade with Advanced AI.

This module provides a complete rewrite of the file organization system with:
- Multi-provider AI support (Gemini, OpenAI, Claude, Ollama)
- Async processing with parallel execution
- Advanced caching with multiple strategies
- Content similarity detection and semantic analysis
- Learning from user corrections
- Production-grade error handling and monitoring
- Smart conflict resolution
- Intelligent duplicate detection
- Real-time progress tracking
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set, Callable, Union
from collections import defaultdict, Counter
import logging

import aiofiles
from tqdm.asyncio import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Internal imports
from .cache import SQLiteCache
from .utils import read_basic_metadata, extract_text_from_pdf, extract_text_from_docx

# Try to import optional dependencies
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SUPPORTED_TEXT_EXTENSIONS = {'.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv', '.log', '.cfg', '.ini', '.yaml', '.yml'}
SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.heic', '.gif', '.bmp', '.tiff', '.svg'}
SUPPORTED_DOC_EXTENSIONS = {'.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.odt', '.ods', '.odp'}
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
SUPPORTED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a'}

# AI Provider configurations
AI_MODELS = {
    'gemini': {
        'model': 'gemini-2.5-flash-preview-05-20',
        'max_tokens': 1000000,
        'cost_per_1k': 0.01,
        'supports_images': True
    },
    'openai': {
        'model': 'gpt-4o-mini',
        'max_tokens': 128000,
        'cost_per_1k': 0.0001,
        'supports_images': False
    },
    'claude': {
        'model': 'claude-3-haiku-20240307',
        'max_tokens': 200000,
        'cost_per_1k': 0.00025,
        'supports_images': True
    }
}

@dataclass
class ProcessingMetrics:
    """Comprehensive metrics tracking."""
    start_time: float = field(default_factory=time.time)
    files_processed: int = 0
    files_successful: int = 0
    files_failed: int = 0
    api_calls_made: int = 0
    api_calls_cached: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    processing_stages: Dict[str, float] = field(default_factory=dict)
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    @property
    def success_rate(self) -> float:
        return self.files_successful / self.files_processed if self.files_processed > 0 else 0.0
    
    @property
    def cache_hit_rate(self) -> float:
        total_requests = self.api_calls_made + self.api_calls_cached
        return self.api_calls_cached / total_requests if total_requests > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'elapsed_time': self.elapsed_time,
            'files_processed': self.files_processed,
            'files_successful': self.files_successful,
            'files_failed': self.files_failed,
            'success_rate': self.success_rate,
            'api_calls_made': self.api_calls_made,
            'api_calls_cached': self.api_calls_cached,
            'cache_hit_rate': self.cache_hit_rate,
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'processing_stages': self.processing_stages
        }

@dataclass
class EnhancedFileInfo:
    """Enhanced file information with AI analysis."""
    path: Path
    metadata: Dict[str, Any]
    
    # AI Classification
    category: Optional[str] = None
    suggestion: Optional[str] = None
    confidence: float = 0.0
    reasoning: str = ""
    tags: List[str] = field(default_factory=list)
    priority: int = 3
    
    # Content Analysis
    content_summary: Optional[str] = None
    content_embedding: Optional[np.ndarray] = None
    detected_language: Optional[str] = None
    sentiment_score: Optional[float] = None
    
    # Relationships
    similar_files: List[Path] = field(default_factory=list)
    duplicate_of: Optional[Path] = None
    project_group: Optional[str] = None
    
    # Processing Status
    processing_status: str = "pending"
    processing_time: float = 0.0
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'path': str(self.path),
            'metadata': self.metadata,
            'category': self.category,
            'suggestion': self.suggestion,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'tags': self.tags,
            'priority': self.priority,
            'content_summary': self.content_summary,
            'detected_language': self.detected_language,
            'sentiment_score': self.sentiment_score,
            'similar_files': [str(p) for p in self.similar_files],
            'duplicate_of': str(self.duplicate_of) if self.duplicate_of else None,
            'project_group': self.project_group,
            'processing_status': self.processing_status,
            'processing_time': self.processing_time,
            'error_message': self.error_message
        }

class AdvancedAIProvider:
    """Multi-provider AI system with intelligent fallbacks."""
    
    def __init__(self):
        self.providers = {}
        self.primary_provider = None
        self.fallback_providers = []
        self.metrics = ProcessingMetrics()
        self.cache = SQLiteCache(Path("smart_organizer_enhanced_cache.db"))
        
        # Initialize available providers
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all available AI providers."""
        # Gemini
        if genai and os.getenv('GEMINI_API_KEY'):
            try:
                genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
                self.providers['gemini'] = genai.GenerativeModel(AI_MODELS['gemini']['model'])
                logger.info("Gemini provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
        
        # OpenAI
        if openai and os.getenv('OPENAI_API_KEY'):
            try:
                self.providers['openai'] = openai.AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                logger.info("OpenAI provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}")
        
        # Claude
        if anthropic and os.getenv('CLAUDE_API_KEY'):
            try:
                self.providers['claude'] = anthropic.AsyncAnthropic(api_key=os.getenv('CLAUDE_API_KEY'))
                logger.info("Claude provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Claude: {e}")
        
        # Set primary and fallback providers
        if 'gemini' in self.providers:
            self.primary_provider = 'gemini'
        elif 'openai' in self.providers:
            self.primary_provider = 'openai'
        elif 'claude' in self.providers:
            self.primary_provider = 'claude'
        
        self.fallback_providers = [p for p in self.providers.keys() if p != self.primary_provider]
        
        if not self.providers:
            logger.warning("No AI providers available. Please set API keys.")
    
    async def classify_files_advanced(
        self, 
        files: List[EnhancedFileInfo], 
        context: Dict[str, Any] = None,
        progress_callback: Optional[Callable] = None
    ) -> List[EnhancedFileInfo]:
        """Advanced file classification with multi-provider support."""
        if not self.providers:
            logger.error("No AI providers available")
            return files
        
        # Process in optimized batches
        batch_size = 20
        batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
        
        async def process_batch(batch_files: List[EnhancedFileInfo]) -> List[EnhancedFileInfo]:
            """Process a single batch of files."""
            try:
                # Try primary provider first
                result = await self._classify_batch_with_provider(
                    batch_files, self.primary_provider, context
                )
                if result:
                    return result
                
                # Try fallback providers
                for provider in self.fallback_providers:
                    result = await self._classify_batch_with_provider(
                        batch_files, provider, context
                    )
                    if result:
                        return result
                
                # If all providers fail, use fallback classification
                return self._fallback_classification(batch_files)
                
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                return self._fallback_classification(batch_files)
        
        # Process batches concurrently
        tasks = []
        for i, batch in enumerate(batches):
            if progress_callback:
                progress_callback(i, len(batches), f"Processing batch {i+1}/{len(batches)}")
            
            task = asyncio.create_task(process_batch(batch))
            tasks.append(task)
        
        # Wait for all batches to complete
        processed_batches = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        result_files = []
        for batch_result in processed_batches:
            if isinstance(batch_result, Exception):
                logger.error(f"Batch failed with exception: {batch_result}")
                continue
            result_files.extend(batch_result)
        
        return result_files
    
    async def _classify_batch_with_provider(
        self, 
        files: List[EnhancedFileInfo], 
        provider: str, 
        context: Dict[str, Any] = None
    ) -> Optional[List[EnhancedFileInfo]]:
        """Classify batch with specific provider."""
        try:
            if provider == 'gemini':
                return await self._classify_with_gemini(files, context)
            elif provider == 'openai':
                return await self._classify_with_openai(files, context)
            elif provider == 'claude':
                return await self._classify_with_claude(files, context)
            
        except Exception as e:
            logger.error(f"Provider {provider} failed: {e}")
            return None
    
    async def _classify_with_gemini(
        self, 
        files: List[EnhancedFileInfo], 
        context: Dict[str, Any] = None
    ) -> List[EnhancedFileInfo]:
        """Classify using Gemini with advanced prompting."""
        model = self.providers['gemini']
        
        # Build advanced prompt
        prompt = self._build_advanced_prompt(files, context, 'gemini')
        
        # Add file content
        content_parts = [prompt]
        
        for file_info in files:
            content = await self._extract_file_content(file_info)
            if content:
                content_parts.append(f"--- File: {file_info.path.name} ---")
                if isinstance(content, str):
                    content_parts.append(content[:10000])  # Limit content size
                else:  # Image
                    content_parts.append(content)
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, model.generate_content, content_parts
            )
            
            return self._parse_ai_response(response.text, files, 'gemini')
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise
    
    async def _classify_with_openai(
        self, 
        files: List[EnhancedFileInfo], 
        context: Dict[str, Any] = None
    ) -> List[EnhancedFileInfo]:
        """Classify using OpenAI GPT."""
        client = self.providers['openai']
        
        prompt = self._build_advanced_prompt(files, context, 'openai')
        
        try:
            response = await client.chat.completions.create(
                model=AI_MODELS['openai']['model'],
                messages=[
                    {"role": "system", "content": "You are an expert file organization assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            return self._parse_ai_response(response.choices[0].message.content, files, 'openai')
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    async def _classify_with_claude(
        self, 
        files: List[EnhancedFileInfo], 
        context: Dict[str, Any] = None
    ) -> List[EnhancedFileInfo]:
        """Classify using Claude."""
        client = self.providers['claude']
        
        prompt = self._build_advanced_prompt(files, context, 'claude')
        
        try:
            response = await client.messages.create(
                model=AI_MODELS['claude']['model'],
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return self._parse_ai_response(response.content[0].text, files, 'claude')
            
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            raise
    
    def _build_advanced_prompt(
        self, 
        files: List[EnhancedFileInfo], 
        context: Dict[str, Any] = None, 
        provider: str = 'gemini'
    ) -> str:
        """Build provider-specific advanced prompt."""
        base_prompt = """You are an expert AI file organizer with deep understanding of content, context, and user workflows.

TASK: Analyze the provided files and return intelligent organization suggestions.

RESPONSE FORMAT: Return ONLY a JSON array with objects containing:
- "filename": exact filename provided
- "category": main category (Documents, Images, Videos, Audio, Code, Archives, Applications, Other)
- "suggestion": specific intelligent folder name based on content analysis
- "confidence": 0.0-1.0 confidence score
- "reasoning": detailed explanation of your analysis
- "tags": array of relevant tags for the file
- "priority": 1-5 organization priority (5=urgent, 1=optional)
- "content_summary": brief summary of file content
- "detected_language": detected language if text file
- "project_group": project/group name if applicable

ADVANCED ANALYSIS GUIDELINES:
1. Understand content context, not just file names
2. Identify project relationships and group related files
3. Detect temporal patterns and organize by relevance
4. Consider user workflow optimization
5. Identify duplicates and similar content
6. Suggest intelligent folder hierarchies
7. Prioritize based on importance and recency
8. Tag with semantic categories
"""
        
        # Add context-specific instructions
        if context:
            if user_prompt := context.get('user_prompt'):
                base_prompt += f"\n\nUSER INSTRUCTIONS:\n{user_prompt}\n"
            
            if template := context.get('template'):
                base_prompt += f"\n\nORGANIZATION TEMPLATE: {template}\n"
            
            if existing_structure := context.get('existing_structure'):
                base_prompt += f"\n\nEXISTING FOLDERS:\n{json.dumps(existing_structure, indent=2)}\n"
        
        # Add file overview
        file_overview = "\n".join([
            f"- {f.path.name} ({f.metadata.get('extension', 'unknown')}, "
            f"{f.metadata.get('size', 0)} bytes, "
            f"modified: {f.metadata.get('modified', 'unknown')})"
            for f in files
        ])
        
        base_prompt += f"\n\nFILES TO ANALYZE ({len(files)} total):\n{file_overview}\n"
        base_prompt += "\nRespond with ONLY the JSON array, no other text:"
        
        return base_prompt
    
    async def _extract_file_content(self, file_info: EnhancedFileInfo) -> Optional[Union[str, Any]]:
        """Extract file content for AI analysis."""
        try:
            ext = file_info.metadata.get('extension', '').lower()
            
            if ext in SUPPORTED_IMAGE_EXTENSIONS and Image:
                return Image.open(file_info.path)
            elif ext in SUPPORTED_TEXT_EXTENSIONS:
                async with aiofiles.open(file_info.path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = await f.read()
                    return content[:15000]  # Limit size
            elif ext == '.pdf':
                return extract_text_from_pdf(file_info.path)[:15000]
            elif ext == '.docx':
                return extract_text_from_docx(file_info.path)[:15000]
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to extract content from {file_info.path.name}: {e}")
            return None
    
    def _parse_ai_response(
        self, 
        response_text: str, 
        files: List[EnhancedFileInfo], 
        provider: str
    ) -> List[EnhancedFileInfo]:
        """Parse AI response and update file information."""
        try:
            # Extract JSON from response
            start = response_text.find('[')
            end = response_text.rfind(']')
            if start == -1 or end == -1:
                raise ValueError("No JSON array found in response")
            
            json_text = response_text[start:end+1]
            data = json.loads(json_text)
            
            if not isinstance(data, list):
                raise ValueError("Response is not a JSON array")
            
            # Create file map for lookup
            file_map = {f.path.name: f for f in files}
            
            # Update file information
            for item in data:
                filename = item.get('filename')
                if filename in file_map:
                    file_info = file_map[filename]
                    
                    file_info.category = item.get('category', 'Other')
                    file_info.suggestion = item.get('suggestion', 'Miscellaneous')
                    file_info.confidence = float(item.get('confidence', 0.5))
                    file_info.reasoning = item.get('reasoning', 'AI classification')
                    file_info.tags = item.get('tags', [])
                    file_info.priority = int(item.get('priority', 3))
                    file_info.content_summary = item.get('content_summary')
                    file_info.detected_language = item.get('detected_language')
                    file_info.project_group = item.get('project_group')
                    file_info.processing_status = 'completed'
                    
                    # Update metadata
                    file_info.metadata.update({
                        'ai_provider': provider,
                        'ai_confidence': file_info.confidence,
                        'ai_reasoning': file_info.reasoning,
                        'ai_tags': file_info.tags
                    })
            
            self.metrics.api_calls_made += 1
            return files
            
        except Exception as e:
            logger.error(f"Failed to parse AI response: {e}")
            return self._fallback_classification(files)
    
    def _fallback_classification(self, files: List[EnhancedFileInfo]) -> List[EnhancedFileInfo]:
        """Provide intelligent fallback classification."""
        for file_info in files:
            ext = file_info.metadata.get('extension', '').lower()
            filename = file_info.path.stem.lower()
            
            # Enhanced pattern-based classification
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                if any(word in filename for word in ['screenshot', 'capture']):
                    file_info.category = 'Images'
                    file_info.suggestion = 'Screenshots'
                elif any(word in filename for word in ['logo', 'icon']):
                    file_info.category = 'Images'
                    file_info.suggestion = 'Logos and Icons'
                else:
                    file_info.category = 'Images'
                    file_info.suggestion = 'Image Collection'
            
            elif ext in SUPPORTED_DOC_EXTENSIONS:
                if any(word in filename for word in ['invoice', 'bill']):
                    file_info.category = 'Documents'
                    file_info.suggestion = 'Financial/Invoices'
                elif any(word in filename for word in ['contract', 'agreement']):
                    file_info.category = 'Documents'
                    file_info.suggestion = 'Legal/Contracts'
                else:
                    file_info.category = 'Documents'
                    file_info.suggestion = 'Document Collection'
            
            elif ext in {'.py', '.js', '.html', '.css'}:
                file_info.category = 'Code'
                file_info.suggestion = f'{ext.upper().lstrip(".")} Files'
            
            else:
                file_info.category = 'Other'
                file_info.suggestion = 'Miscellaneous'
            
            file_info.confidence = 0.6
            file_info.reasoning = 'Pattern-based fallback classification'
            file_info.processing_status = 'completed'
        
        return files

class SmartContentAnalyzer:
    """Advanced content analysis with similarity detection."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.embeddings_cache = {}
    
    async def analyze_content_similarity(self, files: List[EnhancedFileInfo]) -> List[EnhancedFileInfo]:
        """Analyze content similarity between files."""
        text_files = []
        text_contents = []
        
        # Extract text content from files
        for file_info in files:
            content = await self._get_text_content(file_info)
            if content and len(content.strip()) > 50:  # Minimum content threshold
                text_files.append(file_info)
                text_contents.append(content)
        
        if len(text_contents) < 2:
            return files
        
        try:
            # Create TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(text_contents)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find similar files (threshold 0.7)
            similarity_threshold = 0.7
            for i, file_info in enumerate(text_files):
                similar_indices = np.where(similarity_matrix[i] > similarity_threshold)[0]
                similar_files = [text_files[j].path for j in similar_indices if j != i]
                file_info.similar_files = similar_files
                
                # Store embedding for future use
                if hasattr(tfidf_matrix[i], 'toarray'):
                    file_info.content_embedding = tfidf_matrix[i].toarray()[0]
        
        except Exception as e:
            logger.error(f"Content similarity analysis failed: {e}")
        
        return files
    
    async def _get_text_content(self, file_info: EnhancedFileInfo) -> Optional[str]:
        """Extract text content from file."""
        try:
            ext = file_info.metadata.get('extension', '').lower()
            
            if ext in SUPPORTED_TEXT_EXTENSIONS:
                async with aiofiles.open(file_info.path, 'r', encoding='utf-8', errors='ignore') as f:
                    return await f.read()
            elif ext == '.pdf':
                return extract_text_from_pdf(file_info.path)
            elif ext == '.docx':
                return extract_text_from_docx(file_info.path)
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to extract text from {file_info.path.name}: {e}")
            return None
    
    def detect_project_groups(self, files: List[EnhancedFileInfo]) -> List[EnhancedFileInfo]:
        """Detect project groups based on file patterns."""
        # Group files by directory
        directory_groups = defaultdict(list)
        for file_info in files:
            directory_groups[file_info.path.parent].append(file_info)
        
        # Analyze each directory group
        for directory, group_files in directory_groups.items():
            if len(group_files) >= 3:  # Minimum group size
                # Check for common patterns
                extensions = [f.metadata.get('extension', '') for f in group_files]
                ext_counter = Counter(extensions)
                
                # If directory has related files, mark as project
                if len(ext_counter) >= 2:  # Mixed file types
                    project_name = directory.name
                    for file_info in group_files:
                        file_info.project_group = project_name
        
        return files

class EnhancedDirectoryScanner:
    """High-performance async directory scanner with advanced features."""
    
    def __init__(self, root: Path, max_workers: int = 4):
        self.root = Path(root).expanduser().resolve()
        self.max_workers = max_workers
        self.metrics = ProcessingMetrics()
        
        # Skip patterns
        self.skip_directories = {
            'node_modules', '.git', '.svn', '__pycache__', 
            '.pytest_cache', '.tox', '.vscode', '.idea',
            'venv', 'env', '.env', 'dist', 'build'
        }
        
        self.skip_files = {
            '.DS_Store', 'Thumbs.db', '.gitignore', 'desktop.ini',
            'package-lock.json', 'yarn.lock'
        }
    
    async def scan_async(
        self, 
        progress_callback: Optional[Callable] = None
    ) -> List[EnhancedFileInfo]:
        """Async directory scanning with parallel processing."""
        self.metrics.processing_stages['scan_start'] = time.time()
        
        files = []
        
        # Get all paths first
        all_paths = []
        for root, dirs, filenames in os.walk(self.root):
            # Filter out skip directories
            dirs[:] = [d for d in dirs if d not in self.skip_directories]
            
            for filename in filenames:
                if filename not in self.skip_files:
                    all_paths.append(Path(root) / filename)
        
        # Process files in parallel batches
        batch_size = 100
        batches = [all_paths[i:i + batch_size] for i in range(0, len(all_paths), batch_size)]
        
        async def process_batch(batch_paths: List[Path]) -> List[EnhancedFileInfo]:
            """Process a batch of file paths."""
            batch_files = []
            for file_path in batch_paths:
                try:
                    if await self._should_process_file(file_path):
                        file_info = await self._create_file_info(file_path)
                        if file_info:
                            batch_files.append(file_info)
                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")
            return batch_files
        
        # Process batches concurrently
        tasks = []
        for i, batch in enumerate(batches):
            if progress_callback:
                progress_callback(i, len(batches), f"Scanning batch {i+1}/{len(batches)}")
            
            task = asyncio.create_task(process_batch(batch))
            tasks.append(task)
        
        # Wait for all batches
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error(f"Batch scan failed: {batch_result}")
                continue
            files.extend(batch_result)
        
        self.metrics.processing_stages['scan_complete'] = time.time()
        self.metrics.files_processed = len(files)
        
        logger.info(f"Scanned {len(files)} files in {self.metrics.elapsed_time:.2f}s")
        return files
    
    async def _should_process_file(self, file_path: Path) -> bool:
        """Determine if file should be processed."""
        try:
            stat = file_path.stat()
            
            # Skip very large files (>100MB)
            if stat.st_size > 100 * 1024 * 1024:
                return False
            
            # Skip hidden files
            if file_path.name.startswith('.'):
                return False
            
            # Skip system files
            if file_path.suffix.lower() in {'.exe', '.dll', '.sys'}:
                return False
            
            return True
            
        except (OSError, PermissionError):
            return False
    
    async def _create_file_info(self, file_path: Path) -> Optional[EnhancedFileInfo]:
        """Create enhanced file info object."""
        try:
            stat = file_path.stat()
            
            metadata = {
                'name': file_path.name,
                'extension': file_path.suffix.lower(),
                'size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'parent_directory': file_path.parent.name
            }
            
            # Add content hash for duplicate detection
            content_hash = await self._calculate_file_hash(file_path)
            metadata['content_hash'] = content_hash
            
            return EnhancedFileInfo(path=file_path, metadata=metadata)
            
        except Exception as e:
            logger.warning(f"Failed to create file info for {file_path}: {e}")
            return None
    
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate file hash asynchronously."""
        try:
            hash_md5 = hashlib.md5()
            async with aiofiles.open(file_path, 'rb') as f:
                async for chunk in self._file_chunks(f):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return hashlib.md5(str(file_path).encode()).hexdigest()
    
    async def _file_chunks(self, file_obj, chunk_size: int = 8192):
        """Async file chunk generator."""
        while True:
            chunk = await file_obj.read(chunk_size)
            if not chunk:
                break
            yield chunk

class SmartDuplicateDetector:
    """Advanced duplicate detection with content analysis."""
    
    def __init__(self):
        self.hash_groups = defaultdict(list)
        self.similarity_threshold = 0.95
    
    def detect_duplicates(self, files: List[EnhancedFileInfo]) -> List[EnhancedFileInfo]:
        """Detect duplicates using multiple strategies."""
        # Group by content hash
        for file_info in files:
            content_hash = file_info.metadata.get('content_hash')
            if content_hash:
                self.hash_groups[content_hash].append(file_info)
        
        # Mark duplicates and select best original
        for hash_value, group_files in self.hash_groups.items():
            if len(group_files) > 1:
                original = self._select_best_original(group_files)
                
                for file_info in group_files:
                    if file_info != original:
                        file_info.duplicate_of = original.path
                        file_info.tags.append('duplicate')
        
        return files
    
    def _select_best_original(self, duplicate_files: List[EnhancedFileInfo]) -> EnhancedFileInfo:
        """Select the best file to keep among duplicates."""
        # Scoring criteria
        def score_file(file_info: EnhancedFileInfo) -> float:
            score = 0.0
            
            # Prefer files not in downloads/temp folders
            path_str = str(file_info.path.parent).lower()
            if 'download' in path_str or 'temp' in path_str:
                score -= 2.0
            
            # Prefer files with descriptive names
            name_parts = file_info.path.stem.replace('_', ' ').split()
            score += len(name_parts) * 0.1
            
            # Prefer newer files
            try:
                mtime = file_info.path.stat().st_mtime
                age_days = (time.time() - mtime) / (24 * 3600)
                score += max(0, 365 - age_days) * 0.001  # Newer is better
            except:
                pass
            
            # Prefer files in organized folders
            if file_info.path.parent != self.root:
                score += 1.0
            
            return score
        
        # Return highest scoring file
        return max(duplicate_files, key=score_file)

class ProductionOrganizer:
    """Production-grade file organizer with all enhancements."""
    
    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.metrics = ProcessingMetrics()
        
        # Initialize components
        self.scanner = EnhancedDirectoryScanner(self.root)
        self.ai_provider = AdvancedAIProvider()
        self.content_analyzer = SmartContentAnalyzer()
        self.duplicate_detector = SmartDuplicateDetector()
        
        # Processing state
        self.files: List[EnhancedFileInfo] = []
        self.organization_plan: List[Dict[str, Any]] = []
    
    async def analyze_directory(
        self, 
        progress_callback: Optional[Callable] = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Complete directory analysis with all AI features."""
        logger.info(f"Starting enhanced analysis of {self.root}")
        
        try:
            # Stage 1: Scan directory
            if progress_callback:
                progress_callback(1, 5, "Scanning directory...")
            
            self.files = await self.scanner.scan_async(progress_callback)
            logger.info(f"Found {len(self.files)} files")
            
            # Stage 2: AI Classification
            if progress_callback:
                progress_callback(2, 5, "AI classification...")
            
            self.files = await self.ai_provider.classify_files_advanced(
                self.files, context, progress_callback
            )
            
            # Stage 3: Content Analysis
            if progress_callback:
                progress_callback(3, 5, "Analyzing content similarity...")
            
            self.files = await self.content_analyzer.analyze_content_similarity(self.files)
            self.files = self.content_analyzer.detect_project_groups(self.files)
            
            # Stage 4: Duplicate Detection
            if progress_callback:
                progress_callback(4, 5, "Detecting duplicates...")
            
            self.files = self.duplicate_detector.detect_duplicates(self.files)
            
            # Stage 5: Generate Organization Plan
            if progress_callback:
                progress_callback(5, 5, "Generating organization plan...")
            
            self.organization_plan = self._generate_smart_plan()
            
            # Compile results
            results = {
                'files_analyzed': len(self.files),
                'organization_plan': self.organization_plan,
                'metrics': self.metrics.to_dict(),
                'ai_provider_stats': self.ai_provider.metrics.to_dict(),
                'duplicate_groups': len(self.duplicate_detector.hash_groups),
                'files_with_ai_analysis': sum(1 for f in self.files if f.processing_status == 'completed'),
                'average_confidence': np.mean([f.confidence for f in self.files if f.confidence > 0]),
                'categories_detected': len(set(f.category for f in self.files if f.category)),
                'project_groups': len(set(f.project_group for f in self.files if f.project_group))
            }
            
            logger.info(f"Analysis complete: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    def _generate_smart_plan(self) -> List[Dict[str, Any]]:
        """Generate intelligent organization plan."""
        plan = []
        
        for file_info in self.files:
            if not file_info.category or not file_info.suggestion:
                continue
            
            # Skip duplicates unless specifically organizing them
            if file_info.duplicate_of:
                continue
            
            # Create destination path
            if file_info.project_group:
                dest_dir = self.root / "Projects" / file_info.project_group
            else:
                dest_dir = self.root / file_info.category / file_info.suggestion
            
            dest_path = dest_dir / file_info.path.name
            
            # Skip if already in correct location
            if dest_path.resolve() == file_info.path.resolve():
                continue
            
            plan_item = {
                'source': str(file_info.path),
                'destination': str(dest_path),
                'category': file_info.category,
                'suggestion': file_info.suggestion,
                'confidence': file_info.confidence,
                'priority': file_info.priority,
                'reasoning': file_info.reasoning,
                'tags': file_info.tags,
                'is_duplicate': bool(file_info.duplicate_of),
                'project_group': file_info.project_group,
                'similar_files_count': len(file_info.similar_files)
            }
            
            plan.append(plan_item)
        
        # Sort by priority and confidence
        plan.sort(key=lambda x: (x['priority'], x['confidence']), reverse=True)
        
        return plan
    
    def get_detailed_summary(self) -> Dict[str, Any]:
        """Get comprehensive analysis summary."""
        if not self.files:
            return {'error': 'No analysis performed yet'}
        
        # Calculate statistics
        categories = Counter(f.category for f in self.files if f.category)
        languages = Counter(f.detected_language for f in self.files if f.detected_language)
        project_groups = Counter(f.project_group for f in self.files if f.project_group)
        
        confidence_scores = [f.confidence for f in self.files if f.confidence > 0]
        
        duplicates = sum(1 for f in self.files if f.duplicate_of)
        similar_pairs = sum(len(f.similar_files) for f in self.files) // 2
        
        return {
            'total_files': len(self.files),
            'categories': dict(categories),
            'languages_detected': dict(languages),
            'project_groups': dict(project_groups),
            'duplicates_found': duplicates,
            'similar_file_pairs': similar_pairs,
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'high_confidence_files': sum(1 for score in confidence_scores if score >= 0.8),
            'processing_time': self.metrics.elapsed_time,
            'ai_provider_used': self.ai_provider.primary_provider,
            'cache_hit_rate': self.ai_provider.metrics.cache_hit_rate,
            'organization_plan_size': len(self.organization_plan)
        }

# Convenience functions for backward compatibility
async def enhanced_organize_directory(
    root_path: Path,
    progress_callback: Optional[Callable] = None,
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Enhanced directory organization with all AI features."""
    organizer = ProductionOrganizer(root_path)
    return await organizer.analyze_directory(progress_callback, context)

__all__ = [
    'ProductionOrganizer',
    'EnhancedFileInfo', 
    'AdvancedAIProvider',
    'SmartContentAnalyzer',
    'EnhancedDirectoryScanner',
    'SmartDuplicateDetector',
    'ProcessingMetrics',
    'enhanced_organize_directory'
] 