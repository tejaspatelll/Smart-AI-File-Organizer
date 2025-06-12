"""Core organizing logic for Smart File Organizer.

This module provides high-level classes:
    • DirectoryScanner – enumerate files in a folder recursively.
    • GeminiClassifier – call Gemini to get content-based categories.
    • FileOrganizer – combine metadata + Gemini suggestions and plan moves.

Actual file moves are only performed in `apply_plan`, allowing a safe dry-run preview.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set, Callable
from collections import defaultdict, Counter

from tqdm import tqdm

from .utils import read_basic_metadata, extract_text_from_pdf, extract_text_from_docx

try:
    import google.generativeai as genai
    from PIL import Image
except ImportError:
    genai = None
    Image = None

# --- Constants ---
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
SUPPORTED_TEXT_EXTENSIONS = {'.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv', '.log', '.cfg', '.ini', '.yaml', '.yml'}
SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.heic', '.gif', '.bmp', '.tiff', '.svg'}
SUPPORTED_DOC_EXTENSIONS = {'.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.odt', '.ods', '.odp'}
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
SUPPORTED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a'}
BATCH_SIZE = 25  # Number of files to process in one API call

# Temporary/system files to skip
SKIP_FILES = {'.DS_Store', 'Thumbs.db', '.gitignore', '.gitkeep', 'desktop.ini', '.localized'}
SKIP_EXTENSIONS = {'.tmp', '.temp', '.lock', '.log', '.cache', '.bak', '.swp', '.swo'}


@dataclass
class FileInfo:
    path: Path
    metadata: Dict[str, Any]
    category: str | None = None
    suggestion: str | None = None


class DirectoryScanner:
    """Recursively collect all file paths under a given directory with duplicate detection."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"Path {self.root} is not a directory")
        self.file_hashes: Dict[str, List[Path]] = defaultdict(list)  # hash -> list of files with that hash

    def scan(self, progress_callback: Callable[[int, int, str], None] | None = None) -> List[FileInfo]:
        """Scan directory and gather file info."""
        # Directories to skip entirely
        SKIP_DIRECTORIES = {
            'node_modules', '.git', '.svn', '.hg', '__pycache__', 
            '.pytest_cache', '.tox', '.coverage', '.vscode', '.idea',
            'venv', 'env', '.env', 'virtualenv', '.virtualenv',
            'dist', 'build', '.next', '.nuxt', 'coverage',
            '.DS_Store', 'Thumbs.db', '.Spotlight-V100', '.Trashes',
            'System Volume Information', '$RECYCLE.BIN',
            'Electron.app', '.app', 'Contents', 'MacOS', 'Resources',
            'Frameworks', 'Libraries', 'PlugIns', 'XPCServices'
        }
        
        files = []
        all_files = list(self.root.rglob('*'))
        
        for file_path in tqdm(all_files, desc="Scanning files", unit="file", leave=False, file=sys.stderr):
            # Skip if any parent directory is in skip list
            if any(part.name in SKIP_DIRECTORIES for part in file_path.parents):
                continue
            
            # Skip the file itself if it's a directory or shouldn't be scanned
            if file_path.is_dir() or not self._should_scan_file(file_path):
                continue
            
            if progress_callback:
                progress_callback(len(files), len(all_files), f"Scanning {file_path.name}")
            
            try:
                # Compute file metadata
                stat = file_path.stat()
                metadata = {
                    "extension": file_path.suffix.lower(),
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
                
                file_info = FileInfo(path=file_path, metadata=metadata)
                files.append(file_info)
                
                # Track file hash for duplicate detection
                file_hash = self._compute_file_hash(file_path)
                self.file_hashes[file_hash].append(file_path)
                
            except (OSError, PermissionError):
                # Skip files we can't access
                continue
        
        return files
    
    def _should_scan_file(self, file_path: Path) -> bool:
        """Determine if a file should be scanned based on name and extension."""
        # Extended list of files to skip
        SKIP_FILES_EXTENDED = {
            '.DS_Store', 'Thumbs.db', 'desktop.ini', '.gitignore', '.gitkeep',
            '.env', '.env.local', '.env.development', '.env.production',
            'package-lock.json', 'yarn.lock', 'composer.lock', 'Pipfile.lock',
            'requirements.txt', 'setup.py', 'setup.cfg', 'pyproject.toml',
            'Makefile', 'Dockerfile', 'docker-compose.yml', '.dockerignore',
            'README.md', 'LICENSE', 'CHANGELOG.md', 'CONTRIBUTING.md',
            '.prettierrc', '.eslintrc', '.babelrc', 'tsconfig.json',
            'webpack.config.js', 'rollup.config.js', 'vite.config.js',
            '.travis.yml', '.github', 'appveyor.yml', 'circle.yml'
        }
        
        # Extended list of extensions to skip
        SKIP_EXTENSIONS_EXTENDED = {
            '.exe', '.dll', '.so', '.dylib', '.a', '.lib',
            '.pyc', '.pyo', '.pyd', '.class', '.jar',
            '.log', '.tmp', '.temp', '.cache', '.lock',
            '.pid', '.swap', '.bak', '.backup',
            '.node', '.wasm', '.map', '.d.ts'
        }
        
        if file_path.name in SKIP_FILES_EXTENDED:
            return False
        
        ext = file_path.suffix.lower()
        if ext in SKIP_EXTENSIONS_EXTENDED:
            return False
            
        # Skip hidden files (starting with .) except common ones
        if file_path.name.startswith('.') and file_path.name not in {'.gitignore', '.gitkeep'}:
            return False
            
        # Skip very large files (>100MB) for performance
        try:
            if file_path.stat().st_size > 100 * 1024 * 1024:
                return False
        except OSError:
            return False
            
        # Skip files that are clearly system or development related
        filename_lower = file_path.name.lower()
        skip_patterns = [
            'node_modules', 'package', 'webpack', 'babel', 'eslint',
            'prettier', 'tsconfig', 'jest', 'cypress', 'selenium'
        ]
        
        if any(pattern in filename_lower for pattern in skip_patterns):
            return False
            
        return True
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of file for duplicate detection."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            # If we can't read the file, return a unique hash based on path
            return hashlib.md5(str(file_path).encode()).hexdigest()
    
    def get_duplicates(self) -> Dict[str, List[Path]]:
        """Return a dictionary of hash -> file paths for files with duplicates."""
        return {h: paths for h, paths in self.file_hashes.items() if len(paths) > 1}


class GeminiClassifier:
    """Classify files via Google Gemini using batch processing."""

    def __init__(self, api_key: str | None = None) -> None:
        if genai is None or Image is None:
            raise RuntimeError("Required packages not found. Run 'pip install google-generativeai pillow'")
        
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(MODEL_NAME)

    def _get_file_content_parts(self, file_info: FileInfo) -> Any:
        """Extract content from file for Gemini."""
        ext = file_info.metadata.get("extension", "")
        
        try:
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                return Image.open(file_info.path)
            elif ext in SUPPORTED_TEXT_EXTENSIONS:
                return file_info.path.read_text(encoding='utf-8', errors='ignore')
            elif ext == '.pdf':
                return extract_text_from_pdf(file_info.path)
            elif ext == '.docx':
                return extract_text_from_docx(file_info.path)
        except Exception as e:
            tqdm.write(f"Warning: Could not read {file_info.path.name}: {e}", file=sys.stderr)
        return None

    def classify_batch(self, files: List[FileInfo], progress_callback: Callable[[int, int, str], None] | None = None) -> List[FileInfo]:
        """Classify a list of files in batches to reduce API calls."""
        all_supported_extensions = (SUPPORTED_TEXT_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS | 
                                   SUPPORTED_DOC_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS | 
                                   SUPPORTED_AUDIO_EXTENSIONS)
        
        files_to_process = [f for f in files if f.metadata.get("extension") in all_supported_extensions]
        files_to_fallback = [f for f in files if f not in files_to_process]

        total_batches = (len(files_to_process) + BATCH_SIZE - 1) // BATCH_SIZE

        # Process supported files in batches
        for i, batch_start_index in enumerate(tqdm(range(0, len(files_to_process), BATCH_SIZE), desc="Analyzing batches", unit="batch", leave=False, file=sys.stderr)):
            batch = files_to_process[batch_start_index:batch_start_index + BATCH_SIZE]
            if progress_callback:
                current_file_name = batch[0].path.name if batch else ""
                progress_callback(i + 1, total_batches, f"Analyzing batch {i+1}/{total_batches} ({current_file_name}...)")
            self._process_one_batch(batch)

        # Apply fallback for unsupported files
        for file in files_to_fallback:
            file.category, file.suggestion = self._fallback_classification(file)

        return files

    def _process_one_batch(self, batch: List[FileInfo]):
        """Builds a prompt, calls the API, and parses the response for a single batch."""
        prompt_parts = [self._build_batch_prompt_header()]
        
        files_in_batch = []
        for file in batch:
            content = self._get_file_content_parts(file)
            if content:
                # Truncate large text files to avoid hitting token limits
                if isinstance(content, str) and len(content) > 15000:
                    content = content[:15000] + "\n... (truncated)"
                
                # Include metadata context for better classification
                metadata_context = self._format_metadata_context(file)
                prompt_parts.append(f"--- File Name: `{file.path.name}` ---")
                prompt_parts.append(f"Metadata: {metadata_context}")
                prompt_parts.append(content)
                files_in_batch.append(file)
        
        if not files_in_batch:
            return # All files in batch were unreadable

        try:
            response = self.model.generate_content(prompt_parts)
            self._parse_batch_response(response, files_in_batch)
        except Exception as e:
            tqdm.write(f"Error classifying batch: {e}. Using fallback for {len(files_in_batch)} files.", file=sys.stderr)
            for file in files_in_batch:
                file.category, file.suggestion = self._fallback_classification(file)
    
    def _format_metadata_context(self, file_info: FileInfo) -> str:
        """Format file metadata for better AI classification."""
        meta = file_info.metadata
        context_parts = []
        
        if 'size' in meta:
            size_mb = meta['size'] / (1024 * 1024)
            context_parts.append(f"Size: {size_mb:.1f}MB")
        
        if 'created' in meta:
            context_parts.append(f"Created: {meta['created']}")
        
        if 'modified' in meta:
            context_parts.append(f"Modified: {meta['modified']}")
            
        if 'extension' in meta:
            context_parts.append(f"Type: {meta['extension']}")
        
        return " | ".join(context_parts)

    def _build_batch_prompt_header(self) -> str:
        return """
You are an expert file organizer with deep understanding of file types and content context. I will provide files with their metadata and content.

Analyze each file and return a single JSON array. Each object MUST have these keys:
1. "filename": The exact filename I provided
2. "category": Choose from: Documents, Images, Videos, Audio, Code, Archives, Other
3. "suggestion": A descriptive folder name based on content analysis (be specific and meaningful)

Guidelines for suggestions:
- For documents: Use content themes (e.g., "Financial Reports 2024", "Meeting Notes", "Contracts")
- For images: Use subject/context (e.g., "Vacation Photos", "Product Screenshots", "Family Events")
- For code: Use project/language (e.g., "Python Scripts", "Web Frontend", "Database Queries")
- For media: Use content type (e.g., "Music Collection", "Training Videos", "Podcasts")
- Be specific but not overly granular - aim for folders that would contain 3-20 related files

Example response:
[
  {"filename": "invoice_jan_2024.pdf", "category": "Documents", "suggestion": "Invoices 2024"},
  {"filename": "family_vacation.jpg", "category": "Images", "suggestion": "Family Vacation Photos"},
  {"filename": "app.py", "category": "Code", "suggestion": "Python Applications"}
]

Respond with ONLY the JSON array, no other text:
"""

    def _parse_batch_response(self, response: genai.GenerationResponse, batch_files: List[FileInfo]):
        try:
            text = response.text.strip()
            # Clean the response to be valid JSON
            start = text.find('[')
            end = text.rfind(']')
            if start == -1 or end == -1:
                raise json.JSONDecodeError("No JSON array found in response", text, 0)
            
            json_text = text[start:end+1]
            data = json.loads(json_text)

            if not isinstance(data, list):
                 raise TypeError("Response is not a JSON list.")

            # Create a mapping from filename to FileInfo object for easy lookup
            file_map = {f.path.name: f for f in batch_files}

            for item in data:
                filename = item.get("filename")
                file_info = file_map.get(filename)
                
                if file_info:
                    file_info.category = item.get("category", "Other")
                    file_info.suggestion = item.get("suggestion", self._fallback_classification(file_info)[1])
                    # Mark as processed
                    file_map.pop(filename)

            # Any files remaining in the map were not in the Gemini response, so use fallback
            for unhandled_file in file_map.values():
                unhandled_file.category, unhandled_file.suggestion = self._fallback_classification(unhandled_file)

        except (json.JSONDecodeError, AttributeError, ValueError, TypeError) as e:
            tqdm.write(f"Warning: Could not parse Gemini batch response: {e}. Raw: {response.text}. Using fallback for this batch.", file=sys.stderr)
            for file in batch_files:
                file.category, file.suggestion = self._fallback_classification(file)

    def _fallback_classification(self, file_info: FileInfo) -> Tuple[str, str]:
        """Enhanced fallback to extension-based classification with smarter suggestions."""
        ext = file_info.metadata.get("extension", "").lower()
        filename = file_info.path.stem.lower()
        
        # Smart classification based on extension
        if ext in SUPPORTED_IMAGE_EXTENSIONS:
            # Try to infer image type from filename
            if any(word in filename for word in ['screenshot', 'screen_shot', 'capture']):
                return "Images", "Screenshots"
            elif any(word in filename for word in ['avatar', 'profile', 'headshot']):
                return "Images", "Profile Pictures"
            elif any(word in filename for word in ['logo', 'icon', 'brand']):
                return "Images", "Logos and Icons"
            else:
                return "Images", "Image Collection"
                
        elif ext in SUPPORTED_DOC_EXTENSIONS:
            # Try to infer document type from filename
            if any(word in filename for word in ['invoice', 'bill', 'receipt']):
                return "Documents", "Invoices and Bills"
            elif any(word in filename for word in ['contract', 'agreement', 'terms']):
                return "Documents", "Contracts and Legal"
            elif any(word in filename for word in ['report', 'analysis', 'summary']):
                return "Documents", "Reports and Analysis"
            elif any(word in filename for word in ['resume', 'cv', 'curriculum']):
                return "Documents", "Resume and CV"
            else:
                return "Documents", "Document Collection"
                
        elif ext in SUPPORTED_TEXT_EXTENSIONS:
            if ext in {'.py', '.js', '.html', '.css', '.json', '.xml', '.yaml', '.yml'}:
                return "Code", f"{ext.upper().lstrip('.')} Files"
            elif any(word in filename for word in ['config', 'settings', 'conf']):
                return "Code", "Configuration Files"
            else:
                return "Documents", "Text Files"
                
        elif ext in SUPPORTED_VIDEO_EXTENSIONS:
            if any(word in filename for word in ['tutorial', 'training', 'course']):
                return "Videos", "Training Videos"
            elif any(word in filename for word in ['meeting', 'call', 'conference']):
                return "Videos", "Meeting Recordings"
            else:
                return "Videos", "Video Collection"
                
        elif ext in SUPPORTED_AUDIO_EXTENSIONS:
            if any(word in filename for word in ['music', 'song', 'album']):
                return "Audio", "Music Collection"
            elif any(word in filename for word in ['podcast', 'interview', 'talk']):
                return "Audio", "Podcasts and Talks"
            elif any(word in filename for word in ['recording', 'audio', 'voice']):
                return "Audio", "Audio Recordings"
            else:
                return "Audio", "Audio Files"
                
        elif ext in {'.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz'}:
            return "Archives", "Compressed Files"
            
        elif ext in {'.exe', '.msi', '.dmg', '.pkg', '.deb', '.rpm'}:
            return "Applications", "Software and Installers"
            
        else:
            # Try to categorize based on filename patterns
            if any(word in filename for word in ['backup', 'bak', 'old']):
                return "Archives", "Backup Files"
            elif any(word in filename for word in ['temp', 'temporary', 'tmp']):
                return "Other", "Temporary Files"
            else:
                return "Other", f"{ext.upper().lstrip('.') if ext else 'Unknown'} Files"


class CustomPromptClassifier:
    """Enhanced classifier that uses custom user prompts for smarter organization."""

    def __init__(self, api_key: str | None = None) -> None:
        if genai is None:
            raise RuntimeError("Required packages not found. Run 'pip install google-generativeai'")
        
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(MODEL_NAME)
        self.user_prompt = ""
        self.organization_template = ""

    def set_custom_prompt(self, user_prompt: str) -> None:
        """Set custom user instructions for organization."""
        self.user_prompt = user_prompt.strip()

    def set_organization_template(self, template_type: str) -> None:
        """Set predefined organization template."""
        templates = {
            "creative": """
            Organize files for a creative professional:
            - Group by project type (design, photography, video, etc.)
            - Separate client work from personal projects
            - Create asset libraries for reusable elements
            - Archive old projects by year
            - Keep work-in-progress separate from finished work
            """,
            "business": """
            Organize files for business use:
            - Separate by department (Finance, HR, Marketing, Operations)
            - Group documents by importance (Critical, Important, Archive)
            - Organize by date and project
            - Keep meeting notes and presentations accessible
            - Separate internal from client-facing documents
            """,
            "student": """
            Organize files for academic use:
            - Group by subject/course
            - Separate by academic year and semester
            - Keep research materials organized by topic
            - Archive completed assignments
            - Separate personal study materials from official documents
            """,
            "personal": """
            Organize files for personal use:
            - Group photos by events and dates
            - Separate important documents (financial, legal, medical)
            - Organize by life categories (work, family, hobbies, travel)
            - Keep frequently accessed files easily available
            - Archive old files while keeping recent ones accessible
            """
        }
        self.organization_template = templates.get(template_type, "")

    def _build_enhanced_prompt_header(self) -> str:
        """Build AI prompt with custom user instructions."""
        base_prompt = """
You are an expert file organizer with deep understanding of file types, content, and user workflow patterns.

"""
        
        if self.user_prompt:
            base_prompt += f"""
USER'S CUSTOM ORGANIZATION INSTRUCTIONS:
{self.user_prompt}

"""
        
        if self.organization_template:
            base_prompt += f"""
ORGANIZATION TEMPLATE TO FOLLOW:
{self.organization_template}

"""
        
        base_prompt += """
I will provide files with their metadata and content. Analyze each file and return a single JSON array.

For each file, provide:
1. "filename": The exact filename I provided
2. "category": Choose from: Documents, Images, Videos, Audio, Code, Archives, Applications, Other
3. "suggestion": A specific folder name based on the user's instructions and content analysis
4. "reasoning": Explain why this organization makes sense for the user's workflow
5. "priority": Rate 1-5 how important this file move is (5 = must organize, 1 = optional)
6. "confidence": Rate 0.0-1.0 how confident you are about this categorization

CRITICAL GUIDELINES:
- Follow the user's specific instructions above everything else
- Be intelligent about understanding the user's intent, not just literal instructions
- Create folder structures that make sense for the user's workflow
- Consider file relationships and how they're used together
- Prioritize moves that create the most value for organization
- Be specific with folder names but not overly granular
- Group related files together intelligently

Example response:
[
  {
    "filename": "project_proposal.pdf",
    "category": "Documents", 
    "suggestion": "Active Projects/Client ABC/Proposals",
    "reasoning": "Groups with related project files for easy access during active work",
    "priority": 4,
    "confidence": 0.9
  }
]

Respond with ONLY the JSON array, no other text:
"""
        return base_prompt

    def classify_batch_with_prompt(self, files: List[FileInfo], progress_callback: Callable[[int, int, str], None] | None = None) -> List[FileInfo]:
        """Classify files using custom prompt or template."""
        all_supported_extensions = (SUPPORTED_TEXT_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS | 
                                   SUPPORTED_DOC_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS | 
                                   SUPPORTED_AUDIO_EXTENSIONS)
        
        files_to_process = [f for f in files if f.metadata.get("extension") in all_supported_extensions]
        files_to_fallback = [f for f in files if f not in files_to_process]

        total_batches = (len(files_to_process) + BATCH_SIZE - 1) // BATCH_SIZE

        # Process supported files in batches
        for i, batch_start_index in enumerate(tqdm(range(0, len(files_to_process), BATCH_SIZE), desc="AI analyzing with custom instructions", unit="batch", leave=False, file=sys.stderr)):
            batch = files_to_process[batch_start_index:batch_start_index + BATCH_SIZE]
            if progress_callback:
                current_file_name = batch[0].path.name if batch else ""
                progress_callback(i + 1, total_batches, f"Processing batch {i+1}/{total_batches} with custom logic ({current_file_name}...)")
            self._process_enhanced_batch(batch)

        # Apply intelligent fallback for unsupported files
        for file in files_to_fallback:
            file.category, file.suggestion = self._intelligent_fallback_classification(file)
            file.metadata['ai_reasoning'] = "Classified using intelligent pattern recognition"
            file.metadata['ai_priority'] = 3
            file.metadata['ai_confidence'] = 0.7

        return files

    def _process_enhanced_batch(self, batch: List[FileInfo]):
        """Process batch with enhanced custom prompt logic."""
        prompt_parts = [self._build_enhanced_prompt_header()]
        
        files_in_batch = []
        for file in batch:
            content = self._get_file_content_parts(file)
            if content:
                # Truncate large text files
                if isinstance(content, str) and len(content) > 15000:
                    content = content[:15000] + "\n... (truncated)"
                
                # Enhanced metadata context
                metadata_context = self._format_enhanced_metadata_context(file)
                prompt_parts.append(f"--- File Name: `{file.path.name}` ---")
                prompt_parts.append(f"Metadata: {metadata_context}")
                prompt_parts.append(content)
                files_in_batch.append(file)
        
        if not files_in_batch:
            return

        try:
            response = self.model.generate_content(prompt_parts)
            self._parse_enhanced_response(response, files_in_batch)
        except Exception as e:
            tqdm.write(f"Error in enhanced classification: {e}. Using intelligent fallback.", file=sys.stderr)
            for file in files_in_batch:
                file.category, file.suggestion = self._intelligent_fallback_classification(file)
                file.metadata['ai_reasoning'] = "Fallback classification due to API error"
                file.metadata['ai_priority'] = 2
                file.metadata['ai_confidence'] = 0.6

    def _get_file_content_parts(self, file_info: FileInfo) -> Any:
        """Extract content from file for analysis (same as GeminiClassifier)."""
        ext = file_info.metadata.get("extension", "")
        
        try:
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                return Image.open(file_info.path)
            elif ext in SUPPORTED_TEXT_EXTENSIONS:
                return file_info.path.read_text(encoding='utf-8', errors='ignore')
            elif ext == '.pdf':
                return extract_text_from_pdf(file_info.path)
            elif ext == '.docx':
                return extract_text_from_docx(file_info.path)
        except Exception as e:
            tqdm.write(f"Warning: Could not read {file_info.path.name}: {e}", file=sys.stderr)
        return None

    def _format_enhanced_metadata_context(self, file_info: FileInfo) -> str:
        """Enhanced metadata formatting for better AI understanding."""
        meta = file_info.metadata
        context_parts = []
        
        # File details
        if 'size' in meta:
            size_mb = meta['size'] / (1024 * 1024)
            if size_mb > 1:
                context_parts.append(f"Size: {size_mb:.1f}MB")
            else:
                size_kb = meta['size'] / 1024
                context_parts.append(f"Size: {size_kb:.1f}KB")
        
        # Timestamps
        if 'created' in meta:
            context_parts.append(f"Created: {meta['created']}")
        if 'modified' in meta:
            context_parts.append(f"Modified: {meta['modified']}")
            
        # File type and location
        if 'extension' in meta:
            context_parts.append(f"Type: {meta['extension']}")
        
        # Current location context (helps AI understand existing organization)
        current_dir = file_info.path.parent.name
        if current_dir != str(file_info.path.parents[1].name):  # Not in root
            context_parts.append(f"Current folder: {current_dir}")
        
        return " | ".join(context_parts)

    def _parse_enhanced_response(self, response: genai.GenerationResponse, batch_files: List[FileInfo]):
        """Parse enhanced AI response with custom fields."""
        try:
            text = response.text.strip()
            start = text.find('[')
            end = text.rfind(']')
            if start == -1 or end == -1:
                raise json.JSONDecodeError("No JSON array found in response", text, 0)
            
            json_text = text[start:end+1]
            data = json.loads(json_text)

            if not isinstance(data, list):
                raise TypeError("Response is not a JSON list.")

            file_map = {f.path.name: f for f in batch_files}

            for item in data:
                filename = item.get("filename")
                file_info = file_map.get(filename)
                
                if file_info:
                    file_info.category = item.get("category", "Other")
                    file_info.suggestion = item.get("suggestion", self._intelligent_fallback_classification(file_info)[1])
                    # Store enhanced AI metadata
                    file_info.metadata['ai_reasoning'] = item.get("reasoning", "AI-powered classification")
                    file_info.metadata['ai_priority'] = item.get("priority", 3)
                    file_info.metadata['ai_confidence'] = item.get("confidence", 0.5)
                    file_map.pop(filename)

            # Handle unprocessed files
            for unhandled_file in file_map.values():
                unhandled_file.category, unhandled_file.suggestion = self._intelligent_fallback_classification(unhandled_file)
                unhandled_file.metadata['ai_reasoning'] = "Not found in AI response, using intelligent fallback"
                unhandled_file.metadata['ai_priority'] = 2
                unhandled_file.metadata['ai_confidence'] = 0.5

        except (json.JSONDecodeError, AttributeError, ValueError, TypeError) as e:
            tqdm.write(f"Warning: Could not parse enhanced AI response: {e}. Using intelligent fallback.", file=sys.stderr)
            for file in batch_files:
                file.category, file.suggestion = self._intelligent_fallback_classification(file)

    def _intelligent_fallback_classification(self, file_info: FileInfo) -> Tuple[str, str]:
        """Enhanced fallback with smarter pattern recognition."""
        ext = file_info.metadata.get("extension", "").lower()
        filename = file_info.path.stem.lower()
        current_folder = file_info.path.parent.name.lower()
        
        # Apply user prompt logic to fallback if possible
        suggestion = self._apply_prompt_logic_to_fallback(file_info, ext, filename)
        if suggestion:
            category = self._determine_category_from_extension(ext)
            return category, suggestion
        
        # Enhanced pattern recognition
        if ext in SUPPORTED_IMAGE_EXTENSIONS:
            return self._classify_image_intelligently(filename, current_folder)
        elif ext in SUPPORTED_DOC_EXTENSIONS:
            return self._classify_document_intelligently(filename, current_folder)
        elif ext in SUPPORTED_TEXT_EXTENSIONS:
            return self._classify_text_intelligently(filename, ext, current_folder)
        elif ext in SUPPORTED_VIDEO_EXTENSIONS:
            return self._classify_video_intelligently(filename, current_folder)
        elif ext in SUPPORTED_AUDIO_EXTENSIONS:
            return self._classify_audio_intelligently(filename, current_folder)
        else:
            return self._classify_other_intelligently(filename, ext, current_folder)

    def _apply_prompt_logic_to_fallback(self, file_info: FileInfo, ext: str, filename: str) -> str | None:
        """Try to apply user prompt logic to fallback classification."""
        if not self.user_prompt:
            return None
        
        prompt_lower = self.user_prompt.lower()
        
        # Look for specific keywords in user prompt
        if "project" in prompt_lower and any(word in filename for word in ["project", "proj"]):
            return "Active Projects"
        elif "client" in prompt_lower and any(word in filename for word in ["client", "customer"]):
            return "Client Work"
        elif "work" in prompt_lower and "personal" in prompt_lower:
            if any(word in filename for word in ["work", "office", "business"]):
                return "Work Documents"
            elif any(word in filename for word in ["personal", "family", "home"]):
                return "Personal Files"
        elif "date" in prompt_lower or "year" in prompt_lower:
            # Try to extract year from filename
            import re
            year_match = re.search(r'20\d{2}', filename)
            if year_match:
                return f"Archive {year_match.group()}"
        
        return None

    def _determine_category_from_extension(self, ext: str) -> str:
        """Determine category from file extension."""
        if ext in SUPPORTED_IMAGE_EXTENSIONS:
            return "Images"
        elif ext in SUPPORTED_DOC_EXTENSIONS:
            return "Documents"
        elif ext in SUPPORTED_TEXT_EXTENSIONS:
            return "Code" if ext in {'.py', '.js', '.html', '.css', '.json'} else "Documents"
        elif ext in SUPPORTED_VIDEO_EXTENSIONS:
            return "Videos"
        elif ext in SUPPORTED_AUDIO_EXTENSIONS:
            return "Audio"
        else:
            return "Other"

    def _classify_image_intelligently(self, filename: str, current_folder: str) -> Tuple[str, str]:
        """Intelligent image classification."""
        # Screenshots
        if any(word in filename for word in ['screenshot', 'screen_shot', 'capture', 'grab']):
            return "Images", "Screenshots"
        
        # Profile/avatar images
        if any(word in filename for word in ['avatar', 'profile', 'headshot', 'portrait']):
            return "Images", "Profile Pictures"
        
        # Logos and branding
        if any(word in filename for word in ['logo', 'icon', 'brand', 'badge']):
            return "Images", "Logos and Icons"
        
        # Social media content
        if any(word in filename for word in ['instagram', 'facebook', 'twitter', 'social']):
            return "Images", "Social Media"
        
        # Product images
        if any(word in filename for word in ['product', 'catalog', 'inventory']):
            return "Images", "Product Photos"
        
        # Date-based organization
        import re
        year_match = re.search(r'20\d{2}', filename)
        if year_match:
            return "Images", f"Photos {year_match.group()}"
        
        # Event-based
        if any(word in filename for word in ['vacation', 'trip', 'travel', 'holiday']):
            return "Images", "Travel Photos"
        elif any(word in filename for word in ['wedding', 'party', 'event', 'celebration']):
            return "Images", "Events"
        elif any(word in filename for word in ['family', 'kids', 'children']):
            return "Images", "Family Photos"
        
        return "Images", "Photo Collection"

    def _classify_document_intelligently(self, filename: str, current_folder: str) -> Tuple[str, str]:
        """Intelligent document classification."""
        # Financial documents
        if any(word in filename for word in ['invoice', 'bill', 'receipt', 'payment']):
            return "Documents", "Financial/Invoices"
        elif any(word in filename for word in ['tax', 'irs', 'w2', '1099']):
            return "Documents", "Financial/Tax Documents"
        elif any(word in filename for word in ['bank', 'statement', 'account']):
            return "Documents", "Financial/Bank Statements"
        
        # Legal documents
        elif any(word in filename for word in ['contract', 'agreement', 'legal', 'terms']):
            return "Documents", "Legal/Contracts"
        
        # Work-related
        elif any(word in filename for word in ['report', 'analysis', 'presentation', 'proposal']):
            return "Documents", "Work/Reports"
        elif any(word in filename for word in ['meeting', 'notes', 'minutes']):
            return "Documents", "Work/Meeting Notes"
        
        # Personal documents
        elif any(word in filename for word in ['resume', 'cv', 'curriculum']):
            return "Documents", "Personal/Resume"
        elif any(word in filename for word in ['medical', 'health', 'doctor']):
            return "Documents", "Personal/Medical"
        elif any(word in filename for word in ['insurance', 'policy']):
            return "Documents", "Personal/Insurance"
        
        # Academic
        elif any(word in filename for word in ['research', 'paper', 'thesis', 'study']):
            return "Documents", "Academic/Research"
        elif any(word in filename for word in ['assignment', 'homework', 'project']):
            return "Documents", "Academic/Assignments"
        
        return "Documents", "Document Collection"

    def _classify_text_intelligently(self, filename: str, ext: str, current_folder: str) -> Tuple[str, str]:
        """Intelligent text file classification."""
        # Code files
        if ext in {'.py', '.js', '.html', '.css', '.json', '.xml', '.yaml', '.yml'}:
            if any(word in filename for word in ['config', 'settings', 'conf']):
                return "Code", "Configuration"
            elif any(word in filename for word in ['test', 'spec']):
                return "Code", "Tests"
            elif ext == '.py':
                return "Code", "Python Scripts"
            elif ext in {'.js', '.html', '.css'}:
                return "Code", "Web Development"
            else:
                return "Code", f"{ext.upper().lstrip('.')} Files"
        
        # Documentation
        elif ext in {'.md', '.txt'}:
            if any(word in filename for word in ['readme', 'doc', 'documentation']):
                return "Documents", "Documentation"
            elif any(word in filename for word in ['note', 'notes']):
                return "Documents", "Notes"
            else:
                return "Documents", "Text Files"
        
        return "Documents", "Text Files"

    def _classify_video_intelligently(self, filename: str, current_folder: str) -> Tuple[str, str]:
        """Intelligent video classification."""
        if any(word in filename for word in ['tutorial', 'training', 'course', 'lesson']):
            return "Videos", "Educational"
        elif any(word in filename for word in ['meeting', 'call', 'conference', 'zoom']):
            return "Videos", "Meetings"
        elif any(word in filename for word in ['family', 'vacation', 'trip', 'personal']):
            return "Videos", "Personal Videos"
        elif any(word in filename for word in ['work', 'project', 'presentation']):
            return "Videos", "Work Videos"
        else:
            return "Videos", "Video Collection"

    def _classify_audio_intelligently(self, filename: str, current_folder: str) -> Tuple[str, str]:
        """Intelligent audio classification."""
        if any(word in filename for word in ['music', 'song', 'album', 'track']):
            return "Audio", "Music"
        elif any(word in filename for word in ['podcast', 'interview', 'talk']):
            return "Audio", "Podcasts"
        elif any(word in filename for word in ['recording', 'voice', 'memo']):
            return "Audio", "Voice Recordings"
        elif any(word in filename for word in ['meeting', 'call']):
            return "Audio", "Meeting Recordings"
        else:
            return "Audio", "Audio Files"

    def _classify_other_intelligently(self, filename: str, ext: str, current_folder: str) -> Tuple[str, str]:
        """Intelligent classification for other file types."""
        # Archives
        if ext in {'.zip', '.rar', '.7z', '.tar', '.gz', '.bz2'}:
            if any(word in filename for word in ['backup', 'archive']):
                return "Archives", "Backups"
            else:
                return "Archives", "Compressed Files"
        
        # Applications
        elif ext in {'.exe', '.msi', '.dmg', '.pkg', '.deb', '.rpm'}:
            return "Applications", "Software"
        
        # Temporary files
        elif any(word in filename for word in ['temp', 'tmp', 'backup', 'old']):
            return "Other", "Temporary Files"
        
        else:
            return "Other", f"{ext.upper().lstrip('.') if ext else 'Unknown'} Files"


@dataclass
class MovePlanItem:
    source: Path
    destination: Path
    priority: int = 1  # 1-5 scale, 5 being highest priority
    confidence: float = 0.0  # 0.0-1.0 scale, 1.0 being highest confidence
    reason: str = ""  # Explanation for the move
    is_duplicate: bool = False  # Whether this file is a duplicate
    original_file: Path | None = None  # Path to original if duplicate


class FileOrganizer:
    """Prepare and apply organizing plans based on metadata & classification, with smart folder reuse and duplicate handling."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.plan: List[MovePlanItem] = []
        self.existing_folders: set[str] = set()
        self.folder_file_map: dict[str, set[str]] = {}
        self.duplicates: Dict[str, List[Path]] = {}
        self.move_history: List[Dict[str, str]] = []  # For undo functionality

    def _scan_existing_folders(self):
        """Scan all folders and build a comprehensive map of existing structure."""
        self.existing_folders = set()
        self.folder_file_map = {}
        
        for dirpath, dirnames, filenames in os.walk(self.root):
            if dirpath == str(self.root):
                continue  # Skip root directory itself
                
            folder_name = os.path.basename(dirpath)
            folder_name_lower = folder_name.lower()
            
            # Store both original and lowercase for matching
            self.existing_folders.add(folder_name_lower)
            
            # Map folder to its contained files (by extension and name patterns)
            if folder_name_lower not in self.folder_file_map:
                self.folder_file_map[folder_name_lower] = {
                    'files': set(),
                    'extensions': set(),
                    'original_name': folder_name,
                    'file_count': 0
                }
                
            for filename in filenames:
                filename_lower = filename.lower()
                ext = Path(filename).suffix.lower()
                
                self.folder_file_map[folder_name_lower]['files'].add(filename_lower)
                self.folder_file_map[folder_name_lower]['extensions'].add(ext)
                self.folder_file_map[folder_name_lower]['file_count'] += 1

    def _find_best_existing_folder(self, suggestion: str, filename: str, category: str, file_ext: str) -> str | None:
        """Enhanced folder matching with similarity scoring."""
        suggestion_lower = suggestion.lower()
        filename_lower = filename.lower()
        
        best_match = None
        best_score = 0.0
        
        for folder_lower, folder_data in self.folder_file_map.items():
            score = 0.0
            
            # Exact suggestion match (highest priority)
            if suggestion_lower == folder_lower:
                return folder_data['original_name']
            
            # Substring matches
            if suggestion_lower in folder_lower or folder_lower in suggestion_lower:
                score += 0.7
            
            # Category compatibility
            if category.lower() in folder_lower:
                score += 0.5
            
            # File extension compatibility
            if file_ext in folder_data['extensions']:
                score += 0.4
            
            # Similar file patterns
            filename_words = set(filename_lower.replace('_', ' ').replace('-', ' ').split())
            folder_words = set(folder_lower.replace('_', ' ').replace('-', ' ').split())
            
            if filename_words & folder_words:  # Common words
                score += 0.3
            
            # Prefer folders with reasonable file counts (not too empty, not overcrowded)
            file_count = folder_data['file_count']
            if 3 <= file_count <= 50:
                score += 0.2
            elif file_count > 100:
                score -= 0.2
            
            if score > best_score and score >= 0.6:  # Minimum threshold
                best_score = score
                best_match = folder_data['original_name']
        
        return best_match
    
    def _handle_duplicates(self, files: List[FileInfo], scanner: DirectoryScanner) -> List[FileInfo]:
        """Identify and mark duplicate files, keeping the best version."""
        self.duplicates = scanner.get_duplicates()
        
        if not self.duplicates:
            return files
        
        # Create a set of files to mark as duplicates
        files_to_mark = set()
        
        for file_hash, duplicate_paths in self.duplicates.items():
            if len(duplicate_paths) < 2:
                continue
                
            # Choose the best file to keep based on several criteria
            best_file = self._choose_best_duplicate(duplicate_paths)
            
            # Mark others as duplicates
            for path in duplicate_paths:
                if path != best_file:
                    files_to_mark.add(path)
        
        # Update FileInfo objects
        for file_info in files:
            if file_info.path in files_to_mark:
                file_info.metadata['is_duplicate'] = True
                
        return files
    
    def _choose_best_duplicate(self, duplicate_paths: List[Path]) -> Path:
        """Choose the best file among duplicates based on location, name, and metadata."""
        scored_files = []
        
        for path in duplicate_paths:
            score = 0
            
            # Prefer files in organized folders (not root)
            if path.parent != self.root:
                score += 2
            
            # Prefer files with descriptive names (longer, more words)
            name_parts = path.stem.replace('_', ' ').replace('-', ' ').split()
            score += len(name_parts) * 0.5
            
            # Prefer files not in temp/download-like folders
            path_str = str(path.parent).lower()
            if any(word in path_str for word in ['download', 'temp', 'trash', 'recycle']):
                score -= 3
            
            # Prefer files with newer modification times
            try:
                mtime = path.stat().st_mtime
                score += mtime / 1000000  # Small boost for newer files
            except OSError:
                pass
            
            scored_files.append((score, path))
        
        # Return the highest scored file
        return max(scored_files, key=lambda x: x[0])[1]

    def build_plan(self, files: List[FileInfo], scanner: DirectoryScanner | None = None) -> List[MovePlanItem]:
        """Generate comprehensive organization plan with duplicate handling."""
        self._scan_existing_folders()
        
        # Handle duplicates if scanner provided
        if scanner:
            files = self._handle_duplicates(files, scanner)
        
        plan: List[MovePlanItem] = []
        
        for fi in files:
            if not fi.category or not fi.suggestion:
                continue

            # Skip files marked as duplicates by default
            if fi.metadata.get('is_duplicate', False):
                # Create a special duplicate handling plan item
                duplicate_plan = self._create_duplicate_plan_item(fi, scanner)
                if duplicate_plan:
                    plan.append(duplicate_plan)
                continue

            # Smart folder matching with enhanced logic
            best_folder = self._find_best_existing_folder(
                fi.suggestion, 
                fi.path.name, 
                fi.category,
                fi.metadata.get("extension", "")
            )
            
            if best_folder:
                dest_dir = self.root / fi.category / best_folder
            else:
                dest_dir = self.root / fi.category / fi.suggestion
                
            dest = dest_dir / fi.path.name

            # Skip if already in the right place
            if dest.resolve() == fi.path.resolve():
                continue

            # Enhanced logic to avoid unnecessary moves
            if self._should_skip_move(fi):
                continue

            # Calculate enhanced priority and confidence
            priority, confidence, reason = self._calculate_enhanced_priority(fi, best_folder is not None)
            
            plan.append(MovePlanItem(
                source=fi.path,
                destination=dest,
                priority=priority,
                confidence=confidence,
                reason=reason
            ))

        # Sort by priority (highest first), then by confidence
        plan.sort(key=lambda x: (x.priority, x.confidence), reverse=True)
        self.plan = plan
        return plan
    
    def _create_duplicate_plan_item(self, file_info: FileInfo, scanner: DirectoryScanner | None) -> MovePlanItem | None:
        """Create a plan item for handling duplicates."""
        if not scanner or not self.duplicates:
            return None
            
        # Find which duplicate group this file belongs to
        file_hash = None
        for hash_val, paths in self.duplicates.items():
            if file_info.path in paths:
                file_hash = hash_val
                break
                
        if not file_hash:
            return None
        
        # Move duplicate to a special duplicates folder
        duplicates_dir = self.root / "Duplicates" / "Duplicate Files"
        dest = duplicates_dir / file_info.path.name
        
        return MovePlanItem(
            source=file_info.path,
            destination=dest,
            priority=2,  # Lower priority for duplicates
            confidence=0.9,
            reason="Duplicate file detected",
            is_duplicate=True
        )
    
    def _should_skip_move(self, file_info: FileInfo) -> bool:
        """Enhanced logic to determine if a file move should be skipped."""
        current_folder = file_info.path.parent
        
        # If file is already in a well-organized subfolder, be conservative
        current_folder_name = current_folder.name.lower()
        
        # Check if current location makes sense
        if file_info.category.lower() in current_folder_name:
            return True
        
        # If suggestion is very similar to current folder, don't move
        if file_info.suggestion and current_folder_name in file_info.suggestion.lower():
            return True
        
        # If current folder has many similar files, prefer to keep together
        current_folder_files = list(current_folder.glob('*'))
        similar_files = [f for f in current_folder_files 
                        if f.is_file() and f.suffix == file_info.path.suffix]
        
        if len(similar_files) >= 5:  # Threshold for "many similar files"
            return True
            
        return False
    
    def _calculate_enhanced_priority(self, file_info: FileInfo, has_existing_folder: bool) -> Tuple[int, float, str]:
        """Enhanced priority calculation with AI reasoning integration."""
        # Start with AI-provided values if available
        ai_priority = file_info.metadata.get('ai_priority', 3)
        ai_confidence = file_info.metadata.get('ai_confidence', 0.5)
        ai_reasoning = file_info.metadata.get('ai_reasoning', "")
        
        priority = ai_priority
        confidence = ai_confidence
        reasons = [ai_reasoning] if ai_reasoning else []
        
        ext = file_info.metadata.get("extension", "").lower()
        file_size = file_info.metadata.get("size", 0)
        
        # Additional priority adjustments beyond AI
        if has_existing_folder:
            confidence += 0.2
            reasons.append("Matching existing folder structure")
        
        # File type context adjustments
        if ext in {'.pdf', '.docx', '.doc'} and any(word in file_info.path.stem.lower() for word in ['important', 'urgent', 'critical']):
            priority = min(5, priority + 1)
            reasons.append("Filename indicates importance")
        
        # Filename quality assessment
        filename_words = len(file_info.path.stem.replace('_', ' ').replace('-', ' ').split())
        if filename_words >= 4:
            confidence += 0.1
            reasons.append("Very descriptive filename")
        elif filename_words == 1:
            confidence -= 0.1
            reasons.append("Generic filename reduces confidence")
        
        # Size-based adjustments for context
        if file_size > 100 * 1024 * 1024:  # > 100MB
            reasons.append("Large file size - consider archive location")
        elif file_size < 1024:  # < 1KB
            priority = max(1, priority - 1)
            confidence -= 0.1
            reasons.append("Very small file may be less important")
        
        # Date-based priority for organization
        try:
            modified_time = file_info.path.stat().st_mtime
            import time
            days_old = (time.time() - modified_time) / (24 * 3600)
            if days_old > 365:  # Over a year old
                priority = max(1, priority - 1)
                reasons.append("Old file - lower organization priority")
            elif days_old < 7:  # Recently modified
                priority = min(5, priority + 1)
                reasons.append("Recently modified - higher priority")
        except OSError:
            pass
        
        # Ensure bounds
        priority = max(1, min(5, priority))
        confidence = max(0.0, min(1.0, confidence))
        
        # Create comprehensive reason
        reason = "; ".join(filter(None, reasons)) if reasons else "Standard organization"
        if len(reason) > 100:  # Truncate if too long
            reason = reason[:97] + "..."
        
        return priority, confidence, reason

    def apply_plan(self, items_to_apply: List[Dict[str, str]]) -> None:
        """Execute a specific list of move items with history tracking."""
        if not items_to_apply:
            return

        successful_moves = []
        failed_moves = []

        for item_dict in tqdm(items_to_apply, desc="Applying plan", unit="file", leave=False, file=sys.stderr):
            source = Path(item_dict['source'])
            destination = Path(item_dict['destination'])
            
            try:
                if source.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Handle filename conflicts
                    final_destination = self._resolve_filename_conflict(destination)
                    
                    # Perform the move
                    shutil.move(str(source), str(final_destination))
                    
                    # Track the move for undo functionality
                    move_record = {
                        "original_source": str(source),
                        "final_destination": str(final_destination),
                        "timestamp": datetime.now().isoformat(),
                        "reason": item_dict.get("reason", "")
                    }
                    successful_moves.append(move_record)
                    
            except Exception as e:
                failed_moves.append({
                    "source": str(source),
                    "destination": str(destination),
                    "error": str(e)
                })
                tqdm.write(f"Error moving {source.name}: {e}", file=sys.stderr)
        
        # Update move history
        self.move_history.extend(successful_moves)
        
        # Report results
        if failed_moves:
            tqdm.write(f"Warning: {len(failed_moves)} file(s) failed to move", file=sys.stderr)
    
    def _resolve_filename_conflict(self, destination: Path) -> Path:
        """Resolve filename conflicts by adding a number suffix."""
        if not destination.exists():
            return destination
        
        base = destination.stem
        suffix = destination.suffix
        parent = destination.parent
        counter = 1
        
        while True:
            new_name = f"{base} ({counter}){suffix}"
            new_destination = parent / new_name
            if not new_destination.exists():
                return new_destination
            counter += 1
    
    def create_undo_plan(self, limit: int = 50) -> List[Dict[str, str]]:
        """Create an undo plan for the most recent moves."""
        if not self.move_history:
            return []
        
        # Get the most recent moves (up to limit)
        recent_moves = self.move_history[-limit:]
        
        undo_plan = []
        for move in reversed(recent_moves):  # Reverse order for undo
            undo_item = {
                "source": move["final_destination"],
                "destination": move["original_source"],
                "reason": f"Undo: {move['reason']}"
            }
            undo_plan.append(undo_item)
        
        return undo_plan

    def export_plan_json(self) -> str:
        return json.dumps([
            {
                "source": str(i.source),
                "destination": str(i.destination),
                "priority": i.priority,
                "confidence": i.confidence,
                "reason": i.reason,
                "is_duplicate": i.is_duplicate
            } for i in self.plan
        ], indent=2) 
    
    def get_plan_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the current plan."""
        if not self.plan:
            return {
                "total_files": 0, 
                "categories": {}, 
                "priorities": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}, 
                "duplicates": 0,
                "avg_confidence": 0.0,
                "high_confidence_count": 0
            }
        
        summary = {
            "total_files": len(self.plan),
            "categories": {},
            "priorities": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            "duplicates": sum(1 for item in self.plan if item.is_duplicate),
            "avg_confidence": sum(item.confidence for item in self.plan) / len(self.plan),
            "high_confidence_count": sum(1 for item in self.plan if item.confidence >= 0.8)
        }
        
        # Count by categories and priorities
        for item in self.plan:
            # Extract category from destination path
            path_parts = str(item.destination).split('/')
            category = path_parts[-3] if len(path_parts) >= 3 else "Other"
            
            summary["categories"][category] = summary["categories"].get(category, 0) + 1
            summary["priorities"][item.priority] += 1
        
        return summary 