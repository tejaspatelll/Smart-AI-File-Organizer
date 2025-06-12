import hashlib
import os
from pathlib import Path
from typing import Dict, Any

# Conditional imports for document parsing
try:
    import docx
except ImportError:
    docx = None

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

__all__ = [
    "calculate_sha256",
    "read_basic_metadata",
    "extract_text_from_pdf",
    "extract_text_from_docx",
]


def calculate_sha256(filepath: str | Path, block_size: int = 1 << 20) -> str:
    """Return SHA-256 hash of a file (large-file friendly)."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def read_basic_metadata(filepath: str | Path) -> Dict[str, Any]:
    """Return basic os.stat-based metadata for a file."""
    p = Path(filepath)
    stat = p.stat()
    return {
        "name": p.name,
        "extension": p.suffix.lower(),
        "size_bytes": stat.st_size,
        "created_ts": stat.st_ctime,
        "modified_ts": stat.st_mtime,
        "relative_path": str(p),
    }


def extract_text_from_pdf(filepath: str | Path) -> str:
    """Extract all text from a PDF file."""
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    except Exception:
        return ""


def extract_text_from_docx(filepath: str | Path) -> str:
    """Extract all text from a DOCX file."""
    if docx is None:
        return ""
    try:
        doc = docx.Document(filepath)
        return "\n".join(para.text for para in doc.paragraphs if para.text)
    except Exception:
        return "" 