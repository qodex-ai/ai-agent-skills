"""
Text Processing and Chunking Module

Handles document preprocessing and intelligent chunking.
"""

import re
from typing import List, Dict


def preprocess_document(text: str) -> str:
    """Clean and preprocess document text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\!\?\,-]', '', text)

    return text


def chunk_text_recursive(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into chunks with overlap."""
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        # Try to split on sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size // 2:
                end = start + last_period + 1
                chunk = text[start:end]

        chunks.append(chunk)
        start = end - overlap

    return chunks


def extract_metadata(document: Dict) -> Dict:
    """Extract metadata from document."""
    return {
        "title": document.get("metadata", {}).get("title", "Unknown"),
        "author": document.get("metadata", {}).get("author", "Unknown"),
        "date": document.get("metadata", {}).get("date", "Unknown"),
        "language": "en",
        "word_count": len(document.get("text", "").split()),
        "source": document.get("source", "unknown")
    }


def preserve_document_structure(text: str) -> List[Dict]:
    """Preserve heading hierarchy in chunks."""
    chunks = []
    current_section = ""
    current_context = ""

    for line in text.split('\n'):
        # Check if line is a heading (simple heuristic)
        if line.startswith(('#', '##', '###')) or line.isupper():
            if current_section:
                chunks.append({
                    "text": current_section,
                    "context": current_context,
                    "heading": current_context
                })
            current_context = line
            current_section = ""
        else:
            current_section += line + "\n"

    return chunks
