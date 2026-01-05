"""
Paper Content Processing Module

Handles downloading and extracting content from papers.
"""

import PyPDF2
import requests
from io import BytesIO
from typing import Dict, List


class PaperContentProcessor:
    """Processes paper content from PDFs."""

    def __init__(self):
        """Initialize processor."""
        self.cache = {}

    def download_and_process_paper(self, pdf_url: str, arxiv_id: str) -> Dict:
        """Download and extract content from paper."""
        try:
            pdf_content = self.download_pdf(pdf_url)
            text = self.extract_text_from_pdf(pdf_content)

            sections = self.parse_paper_structure(text)

            self.cache[arxiv_id] = {
                "text": text,
                "sections": sections,
                "citations": self.extract_citations(text)
            }

            return self.cache[arxiv_id]

        except Exception as e:
            print(f"Error processing paper: {e}")
            return None

    def download_pdf(self, pdf_url: str) -> BytesIO:
        """Download PDF from URL."""
        response = requests.get(pdf_url, timeout=10)
        response.raise_for_status()
        return BytesIO(response.content)

    def extract_text_from_pdf(self, pdf_content: BytesIO) -> str:
        """Extract text from PDF."""
        reader = PyPDF2.PdfReader(pdf_content)
        text = ""

        for page in reader.pages:
            text += page.extract_text()

        return text

    def parse_paper_structure(self, text: str) -> Dict:
        """Parse paper into sections."""
        sections = {
            "abstract": self.extract_section(text, "abstract"),
            "introduction": self.extract_section(text, "introduction"),
            "methodology": self.extract_section(text, "methodology|method|approach"),
            "results": self.extract_section(text, "results|findings"),
            "conclusion": self.extract_section(text, "conclusion|discussion"),
            "references": self.extract_section(text, "references|bibliography")
        }

        return sections

    def extract_section(self, text: str, section_keywords: str) -> str:
        """Extract specific section from paper."""
        import re

        pattern = f"(?i)({section_keywords})\\s*\\n"
        matches = list(re.finditer(pattern, text))

        if not matches:
            return ""

        start_pos = matches[0].end()

        section_pattern = r"(?i)(abstract|introduction|related work|methodology|method|results|conclusion|references|bibliography)\s*\n"
        next_matches = list(re.finditer(section_pattern, text[start_pos:]))

        if next_matches:
            end_pos = start_pos + next_matches[0].start()
        else:
            end_pos = len(text)

        return text[start_pos:end_pos].strip()

    def extract_citations(self, text: str) -> List[Dict]:
        """Extract citations from paper."""
        import re

        citations = []
        patterns = [
            r'\[(\d+)\]\s*(.+?)(?=\[|\Z)',
            r'(\w+\s+et\s+al\.?.*?\(\d{4}\))',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                citations.append({
                    "text": match.group(0)[:200],
                    "position": match.start()
                })

        return citations

    def get_cached_paper(self, arxiv_id: str) -> Dict:
        """Get cached paper content."""
        return self.cache.get(arxiv_id)

    def chunk_paper_for_rag(self, paper_content: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split paper into chunks for RAG."""
        chunks = []
        start = 0

        while start < len(paper_content):
            end = min(start + chunk_size, len(paper_content))
            chunks.append(paper_content[start:end])
            start = end - overlap

        return chunks
