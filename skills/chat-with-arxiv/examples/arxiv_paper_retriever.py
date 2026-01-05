"""
ArXiv Paper Retriever Module

Handles paper discovery and retrieval from ArXiv.
"""

import arxiv
from datetime import datetime, timedelta
from typing import List, Dict


class ArXivPaperRetriever:
    """Retrieves papers from ArXiv."""

    def __init__(self, max_results: int = 50):
        """
        Initialize retriever.

        Args:
            max_results: Maximum results per search
        """
        self.client = arxiv.Client()
        self.max_results = max_results

    def search_papers(self, query: str, sort_by=arxiv.SortCriterion.Relevance) -> List[Dict]:
        """Search ArXiv for papers."""
        search = arxiv.Search(
            query=query,
            sort_by=sort_by,
            max_results=self.max_results
        )

        papers = []
        for result in self.client.results(search):
            papers.append({
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "published": result.published,
                "arxiv_id": result.arxiv_id,
                "pdf_url": result.pdf_url,
                "categories": result.categories,
                "doi": result.doi,
            })

        return papers

    def search_by_category(self, category: str, from_date: datetime, to_date: datetime) -> List[Dict]:
        """Search papers in specific category within date range."""
        date_range_query = f'submittedDate:[{from_date.strftime("%Y%m%d%H%M%S")}Z TO {to_date.strftime("%Y%m%d%H%M%S")}Z]'
        query = f'cat:{category} AND {date_range_query}'
        return self.search_papers(query)

    def search_by_author(self, author_name: str) -> List[Dict]:
        """Search papers by specific author."""
        query = f'au:"{author_name}"'
        return self.search_papers(query)

    def search_by_title(self, title_keywords: List[str]) -> List[Dict]:
        """Search papers by title keywords."""
        title_query = ' AND '.join([f'ti:"{keyword}"' for keyword in title_keywords])
        return self.search_papers(title_query)

    def get_trending_papers(self, category: str, days: int = 7) -> List[Dict]:
        """Get trending papers from recent submissions."""
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        return self.search_by_category(category, from_date, to_date)

    def search_similar_papers(self, arxiv_id: str) -> List[Dict]:
        """Find papers similar to given paper."""
        paper = self.get_paper_by_id(arxiv_id)
        if not paper:
            return []

        key_terms = self.extract_key_terms(paper["summary"])
        query = ' OR '.join(key_terms[:5])
        return self.search_papers(query)

    def get_paper_by_id(self, arxiv_id: str) -> Dict:
        """Get specific paper by ArXiv ID."""
        search = arxiv.Search(id_list=[arxiv_id])

        for result in self.client.results(search):
            return {
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "published": result.published,
                "arxiv_id": result.arxiv_id,
                "pdf_url": result.pdf_url,
                "categories": result.categories,
            }
        return None

    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text."""
        try:
            from nltk.tokenize import sent_tokenize
            from nltk.corpus import stopwords

            words = text.lower().split()
            stop_words = set(stopwords.words('english'))

            key_terms = [
                word.strip('.,;:!?') for word in words
                if len(word) > 5 and word.lower() not in stop_words
            ]

            return list(set(key_terms))[:10]
        except:
            # Fallback if NLTK not available
            return text.split()[:10]
