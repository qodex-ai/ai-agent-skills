"""
Document Processors Module

Handles extraction from various document types.
"""

from pypdf import PdfReader
import requests
from bs4 import BeautifulSoup
import re
from typing import Dict, List


def extract_pdf_content(file_path: str) -> Dict:
    """Extract text from PDF."""
    reader = PdfReader(file_path)
    text = ""
    metadata = {}

    for page_num, page in enumerate(reader.pages):
        text += page.extract_text()

    if reader.metadata:
        metadata = {
            "title": reader.metadata.get("/Title", ""),
            "author": reader.metadata.get("/Author", ""),
            "pages": len(reader.pages)
        }

    return {
        "text": text,
        "metadata": metadata,
        "source": file_path
    }


def extract_github_content(repo_url: str) -> List[Dict]:
    """Extract code files from GitHub repository."""
    import base64

    # Parse repo URL and build API URL
    api_url = repo_url.replace("github.com", "api.github.com/repos")
    response = requests.get(f"{api_url}/contents")

    files = []
    for item in response.json():
        if item["type"] == "file":
            content_response = requests.get(item["url"])
            content = base64.b64decode(
                content_response.json()["content"]
            ).decode()
            files.append({
                "path": item["path"],
                "content": content
            })

    return files


def extract_email_content(mailbox_path: str) -> List[Dict]:
    """Extract emails from mailbox."""
    import email
    import os

    emails = []
    for filename in os.listdir(mailbox_path):
        with open(os.path.join(mailbox_path, filename), 'rb') as f:
            msg = email.message_from_binary_file(f)
            emails.append({
                "from": msg["From"],
                "to": msg["To"],
                "subject": msg["Subject"],
                "date": msg["Date"],
                "body": msg.get_payload()
            })

    return emails


def extract_web_content(url: str) -> Dict:
    """Extract content from web page."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    text = soup.get_text()
    headings = [h.get_text() for h in soup.find_all(['h1', 'h2', 'h3'])]
    links = [(a.get_text(), a.get('href')) for a in soup.find_all('a')]

    return {
        "text": text,
        "headings": headings,
        "links": links,
        "url": url
    }


def extract_youtube_content(video_id: str) -> str:
    """Extract transcript from YouTube video."""
    from youtube_transcript_api import YouTubeTranscriptApi

    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join([item["text"] for item in transcript])
    return text


def extract_audio_content(audio_file: str) -> str:
    """Transcribe audio file."""
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)

    text = recognizer.recognize_google(audio)
    return text
