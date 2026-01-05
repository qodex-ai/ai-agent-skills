"""
Voice Assistant Module

Complete voice processing pipeline.
"""

import asyncio
from typing import Optional


class VoiceAssistant:
    """Complete voice assistant implementation."""

    def __init__(self, stt_provider: str = "openai", tts_provider: str = "openai"):
        """
        Initialize voice assistant.

        Args:
            stt_provider: Speech-to-text provider
            tts_provider: Text-to-speech provider
        """
        self.stt_provider = stt_provider
        self.tts_provider = tts_provider
        self.conversation_history = []
        self.llm = None  # Will be initialized with LLM

    async def process_voice_input(self, audio_file: str) -> str:
        """Convert voice to text."""
        from .speech_recognition_providers import (
            transcribe_audio_whisper, transcribe_audio_google
        )

        if self.stt_provider == "openai":
            text = transcribe_audio_whisper(audio_file)
        elif self.stt_provider == "google":
            text = transcribe_audio_google(audio_file)
        else:
            raise ValueError(f"Unknown STT provider: {self.stt_provider}")

        return text

    async def generate_response(self, user_input: str) -> str:
        """Generate AI response."""
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })

        if self.llm:
            response = self.llm(self.conversation_history)
        else:
            response = f"You said: {user_input}"

        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return response

    async def synthesize_response(self, text: str) -> str:
        """Convert text to voice."""
        from .text_to_speech_providers import (
            text_to_speech_openai, text_to_speech_google
        )

        if self.tts_provider == "openai":
            audio_file = text_to_speech_openai(text)
        elif self.tts_provider == "google":
            audio_file = text_to_speech_google(text)
        else:
            raise ValueError(f"Unknown TTS provider: {self.tts_provider}")

        return audio_file

    async def chat(self, audio_input: str) -> str:
        """Complete voice chat pipeline."""
        # Speech to text
        user_text = await self.process_voice_input(audio_input)
        print(f"User said: {user_text}")

        # Generate response
        response_text = await self.generate_response(user_text)

        # Text to speech
        audio_output = await self.synthesize_response(response_text)

        return audio_output

    def get_conversation_history(self):
        """Get conversation history."""
        return self.conversation_history

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
