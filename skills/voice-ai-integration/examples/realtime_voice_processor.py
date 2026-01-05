"""
Real-Time Voice Processing Module

Handles real-time audio streaming.
"""

import pyaudio
import numpy as np


class RealTimeVoiceProcessor:
    """Processes real-time voice input/output."""

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 2048):
        """
        Initialize real-time processor.

        Args:
            sample_rate: Audio sample rate in Hz
            chunk_size: Chunk size for processing
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio = pyaudio.PyAudio()

    def stream_audio_input(self):
        """Stream audio from microphone."""
        stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        try:
            while True:
                data = stream.read(self.chunk_size)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                yield audio_chunk
        finally:
            stream.stop_stream()
            stream.close()

    def stream_audio_output(self, audio_stream):
        """Stream audio to speakers."""
        stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.chunk_size
        )

        try:
            for audio_chunk in audio_stream:
                stream.write(audio_chunk.tobytes())
        finally:
            stream.stop_stream()
            stream.close()

    def detect_speech_activity(self, audio_chunk: np.ndarray, threshold: float = 0.02) -> bool:
        """Detect if audio chunk contains speech."""
        # Simple energy-based VAD
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        return energy > threshold

    def __del__(self):
        """Cleanup audio resources."""
        self.audio.terminate()
