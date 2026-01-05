"""
Speech Recognition Providers Module

Implementations for various speech-to-text services.
"""

import io
import openai
import azure.cognitiveservices.speech as speechsdk
import requests
from typing import Optional


def transcribe_audio_google(audio_file: str) -> str:
    """Transcribe using Google Cloud Speech-to-Text."""
    from google.cloud import speech_v1

    client = speech_v1.SpeechClient()

    with io.open(audio_file, "rb") as audio:
        content = audio.read()

    audio = speech_v1.RecognitionAudio(content=content)
    config = speech_v1.RecognitionConfig(
        encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True,
    )

    response = client.recognize(config=config, audio=audio)

    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript

    return transcript


def transcribe_audio_whisper(audio_file: str) -> str:
    """Transcribe using OpenAI Whisper."""
    with open(audio_file, "rb") as f:
        transcript = openai.Audio.transcribe(
            model="whisper-1",
            file=f,
            language="en"
        )
    return transcript["text"]


def transcribe_audio_azure(audio_file: str, subscription_key: str, region: str) -> Optional[str]:
    """Transcribe using Azure Speech Services."""
    speech_config = speechsdk.SpeechConfig(
        subscription=subscription_key,
        region=region
    )
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file)

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    result = recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        return None
    elif result.reason == speechsdk.ResultReason.Canceled:
        raise Exception(result.cancellation_details.error_details)


def transcribe_audio_assemblyai(audio_file: str, api_key: str) -> str:
    """Transcribe using AssemblyAI."""
    # Upload audio
    with open(audio_file, "rb") as f:
        response = requests.post(
            "https://api.assemblyai.com/v2/upload",
            headers={"authorization": api_key},
            data=f
        )
    audio_url = response.json()["upload_url"]

    # Transcribe
    transcript_response = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        headers={"authorization": api_key},
        json={"audio_url": audio_url}
    )

    transcript_id = transcript_response.json()["id"]

    # Poll for result
    import time
    while True:
        poll_response = requests.get(
            f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
            headers={"authorization": api_key}
        )
        if poll_response.json()["status"] == "completed":
            return poll_response.json()["text"]
        time.sleep(1)
