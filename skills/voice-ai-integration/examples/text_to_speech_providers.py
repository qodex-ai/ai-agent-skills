"""
Text-to-Speech Providers Module

Implementations for various TTS services.
"""

import openai
import requests
import azure.cognitiveservices.speech as speechsdk
from pathlib import Path
from typing import Optional


def text_to_speech_google(text: str, output_file: str = "output.mp3") -> str:
    """Convert text to speech using Google Cloud TTS."""
    from google.cloud import tts_v1

    client = tts_v1.TextToSpeechClient()

    input_text = tts_v1.SynthesisInput(text=text)

    voice = tts_v1.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Neural2-C",
    )

    audio_config = tts_v1.AudioConfig(
        audio_encoding=tts_v1.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=input_text,
        voice=voice,
        audio_config=audio_config
    )

    with open(output_file, "wb") as out:
        out.write(response.audio_content)

    return output_file


def text_to_speech_openai(text: str, output_file: str = "output.mp3") -> str:
    """Convert text to speech using OpenAI TTS."""
    response = openai.Audio.create(
        model="tts-1-hd",
        voice="nova",
        input=text
    )

    response.stream_to_file(output_file)
    return output_file


def text_to_speech_openai_streaming(text: str):
    """Stream text-to-speech from OpenAI."""
    response = openai.Audio.create(
        model="tts-1",
        voice="nova",
        input=text,
        stream=True
    )
    return response


def text_to_speech_azure(text: str, output_file: str, subscription_key: str, region: str) -> Optional[str]:
    """Convert text to speech using Azure Speech Services."""
    speech_config = speechsdk.SpeechConfig(
        subscription=subscription_key,
        region=region
    )

    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file)

    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    result = synthesizer.speak_text(text)

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return output_file
    else:
        raise Exception(f"Speech synthesis failed: {result.reason}")


def text_to_speech_elevenlabs(text: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM",
                              api_key: str = None) -> str:
    """Convert text to speech using Eleven Labs."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }

    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    response = requests.post(url, json=data, headers=headers, stream=True)

    with open("output.mp3", "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)

    return "output.mp3"
