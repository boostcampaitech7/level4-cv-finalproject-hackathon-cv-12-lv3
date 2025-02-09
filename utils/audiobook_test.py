from google.cloud import texttospeech
import base64
from http import HTTPStatus  # For better HTTP status code handling
from pydub import AudioSegment  # Import pydub for audio manipulation
from tqdm import tqdm
from utils.script import write_full_script
import requests
import io
from config.config import VOICE_CONFIG
import urllib.request
import urllib.parse

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, ".."))


# Google Cloud TTS API 키 가져오기
GCP_API_KEY = os.getenv("GCP_API_KEY")

# 서비스 계정 키 환경 변수 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/data/ephemeral/home/google.json"


def synthesize_text(text, voice_type="speaker1"):
    """Google Cloud TTS를 사용하여 텍스트를 음성으로 변환하는 함수"""
    client = texttospeech.TextToSpeechClient()

    if voice_type == "speaker1":
        voice_name = "en-US-Journey-O"
    else:
        voice_name = "en-US-Journey-D"

    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name=voice_name
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    return response.audio_content


def script_to_speech(conversations):
    final_audio = AudioSegment.empty()

    for conversation in tqdm(conversations, desc="Generating podcast segments", unit="segment"):
        jaeseok_text = conversation.get("재석", "")
        haha_text = conversation.get("하하", "")

        if jaeseok_text:
            audio_content = synthesize_text(
                jaeseok_text, voice_type="speaker1")
            jaeseok_audio = AudioSegment.from_file(
                io.BytesIO(audio_content), format="mp3")
            final_audio += jaeseok_audio.fade_in(
                duration=100).fade_out(duration=100)

        if haha_text:
            pause = AudioSegment.silent(duration=200)
            final_audio += pause
            audio_content = synthesize_text(haha_text, voice_type="speaker2")
            haha_audio = AudioSegment.from_file(
                io.BytesIO(audio_content), format="mp3")
            final_audio += haha_audio.fade_in(
                duration=100).fade_out(duration=100)

    return final_audio


if __name__ == "__main__":
    try:
        text_file_path = "/data/ephemeral/home/YJ/level4-cv-finalproject-hackathon-cv-12-lv3/utils/summary.txt"
        with open(text_file_path, "r", encoding="utf-8") as file:
            text_content = file.read()

        conversations = write_full_script(text_content)
        final_audio = script_to_speech(conversations)
        final_audio.export("output.mp3", format="mp3")
        print("Podcast audio generated successfully as 'output.mp3'.")
    except Exception as e:
        print(f"Error occurred: {e}")
