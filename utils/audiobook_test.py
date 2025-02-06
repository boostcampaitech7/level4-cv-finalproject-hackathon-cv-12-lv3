import urllib.request
import urllib.parse

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, ".."))
from config.config import VOICE_CONFIG

import io
import requests
from utils.script import write_full_script
from tqdm import tqdm
from pydub import AudioSegment  # Import pydub for audio manipulation
from http import HTTPStatus  # For better HTTP status code handling


class ClovaTTSAPI:
    def __init__(self, client_id, client_secret, default_speaker="nara", default_format="mp3"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.default_speaker = default_speaker
        self.default_format = default_format
        self.base_url = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"


    def generate_audio(self, text, filename="output.mp3", speaker=None, speed=0, volume=0, pitch=0, format="mp3"):
        if speaker is None:
            speaker = self.default_speaker
        if format is None:
            format = self.default_format
        
        # Check text length (Clova TTS limit)
        if len(text) > 1024:
            raise ValueError("Text length exceeds the 1024 character limit for Clova TTS.")

        payload = urllib.parse.urlencode({
            "speaker": speaker,
            "volume": volume,
            "speed": speed,
            "pitch": pitch,
            "text": text,
            "format": format
        })

        headers = {
            "X-NCP-APIGW-API-KEY-ID": self.client_id,
            "X-NCP-APIGW-API-KEY": self.client_secret,
            "Content-Type": "application/x-www-form-urlencoded" # Important for POST requests
        }

        req = urllib.request.Request(self.base_url, data=payload.encode('utf-8'), headers=headers) # Encode payload

        try:
            with urllib.request.urlopen(req) as response:
                rescode = response.getcode()
                print(f"Clova TTS API Response Code: {rescode}")
                if rescode == HTTPStatus.OK:  # Use HTTPStatus for clarity
                    audio_data = response.read()
                    audio_stream = io.BytesIO(audio_data)
                    audio_segment = AudioSegment.from_file(audio_stream, format=format)
                    audio_stream.close()
                    return audio_segment
                else:
                    print(f"Clova TTS API Error: {rescode}")
                    return False  # Indicate failure

        except urllib.error.URLError as e:
            print(f"Network Error: {e}")
            return False
        except Exception as e:  # Catch other potential exceptions
            print(f"An unexpected error occurred: {e}")
            return False



def synthesize_text(text, type="speaker1"):
    """Synthesizes speech from the input string of text using Naver Clova API."""
    clova_api = ClovaTTSAPI(VOICE_CONFIG["voice_client_id"], VOICE_CONFIG["voice_client_secret"])
    if type == "speaker1":
        speaker = "jinho"
    else:
        speaker = "nara"
    return clova_api.generate_audio(text, speaker=speaker)
    
def script_to_speech(conversations):
    final_audio = AudioSegment.empty()  # Initialize an empty AudioSegment

    for conversation in tqdm(conversations, desc="Generating podcast segments", unit="segment"):
        jaeseok_text = conversation.get("재석", "")
        haha_text = conversation.get("하하", "")

        # 재석의 대화 음성 생성
        if jaeseok_text:
            jaeseok_audio = synthesize_text(jaeseok_text, type="speaker1")
            if not jaeseok_audio:
                print(f"Error generating audio for 재석: {jaeseok_text}")
                continue  # Skip to the next conversation
            jaeseok_audio = jaeseok_audio.fade_in(duration=100).fade_out(duration=100)
            final_audio += jaeseok_audio

        # 하하의 대화 음성 생성
        if haha_text:
            pause = AudioSegment.silent(duration=200)
            final_audio += pause  # 대화 사이에 잠시 멈춤 추가
            haha_audio = synthesize_text(haha_text, type="speaker2")
            if not haha_audio:
                print(f"Error generating audio for 하하: {haha_text}")
                continue  # Skip to the next conversation
            haha_audio = haha_audio.fade_in(duration=100).fade_out(duration=100)
            final_audio += haha_audio
    return final_audio

if __name__ == "__main__":
    try:
        text_file_path = "/data/ephemeral/home/jm/level4-cv-finalproject-hackathon-cv-12-lv3/summary.txt"
        with open(text_file_path, "r", encoding="utf-8") as file:
            text_content = file.read()

        # 스크립트 생성
        conversations = write_full_script(text_content)

        # TTS를 이용해 음성 변환
        final_audio = script_to_speech(conversations)
        final_audio.export("output.mp3", format="mp3")
        print("Podcast audio generated successfully as 'output.mp3'.")

    except Exception as e:
        print(f"Error occurred: {e}")