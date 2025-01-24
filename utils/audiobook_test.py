import urllib.request
import urllib.parse

import sys
import os
sys.path.insert(0, os.path.abspath('/Users/haneol/Documents/Coding/level4-cv-finalproject-hackathon-cv-12-lv3/'))
from config.config import API_CONFIG

import io
import requests
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
    clova_api = ClovaTTSAPI(API_CONFIG["voice_client_id"], API_CONFIG["voice_client_secret"])
    if type == "speaker1":
        speaker = "jinho"
    else:
        speaker = "nara"
    return clova_api.generate_audio(text, speaker=speaker)
    
def script_to_speech(conversations):
    final_audio = AudioSegment.empty()  # Initialize an empty AudioSegment

    for conversation in tqdm(conversations, desc="Generating podcast segments", unit="segment"):
        speaker1_text = conversation["Alex"]
        speaker2_text = conversation["Jamie"]

        audio1_segment = synthesize_text(speaker1_text, type="speaker1")
        # audio1_segment = AudioSegment(audio1, sample_width=2, frame_rate=24000, channels=1)
        audio1_segment = audio1_segment.fade_in(duration=100)
        audio1_segment = audio1_segment.fade_out(duration=100)

        final_audio += audio1_segment
        if len(speaker2_text.strip()) > 0:
            audio2_segment = synthesize_text(speaker2_text, type="speaker2")
            # audio2_segment = AudioSegment(audio2, sample_width=2, frame_rate=24000, channels=1)
            audio2_segment = audio2_segment.fade_in(duration=100)
            audio2_segment = audio2_segment.fade_out(duration=100)

            # Add a pause (e.g., 500 milliseconds = 0.5 seconds)
            pause = AudioSegment.silent(duration=200)
            final_audio += pause  # Add the pause after speaker 1
            final_audio += audio2_segment

    return final_audio

if __name__ == "__main__":
    from pathlib import Path
    file_path = Path('/Users/haneol/Documents/Coding/podcast_thanos.txt')
    text = file_path.read_text()

    """Parses the input text into a list of conversation dictionaries."""
    conversations = []
    current_conversation = {"Alex": "", "Jamie": ""}
    lines = text.strip().split('\n')

    for line in lines:
        if line.startswith("Alex:"):
            current_conversation["Alex"] = line[len("Alex:"):].strip()
        elif line.startswith("Jamie:"):
            current_conversation["Jamie"] = line[len("Jamie:"):].strip()
            conversations.append(current_conversation)
            current_conversation = {"Alex": "", "Jamie": ""}  # Start a new conversation
        # Handle cases where a speaker has multiple consecutive lines
        # elif current_conversation["Alex"] != "" and not line.startswith("Jamie:"):
        #   current_conversation["Alex"] += " " + line.strip() #append new line to the existing one
        # elif current_conversation["Jamie"] != "" and not line.startswith("Alex:"):
        #   current_conversation["Jamie"] += " " + line.strip() #append new line to the existing one


    # Add the last conversation if it's not empty
    if current_conversation["Alex"] or current_conversation["Jamie"]:
        conversations.append(current_conversation)

    final_audio = script_to_speech(conversations)
    final_audio.export("output.mp3", format="mp3")