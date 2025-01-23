import html
import logging
import os
import re
import unicodedata
from copy import copy
import requests
from pdf2zh.cache import TranslationCache
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential
from tencentcloud.common import credential
from tencentcloud.tmt.v20180321.tmt_client import TmtClient
from tencentcloud.tmt.v20180321.models import TextTranslateRequest
from tencentcloud.tmt.v20180321.models import TextTranslateResponse
import argostranslate.package
import argostranslate.translate


def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")

class BaseTranslator:
    name = "base"
    envs = {}
    lang_map = {}
    CustomPrompt = False
    ignore_cache = False

    def __init__(self, lang_in, lang_out, model):
        # 언어 매핑 처리
        lang_in = self.lang_map.get(lang_in.lower(), lang_in)
        lang_out = self.lang_map.get(lang_out.lower(), lang_out)
        self.lang_in = lang_in
        self.lang_out = lang_out
        self.model = model

        # TranslationCache 인스턴스 생성
        self.cache = TranslationCache(
            self.name,
            {
                "lang_in": lang_in,
                "lang_out": lang_out,
                "model": model,
            },
        )

    def set_envs(self, envs):
        """
        환경 변수 설정. 
        """
        self.envs = envs or {}

        # 시스템 환경 변수로부터 envs 업데이트
        for key in self.envs:
            if key in os.environ:
                self.envs[key] = os.environ[key]

    def add_cache_impact_parameters(self, k: str, v):
        """
        번역 품질에 영향을 미칠 수 있는 추가 파라미터를 캐시에 포함.
        """
        self.cache.add_params(k, v)

    def translate(self, text, ignore_cache=False):
        """
        텍스트를 번역. 캐시를 활용하여 중복 번역을 방지.
        """
        # 캐시를 확인하여 번역 결과 가져오기
        if not (self.ignore_cache or ignore_cache):
            cache = self.cache.get(text)
            if cache is not None:
                return cache

        # 실제 번역 실행
        translation = self.do_translate(text)

        # 번역 결과를 캐시에 저장
        self.cache.set(text, translation)
        return translation

    def do_translate(self, text):
        """
        실제 번역 작업을 수행. 하위 클래스에서 구현 필요.
        """
        raise NotImplementedError("do_translate 메서드는 하위 클래스에서 구현해야 합니다.")

    def prompt(self, text, prompt):
        """
        사용자 정의 프롬프트를 활용하여 번역 요청 생성.
        """
        if prompt:
            context = {
                "lang_in": self.lang_in,
                "lang_out": self.lang_out,
                "text": text,
            }
            return eval(prompt.safe_substitute(context))
        else:
            return [
                {
                    "role": "system",
                    "content": "You are a professional, authentic machine translation engine.",
                },
                {
                    "role": "user",
                    "content": f"Translate the following markdown source text to {self.lang_out}. Keep the formula notation {{v*}} unchanged. Output translation directly without any additional text.\nSource Text: {text}\nTranslated Text:",  # noqa: E501
                },
            ]

    def __str__(self):
        return f"{self.name} {self.lang_in} {self.lang_out} {self.model}"



class GoogleTranslator(BaseTranslator):
    name = "google"
    lang_map = {"zh": "zh-CN"}

    def __init__(self, lang_in, lang_out, model, **kwargs):
        super().__init__(lang_in, lang_out, model)
        self.session = requests.Session()
        self.endpoint = "http://translate.google.com/m"
        self.headers = {
            "User-Agent": "Mozilla/4.0 (compatible;MSIE 6.0;Windows NT 5.1;SV1;.NET CLR 1.1.4322;.NET CLR 2.0.50727;.NET CLR 3.0.04506.30)"  # noqa: E501
        }

    def do_translate(self, text):
        text = text[:5000]  # google translate max length
        response = self.session.get(
            self.endpoint,
            params={"tl": self.lang_out, "sl": self.lang_in, "q": text},
            headers=self.headers,
        )
        re_result = re.findall(
            r'(?s)class="(?:t0|result-container)">(.*?)<', response.text
        )
        if response.status_code == 400:
            result = "IRREPARABLE TRANSLATION ERROR"
        else:
            response.raise_for_status()
            result = html.unescape(re_result[0])
        return remove_control_characters(result)

class BingTranslator(BaseTranslator):
    # https://github.com/immersive-translate/old-immersive-translate/blob/6df13da22664bea2f51efe5db64c63aca59c4e79/src/background/translationService.js
    name = "bing"
    lang_map = {"zh": "zh-Hans"}

    def __init__(self, lang_in, lang_out, model, **kwargs):
        super().__init__(lang_in, lang_out, model)
        self.session = requests.Session()
        self.endpoint = "https://www.bing.com/translator"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",  # noqa: E501
        }

    def find_sid(self):
        response = self.session.get(self.endpoint)
        response.raise_for_status()
        url = response.url[:-10]
        ig = re.findall(r"\"ig\":\"(.*?)\"", response.text)[0]
        iid = re.findall(r"data-iid=\"(.*?)\"", response.text)[-1]
        key, token = re.findall(
            r"params_AbusePreventionHelper\s=\s\[(.*?),\"(.*?)\",", response.text
        )[0]
        return url, ig, iid, key, token

    def do_translate(self, text):
        text = text[:1000]  # bing translate max length
        url, ig, iid, key, token = self.find_sid()
        response = self.session.post(
            f"{url}ttranslatev3?IG={ig}&IID={iid}",
            data={
                "fromLang": self.lang_in,
                "to": self.lang_out,
                "text": text,
                "token": token,
                "key": key,
            },
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()[0]["translations"][0]["text"]