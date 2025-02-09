import requests
import time
import json
import re
import os
from mtranslate import translate
from api.api_classes import ChatCompletionsExecutor
from config.config import API_CONFIG
import json


def translate_to_english(korean_text):
    """
    한글 키워드를 영어로 번역하는 함수 (API 없이 작동).
    """
    try:
        translated = translate(korean_text, "en", "ko")  # 한국어 -> 영어 번역
        if not translated.strip():  # 빈 문자열 방지
            raise ValueError("번역 결과가 빈 문자열입니다.")
        return translated
    except Exception as e:
        print(f"⚠️ 번역 실패: {korean_text} → 원래 단어 유지 ({e})")
        return korean_text  # 번역 실패 시 원래 단어 유지


def extract_keywords(text):
    """
    주어진 텍스트에서 '#'이 앞에 붙은 키워드들을 띄어쓰기까지 포함하여 추출하는 함수.
    쉼표가 나오기 전까지를 하나의 키워드로 인식함.
    """
    hashtags = re.findall(r"#([^\n#,]+)", text)  # '#' 이후 쉼표 또는 줄바꿈 전까지 추출
    keywords = [tag.strip() for tag in hashtags][:4]
    print(f"추출된 원본 키워드: {keywords}")
    # 한글 키워드 변환
    translated_keywords = [
        translate_to_english(tag) if re.search(r"[가-힣]", tag) else tag
        for tag in keywords
    ]
    return translated_keywords


def timeline_str(query_list):
    # API 엔드포인트
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    output_str = ""

    # 각 키워드별 논문 검색
    for query in query_list:
        output_str += f"\n검색 키워드: {query}\n"
        url = f"{base_url}?query={query}&fields=title,year,citationCount&limit=10"

        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()

            sorted_papers = sorted(
                data.get("data", []), key=lambda x: x["citationCount"], reverse=True)[:3]

            for paper in sorted_papers:
                short_title = paper["title"]
                output_str += f"- {short_title}\n"
                output_str += f"  - 저자: {paper['authors'] if 'authors' in paper else '정보 없음'}\n"
                output_str += f"  - 출판 연도: {paper['year']}\n"
                output_str += f"  - 인용 수: {paper['citationCount']}\n"
        else:
            output_str += f"API 요청 실패: {response.status_code}\n"

        # 속도 제한을 고려하여 1초 대기
        time.sleep(1)

    return output_str




def abstractive_timeline(user_input):
    """
    - 입력된 키워드 4개에 대해 각각 3개의 논문을 추천
    - 논문 제목, 저자, 출판 연도, 논문 요약, 난이도, 추천 이유 포함
    - JSON 형식으로 반환
    """
    chat_api = ChatCompletionsExecutor(
        host=API_CONFIG["host"],
        api_key=API_CONFIG["api_key"],
        request_id=API_CONFIG["request_id"],
    )

    system_prompt = """
            당신은 인공지능 및 머신러닝 논문 추천이자 시스템입니다.
            반드시 아래에 있는 준수하여 출력해주세요.
            그외의 출력은 잘못된 출력으로 처리합니다.
            사용자로부터 특정 검색 키워드 4개를 입력받습니다. 
            이 데이터를 참고하여, 검색 키워드 4개에 대해서 키워드별로 읽으면 좋은 논문을 각각 3개 찾아서 출력하세요.
            단, 너무 오래된 논문이나 너무 최신의 논문은 추천하지 않습니다.
            **반드시 4개의 키워드 각각에 대해 3개의 논문을 추천해야 합니다.**
            
            
            # 🎯 **요구사항**
            - **키워드 4개를 모두 처리할 때까지 반복 수행하세요.**
            - 키워드별로 논문을 분류하고, 각 키워드에 대한 핵심 논문을 **3개씩** 추천하세요.
            - 논문 제목, 저자, 출판 연도, 논문 요약을 포함하여 정리하세요.
            - 논문 검색 결과가 부족할 경우, 관련 키워드 논문을 검색하여 추천해주세요.
            - 논문이 중요한 이유 또는 읽어야 하는 이유를 간단히 설명하세요.
            - **논문의 난이도**(초급, 중급, 고급)를 분류하여 추천하세요.
            
            #  **출력 형식 (JSON 규칙 준수)**
            - 반드시 준수할 것, 하지만 이는 예시일 뿐 출력 형식을 준수할 것
            - 키워드 4개가 들어가야 하며 각각 3개의 논문을 반드시 추천해야 합니다.
            - **올바른 JSON을 반환해야 합니다** (콤마 누락, 중괄호 닫힘 오류 등 금지). 
            - JSON 데이터가 깨지지 않도록 **잘못된 쉼표(,) 삽입 방지** 및 **올바른 배열 종료(`]`) 유지**. 
            - JSON 이외의 불필요한 텍스트나 설명, 기호는 절대 포함하지 마세요.
            - 불필요한 빈 키워드("키워드1": )를 절대 포함하지 마세요.
            - "키워드1", "키워드2" 같은 자리표시(플레이스홀더) 키워드는 출력 예시일 뿐이니 출력에 포함하지 마세요.
            - id 키는 키워드 안 논문 순서대로 1, 2, 3으로 매겨주세요.(그다음 키워드에서는 4,5,6이 아닌 다시 1,2,3으로 매겨주세요.)
            
            

                  {
                    "Social Capital": [
                        {
                            "id": 1,
                            "논문 제목": "Social Capital and Social Networks: A Conceptual and Empirical Overview",
                            "저자": "Bourdieu, P.",
                            "출판 연도": 2003,
                            "논문 요약": "사회 자본과 사회적 네트워크의 개념적, 경험적 개요를 제공합니다.",
                            "난이도": "중급",
                            "추천 이유": "사회 자본 연구의 기초적 개요를 제공하는 영향력 있는 논문입니다."
                        },
                        {
                            "id": 2,
                            "논문 제목": "The Strength of Weak Ties: A Network Theory Revisited",
                            "저자": "Granovetter, M. S.",
                            "출판 연도": 1973,
                            "논문 요약": "약한 유대의 중요성과 사회 자본에서의 역할에 대해 논의합니다.",
                            "난이도": "초급",
                            "추천 이유": "네트워크 이론에서 널리 인용되는 논문으로 약한 유대의 개념을 설명합니다."
                        },
                        {
                            "id": 3,
                            "논문 제목": "Social Capital: Its Origins and Applications in Modern Sociology",
                            "저자": "Coleman, J. S.",
                            "출판 연도": 1988,
                            "논문 요약": "사회 자본의 기원과 현대 사회학에서의 적용에 대해 설명합니다.",
                            "난이도": "중급",
                            "추천 이유": "사회 자본 개념을 처음 정의하고 이를 사회 현상에 적용한 획기적인 논문입니다."
                        }
                    ],
                      
                    "키워드2":  [] ,
                    "키워드3":  [] ,
                    "키워드4":  []
                }
            

            
            # ✅ **출력 검증**
            - **출력하기 전, 초반에 입력으로 받았던 검색 키워드 4개에 대해 각각 3개의 논문정보가 들어있는지 검토하고, 빠뜨린 부분이 있다면 꼭 형식에 맞게 채워주고 출력해주세요.
            - **또한 출력 형태를 준수하였는지 검토하고 출력해야하며 이외의 불필요한 말이 들어가 있는지 검토하고 출력해주세요.**

            """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content":  ", ".join(user_input)},
    ]
    request_data = {
        "messages": messages,
        "topP": 0.8,
        "topK": 0,
        "maxTokens": 4096,
        "temperature": 0.6,
        "repeatPenalty": 5.0,
        "stopBefore": [],
        "includeAiFilters": True,
        "seed": 1234,
        "responseFormat": "json"  # 🚀 JSON 강제 출력 설정
    }
    response = chat_api.execute(request_data, stream=False)
    # full_prompt = f"{system_prompt}\n\n사용자 입력 키워드: {user_input}"
    # # ✅ Gemini API 호출
    # model = genai.GenerativeModel(model_name="gemini-pro",
    # generation_config={"response_mime_type": "application/json"})  # JSON 모드 강제)  # 🔹 Gemini 모델 지정
    # response = model.generate_content(full_prompt)
    
    # JSON 문자열 추출
    response_text = response["message"]["content"]
    
    print(response_text)

    try:
        # JSON 부분만 추출
        json_string = extract_json(response_text)

        # JSON 변환
        parsed_json = json.loads(json_string)

        # 변환된 JSON 저장
        with open("clova_output.json", "w", encoding="utf-8") as f:
            json.dump(parsed_json, f, ensure_ascii=False, indent=4)

        print("✅ JSON 파일 저장 완료: clova_output.json")

    except ValueError as e:
        print(f"❌ JSON 추출 오류 발생: {e}")

    except json.JSONDecodeError as e:
        print(f"❌ JSON 변환 오류 발생: {e}")

    return parsed_json

def extract_json(response_text):
    """
    Clova 응답에서 JSON 데이터만 추출하는 함수.
    - JSON이 깨질 가능성을 방지하고, 정규식으로 JSON 본문만 추출.
    """
    json_match = re.search(r'({.*})', response_text, re.DOTALL)
    if json_match:
        return json_match.group(1)  # JSON 문자열만 반환
    else:
        raise ValueError("JSON 응답을 감지할 수 없음")