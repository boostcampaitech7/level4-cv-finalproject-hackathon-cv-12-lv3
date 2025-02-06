import requests
import time
import json
import re
import os
import google.generativeai as genai
from mtranslate import translate

def translate_to_english(korean_text):
    """
    한글 키워드를 영어로 번역하는 함수 (API 없이 작동).
    """
    try:
        translated = translate(korean_text, "en", "ko")  # 한국어 -> 영어 번역
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
            
            sorted_papers = sorted(data.get("data", []), key=lambda x: x["citationCount"], reverse=True)[:3]

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

# API 키 설정
api_key = os.getenv("GEMINI_API_KEY")
# ✅ Gemini API 설정
genai.configure(api_key=api_key)

def abstractive_timeline(user_input):
    """
    Gemini API를 사용하여 논문 추천을 수행하는 함수.
    - 입력된 키워드 4개에 대해 각각 3개의 논문을 추천
    - 논문 제목, 저자, 출판 연도, 논문 요약, 난이도, 추천 이유 포함
    - JSON 형식으로 반환
    """
    
    system_prompt="""
            당신은 인공지능 및 머신러닝 논문 추천 시스템입니다.
            사용자로부터 특정 검색 키워드 4개에 대한 논문 목록 3개를 입력받습니다. 
            이 데이터를 참고하여, 검색 키워드 4개에 대해서 키워드별로 읽으면 좋은 논문을 각각 3개 찾아서 출력하세요.
            **반드시 4개의 키워드 각각에 대해 3개의 논문을 추천해야 합니다.**
            **429 오류가 발생하거나 논문이 없을 경우에도 키워드로 관련 논문 검색을 통해 논문을 찾아내야 하며, 그 키워드로 나오는 논문이 없다면, 관련 키워드(임의로)를 사용해 논문을 검색하여 채워야 합니다.** 
            
            # 🎯 **요구사항**
            - **키워드 4개를 모두 처리할 때까지 반복 수행하세요.**
            - 키워드별로 논문을 분류하고, 각 키워드에 대한 핵심 논문을 **3개씩** 추천하세요.
            - 논문 제목, 저자, 출판 연도, 논문 요약을 포함하여 정리하세요.
            - 논문 검색 결과가 부족할 경우, 관련 키워드 논문을 검색하여 추천해주세요.
            - 논문이 중요한 이유 또는 읽어야 하는 이유를 간단히 설명하세요.
            - **논문의 난이도**(초급, 중급, 고급)를 분류하여 추천하세요.
            
            🎯 **429에러나 키워드에 관련된 논문이 없을 때 **
            - 논문 검색 API에서 **429 에러가 발생해서 논문 예시가 없어도 검색을 포기하지 마세요!**
            - **그런 경우에는, 반드시 키워드 자체를 이용해 관련 논문을 찾아야 합니다.**
            - 만약 해당 키워드의 논문이 부족하다면, **유사한 개념의 키워드**를 자동 생성하여 논문을 찾아주세요.
            - 논문 검색 결과가 적더라도 **4개 키워드 × 3개 논문**을 채워야 합니다.
                        
            
            #  **출력 조건
            - 반드시 준수할 것, 하지만 이는 예시일 뿐 출력 형식을 준수할 것
            - 불필요한 빈 키워드(`"키워드1": []`)를 절대 포함하지 마세요.
            - "키워드1", "키워드2" 같은 자리표시(플레이스홀더) 키워드는 출력 예시일 뿐이니 출력에 포함하지 마세요.
            - 출력에는 { } 가장 바깥에 있는 중괄호 안에 있는 내용만 포함해야 하고, ''' 이나 쓸데없는 말은 일절 포함하지 마세요.
            - id 키는 키워드 안 논문 순서대로 1, 2, 3으로 매겨주세요.(그다음 키워드에서는 4,5,6이 아닌 다시 1,2,3으로 매겨주세요.)
            
            
              {"3D Gaussian Splatting": [
#                         {
#                             "id": 1,
                              "논문 제목": "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
#                             "저자": "정보 없음",
#                             "출판 연도": "2023",
#                             "논문 요약": "실시간 레이던스 필드 렌더링을 위한 3D 가우시안 스플래팅 방법을 제안한다.",
#                             "난이도": "고급 (Advanced)",
#                             "추천 이유": "실제 환경에서의 사실적인 빛 시뮬레이션을 가능하게 하는 기술이다."
#                         },
#                         {
    #                         "id": 2,
#                             "논문 제목": "Mip-Splatting: Alias-Free 3D Gaussian Splatting",
#                             "저자": "정보 없음",
#                             "출판 연도": "2023",
#                             "논문 요약": "에일리어스 없는 3D 가우시안 스플래팅 기법을 제안한다.",
#                             "난이도": "초급 (Intermediate)",
#                             "추천 이유": "이미지 품질을 향상시키는 새로운 알고리즘을 제공한다."
#                         },
#                         {
 #                            "id": 3,
#                             "논문 제목": "A Survey on 3D Gaussian Splatting",
#                             "저자": "정보 없음",
#                             "출판 연도": "2024",
#                             "논문 요약": "3D 가우시안 스플래팅 분야의 다양한 접근법을 조사하고 비교한다.",
#                             "난이도": "중급 (Advanced)",
#                             "추천 이유": "해당 주제에 대한 포괄적인 개요를 제공하므로 초보자에게도 유용하다."
#                         }
                ],
                "키워드2": [ ... ],
                "키워드3": [ ... ],
                "키워드4": [ ... ]
            }

            
            # ✅ **출력 검증**
            - **출력하기 전, 초반에 입력으로 받았던 검색 키워드 4개에 대해 각각 3개의 논문정보가 들어있는지 검토하고, 키워드가 부족한 경우에는 반드시 관련 논문을 찾아서라도 논문 3개를 추천해야하고 출력해주세요.또한 출력 형태를 준수하였는지 검토하고 출력해주세요.**

            """
    full_prompt = f"{system_prompt}\n\n사용자 입력 키워드: {user_input}"
    # ✅ Gemini API 호출
    model = genai.GenerativeModel("gemini-pro")  # 🔹 Gemini 모델 지정
    response = model.generate_content(full_prompt)
    gemini_json = json.loads(response.text)  # 문자열을 JSON으로 변환

        # ✅ JSON 파일로 저장
    output_file = "papers_gemini2.json"
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(gemini_json, json_file, ensure_ascii=False, indent=4)

    print(f"📁 JSON 파일 저장 완료: {output_file}")
    print(json.dumps(gemini_json, ensure_ascii=False, indent=4))  # 결과 출력

    return gemini_json