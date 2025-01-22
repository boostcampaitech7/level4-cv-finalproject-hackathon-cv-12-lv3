import re
import json

# 프롬프트 템플릿
QUERY_ENHANCEMENT_PROMPT = """
역할: 당신은 학술 논문 검색 최적화 전문가입니다. 사용자의 질문을 학술적 맥락에 맞는 자연어 형태로 변환하여, 논문 검색 시 더 정확하고 풍부한 결과를 얻을 수 있도록 돕습니다.

입력: {user_query}

지침:
1. 사용자의 질문을 분석하여 연구 목적을 파악합니다.
2. 질문을 학술적 맥락에 맞게 재구성합니다. 이때, 관련된 학술 용어, 키워드, 또는 개념을 포함시킵니다.
3. 최종적으로 완성된 자연어 문장은 논문 검색 엔진에서 사용하기 적합해야 합니다.

출력 형식:
- 최적화된 검색문: [자연어 형태의 학술적 질의]

예시 1:
입력: "딥러닝으로 주가 예측하는 방법 알려줘"
최적화된 검색문: "딥러닝과 시계열 분석을 활용한 주식 시장 예측 모델의 연구 방법론과 성능 평가에 대한 최신 연구"

예시 2:
입력: "트랜스포머가 RNN보다 좋은 점"
최적화된 검색문: "트랜스포머와 RNN 아키텍처의 비교 분석: 장단점 및 자연어 처리 작업에서의 성능 차이"

예시 3:
입력: "똥이 마려운 이유?"
최적화된 검색문: "직장 내압 증가와 관련된 신경 전달 물질 및 수용체의 역할에 대한 생리학적 연구"
"""

def extract_enhanced_query(llm_response):
    """
    LLM 응답에서 최적화된 검색문을 추출하는 함수
    :param llm_response: LLM의 응답 (JSON 형식)
    :return: 최적화된 검색문 (str) 또는 None
    """
    if not llm_response:
        return None

    try:
        response_json = json.loads(llm_response)
        content = response_json["result"]["message"]["content"]
        enhanced_query_match = re.search(r'최적화된 검색문\s*:\s*"(.*?)"', content)
        if enhanced_query_match:
            return enhanced_query_match.group(1).strip()
    except (json.JSONDecodeError, KeyError) as e:
        print(f"JSON 파싱 중 에러 발생: {e}")
    return None

def llm_refine(query, completion_executor):
    """
    사용자 쿼리를 최적화된 검색문으로 변환하는 함수
    :param query: 사용자 쿼리 (str)
    :param completion_executor: ChatCompletionAPI 객체
    :return: 최적화된 검색문 (str)
    """
    prompt = QUERY_ENHANCEMENT_PROMPT.format(user_query=query)
    request_data = {
        'messages': [{"role": "user", "content": prompt}],
        'topP': 0.8,
        'topK': 0,
        'maxTokens': 4096,
        'temperature': 0.5,
        'repeatPenalty': 5.0,
        'stopBefore': [],
        'includeAiFilters': True,
        'seed': 0
    }

    llm_response = completion_executor.execute(request_data)
    enhanced_query = extract_enhanced_query(llm_response)

    return enhanced_query