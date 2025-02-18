import re
import json

# 프롬프트 템플릿
QUERY_ENHANCEMENT_PROMPT = """
역할: 당신은 사용자의 질문을 더 명확하고 구체적으로 만들어주는 도우미입니다. 사용자의 질문을 분석하여 의도를 파악하고, 더 명확하고 구체적인 형태로 변환합니다.

입력: {user_query}

지침:
1. 사용자의 질문을 분석하여 의도를 파악합니다.
2. 질문이 불완전하거나 모호한 경우, 이를 더 명확하고 구체적인 형태로 재구성합니다.
3. 질문의 핵심 의도를 유지하면서, 필요한 세부 사항을 추가합니다.
4. 질문이 이미 명확하고 구체적이라면, 원래 질문을 그대로 반환합니다.

출력 형식:
- 최적화된 검색문: [명확하고 구체적인 질문]

예시 1:
입력: "encoder-decoder structure에 대해 간단하게 알려줘"
최적화된 검색문: "Encoder-Decoder 구조의 기본 개념과 작동 원리를 간단히 설명해주세요."

예시 2:
입력: "딥러닝으로 주가 예측하는 방법 알려줘"
최적화된 검색문: "딥러닝을 사용하여 주가를 예측하는 방법과 주요 알고리즘에 대해 설명해주세요."

예시 3:
입력: "트랜스포머가 RNN보다 좋은 점"
최적화된 검색문: "트랜스포머와 RNN을 비교했을 때, 트랜스포머의 장점과 단점은 무엇인가요?"

예시 4:
입력: "똥이 마려운 이유?"
최적화된 검색문: "똥이 마려운 이유는 무엇인가요? 생리학적 원리를 설명해주세요."

예시 5:
입력: "날씨가 어떠니?"
최적화된 검색문: "현재 날씨는 어떤가요?"

예시 6:
입력: "너는 누구니?"
최적화된 검색문: "당신은 누구인가요?"
"""


def extract_enhanced_query(llm_response):
    """
    LLM 응답에서 최적화된 검색문을 추출하는 함수
    :param llm_response: LLM의 응답 (JSON 문자열 또는 파이썬 딕셔너리)
    :return: 최적화된 검색문 (str) 또는 None
    """
    if not llm_response:
        return None

    try:
        if isinstance(llm_response, str):
            response_json = json.loads(llm_response)
        elif isinstance(llm_response, dict):
            response_json = llm_response
        else:
            raise ValueError("llm_response는 JSON 문자열 또는 딕셔너리여야 합니다.")
        content = response_json["message"]["content"]
        enhanced_query_match = re.search(r'최적화된 검색문\s*:\s*"(.*?)"', content)
        if enhanced_query_match:
            return enhanced_query_match.group(1).strip()
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"오류 발생: {e}")
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

    llm_response = completion_executor.execute(request_data, stream=False)
    enhanced_query = extract_enhanced_query(llm_response)

    return enhanced_query
