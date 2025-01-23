from .baseAPI import BaseAPIExecutor
from config.config import API_CONFIG
import requests
import json
from http import HTTPStatus

class EmbeddingAPI(BaseAPIExecutor):
    def __init__(self, host, api_key, request_id):
        super().__init__(host, api_key, request_id)
        self._endpoint = API_CONFIG['embedding_endpoint']

    def get_embedding(self, text):
        payload = {"text": text}
        result = self.execute(self._endpoint, payload)
        return result.get("embedding", "Error")


class SegmentationAPI(BaseAPIExecutor):
    def __init__(self, host, api_key, request_id):
        super().__init__(host, api_key, request_id)
        self._endpoint = API_CONFIG['segmentation_endpoint']

    def get_segmentation(self, text, alpha=-100, seg_cnt=-1):
        payload = {
            "text": text,
            "alpha": alpha,
            "segCnt": seg_cnt
        }
        result = self.execute(self._endpoint, payload)
        return result.get("topicSeg", "Error")
    
# class ChatCompletionAPI(BaseAPIExecutor):
#     def __init__(self, host, api_key, request_id):
#         super().__init__(host, api_key, request_id)
#         self._endpoint = API_CONFIG['chat_completion_endpoint']

#     def get_completion(self, messages, top_p=0.8, top_k=0, max_tokens=4096,
#                        temperature=0.5, repeat_penalty=5.0, stop_before=None,
#                        include_ai_filters=True, seed=0):
#         """
#         챗봇 응답을 생성하는 메서드 (비스트리밍 방식)
#         :param messages: 대화 메시지 목록
#         :param top_p: 상위 확률 샘플링 임계값
#         :param top_k: 상위 k개 토큰 샘플링
#         :param max_tokens: 최대 생성 토큰 수
#         :param temperature: 생성 다양성 조절
#         :param repeat_penalty: 반복 패널티
#         :param stop_before: 생성 중단 토큰 목록
#         :param include_ai_filters: AI 필터 포함 여부
#         :param seed: 랜덤 시드
#         :return: API 응답 결과
#         """
#         if stop_before is None:
#             stop_before = []

#         # 요청 데이터 구성
#         payload = {
#             'messages': messages,
#             'topP': top_p,
#             'topK': top_k,
#             'maxTokens': max_tokens,
#             'temperature': temperature,
#             'repeatPenalty': repeat_penalty,
#             'stopBefore': stop_before,
#             'includeAiFilters': include_ai_filters,
#             'seed': seed
#         }

#         # 요청 헤더 구성
#         headers = {
#             'Content-Type': 'application/json; charset=utf-8',
#             'Authorization': self._api_key,
#             'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
#             'Accept': 'application/json'  # 비스트리밍 방식
#         }

#         # API 요청 보내기
#         try:
#             response = requests.post(
#                 self._host + self._endpoint,  # 전체 URL
#                 headers=headers,
#                 json=payload
#             )

#             # 응답 처리
#             if response.status_code == HTTPStatus.OK:
#                 return response.json()  # JSON 응답 반환
#             else:
#                 raise ValueError(f"HTTP 오류: {response.status_code}, 메시지: {response.text}")
#         except Exception as e:
#             print(f"상세 에러: {str(e)}")
#             raise ValueError(f"API 요청 중 오류 발생: {str(e)}")
    
class ChatCompletionsExecutor(BaseAPIExecutor):
    def __init__(self, host, api_key, request_id):
        super().__init__(host, api_key, request_id)
        self._endpoint = API_CONFIG['chat_completion_endpoint']

    def execute(self, completion_request, stream=True):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'Authorization': self._api_key,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
            'Accept': 'text/event-stream' if stream else 'application/json'
        }

        if stream:
            with requests.post(self._host + self._endpoint,
                        headers=headers, json=completion_request, stream=True) as r:
                if r.status_code == HTTPStatus.OK:
                    final_result = None
                    for line in r.iter_lines():
                        if line:
                            decoded_line = line.decode("utf-8")
                            if decoded_line.startswith("data:"):
                                try:
                                    data = json.loads(decoded_line.replace("data:", "").strip())
                                    if "message" in data:
                                        final_result = data  # 마지막 결과 저장
                                except json.JSONDecodeError:
                                    continue

                    if final_result:  # 최종 결과 반환
                        return {
                            "content": final_result["message"]["content"],
                            "context": final_result["message"]["content"],
                            "inputLength": final_result["inputLength"],
                            "outputLength": final_result["outputLength"]
                        }
                else:
                    raise ValueError(f"오류 발생[1]: HTTP {r.status_code}, 메시지: {r.text}")
        else:
            return super().execute(self._endpoint, payload=completion_request)
                
class SummarizationExecutor(BaseAPIExecutor):
    def __init__(self, host, api_key, request_id):
        super().__init__(host, api_key, request_id)
        self._endpoint = '/testapp/v1/api-tools/summarization/v2'

    def execute(self, completion_request):
        payload = {
            "texts": completion_request["texts"],
            "autoSentenceSplitter": completion_request.get("autoSentenceSplitter",
                                                           True),
            "segCount": completion_request.get("segCount", -1)
        }
        res, status = super().execute(self._endpoint, payload)
        if status == HTTPStatus.OK and "result" in res:
            return res["result"]["text"]
        else:
            error_message = res.get("status", {}).get("message", "Unknown error") if isinstance(res, dict) else "Unknown error"
            raise ValueError(f"오류 발생[2]: HTTP {status}, 메시지: {error_message}")