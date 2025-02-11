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

    def get_segmentation(self, text, alpha=-1, seg_cnt=-1):
        payload = {
            "text": text,
            "alpha": alpha,
            "segCnt": seg_cnt
        }
        result = self.execute(self._endpoint, payload)
        return result.get("topicSeg", "Error")


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
                                    data = json.loads(
                                        decoded_line.replace("data:", "").strip())
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
                    raise ValueError(
                        f"오류 발생[1]: HTTP {r.status_code}, 메시지: {r.text}")
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
            error_message = res.get("status", {}).get(
                "message", "Unknown error") if isinstance(res, dict) else "Unknown error"
            raise ValueError(f"오류 발생[2]: HTTP {status}, 메시지: {error_message}")
