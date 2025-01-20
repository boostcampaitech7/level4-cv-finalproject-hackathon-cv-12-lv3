from .baseAPI import BaseAPIExecutor
from config.config import API_CONFIG

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
    
class ChatCompletionAPI(BaseAPIExecutor):
    def __init__(self, host, api_key,request_id):
        super().__init__(host, api_key, request_id)
        self._endpoint = "/testapp/v1/chat-completions/HCX-003"
    
    def get_completion(self, messages, top_p=0.8, top_k=0, max_tokens=4096, 
                      temperature=0.5, repeat_penalty=5.0, stop_before=None, 
                      include_ai_filters=True, seed=0):
        """
        챗봇 응답을 생성하는 메서드
        :param messages: 대화 메시지 목록
        :param top_p: 상위 확률 샘플링 임계값
        :param top_k: 상위 k개 토큰 샘플링
        :param max_tokens: 최대 생성 토큰 수
        :param temperature: 생성 다양성 조절
        :param repeat_penalty: 반복 패널티
        :param stop_before: 생성 중단 토큰 목록
        :param include_ai_filters: AI 필터 포함 여부
        :param seed: 랜덤 시드
        :return: API 응답 결과
        """
        if stop_before is None:
            stop_before = []
            
        payload = {
            'messages': messages,
            'topP': top_p,
            'topK': top_k,
            'maxTokens': max_tokens,
            'temperature': temperature,
            'repeatPenalty': repeat_penalty,
            'stopBefore': stop_before,
            'includeAiFilters': include_ai_filters,
            'seed': seed
        }
        
        result = self.execute(self._endpoint, payload)
        return result