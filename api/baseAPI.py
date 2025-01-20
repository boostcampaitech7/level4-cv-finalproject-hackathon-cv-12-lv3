import http.client, json

class BaseAPIExecutor:
    def __init__(self, host, api_key, api_key_primary_val, request_id):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id
    
    def _send_request(self, endpoint, payload):
        
        """
        공통 POST 요청 메서드
        :param endpoint: API 엔드포인트 (예: /v1/api-tools/tokenize/LK-D2)
        :param payload: 요청 데이터 (JSON 형식)
        :return: API 응답 결과
        """
        
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id
        }
        
        conn = http.client.HTTPSConnection(self._host)
        conn.request('POST', endpoint, json.dumps(payload), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()
        return result
    def execute(self, endpoint, payload):
        """
        공통 실행 메서드
        :param endpoint: API 엔드포인트
        :param payload: 요청 데이터
        :return: API 응답 결과
        """
        res = self._send_request(endpoint, payload)
        if res['status']['code'] == '20000':
            return res['result']
        else:
            return f"Error: {res['status']['message']}"