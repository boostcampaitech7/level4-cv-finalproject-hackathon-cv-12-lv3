import http.client
import json
from http import HTTPStatus


class BaseAPIExecutor:
    def __init__(self, host, api_key, request_id):
        # 프로토콜을 제거하지 않고 그대로 저장
        self._host = host
        self._api_key = api_key
        self._request_id = request_id

    def _send_request(self, endpoint, payload):
        """
        공통 POST 요청 메서드
        :param endpoint: API 엔드포인트 (예: /v1/api-tools/tokenize/LK-D2)
        :param payload: 요청 데이터 (JSON 형식)
        :return: API 응답 결과
        """

        headers = {
            'Authorization': self._api_key,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'application/json'
        }

        try:
            # 호스트에서 프로토콜 제거
            host_without_protocol = self._host.replace(
                'https://', '').replace('http://', '')
            conn = http.client.HTTPSConnection(host_without_protocol)
            conn.request('POST', endpoint, json.dumps(payload), headers)
            response = conn.getresponse()
            status = response.status
            result = json.loads(response.read().decode(encoding='utf-8'))
            return result, status
        except Exception as e:
            print(f"상세 에러: {str(e)}")
            raise ValueError(f"API 요청 중 오류 발생: {str(e)}")
        finally:
            conn.close()

    def execute(self, endpoint, payload):
        """
        공통 실행 메서드
        :param endpoint: API 엔드포인트
        :param payload: 요청 데이터
        :return: API 응답 결과
        """
        res, status = self._send_request(endpoint, payload)
        if status == HTTPStatus.OK:
            if res.get('status', {}).get('code') == '20000':
                return res['result']
            else:
                error_message = res.get('status', {}).get(
                    'message', 'Unknown error')
                raise ValueError(f"API 오류: {error_message}")
        else:
            raise ValueError(f"HTTP 오류: {status}")
