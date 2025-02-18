from io import BytesIO
import uuid
import time
import json
import requests


class OCRAPIExecutor:
    def __init__(self, host, secret_key):
        """
        OCR API 실행기를 초기화합니다.
        :param host: API 호스트 URL
        :param secret_key: OCR API 시크릿 키
        """
        self._api_url = f"https://{host}/custom/v1/37528/65e67b30b3cc8f4d84245194c2415f0c980675f1f0961dbfc2ecc51a368e238c/general"
        self._secret_key = secret_key

    def _prepare_request(self, image):
        """
        OCR 요청을 위한 데이터를 준비합니다.
        :param image: PIL Image 객체
        :return: (headers, payload, files)
        """
        image_byte_array = BytesIO()
        image.save(image_byte_array, format='JPEG')
        image_byte_array.seek(0)

        request_json = {
            'images': [
                {
                    'format': 'jpg',
                    'name': 'demo'
                }
            ],
            'requestId': str(uuid.uuid4()),
            'version': 'V2',
            'timestamp': int(round(time.time() * 1000))
        }

        headers = {
            'X-OCR-SECRET': self._secret_key
        }
        payload = {'message': json.dumps(request_json).encode('UTF-8')}
        files = [('file', ('image.jpg', image_byte_array, 'image/jpeg'))]

        return headers, payload, files

    def execute_ocr(self, image):
        """
        OCR API를 호출하여 결과를 반환합니다.
        :param image: PIL Image 객체
        :return: OCR 결과 (JSON 형식)
        """
        headers, payload, files = self._prepare_request(image)
        response = requests.post(
            self._api_url, headers=headers, data=payload, files=files)
        if response.status_code == 200:
            return response.json()
        else:
            return f"Error: {response.status_code}, {response.text}"
