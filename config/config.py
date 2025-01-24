import os
from dotenv import load_dotenv

load_dotenv()

# 데이터베이스 설정
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'port': os.getenv('DB_PORT')
}

# NCP API 설정
API_CONFIG = {
    'host': os.getenv('NCP_HOST'),
    'host2': os.getenv('NCP_HOST2'),
    'api_key': os.getenv('NCP_API_KEY'),
    'request_id': os.getenv('REQUEST_ID'),
    'embedding_endpoint': os.getenv('EMBEDDING_API_ENDPOINT'),
    'segmentation_endpoint': os.getenv('SEGMENTATION_API_ENDPOINT'),
    'chat_completion_endpoint': '/testapp/v1/chat-completions/HCX-003'
}

# OCR 설정
OCR_CONFIG = {
    'host': os.getenv('OCR_HOST'),
    'secret_key': os.getenv('OCR_SECRET_KEY')
}

# Google Scholar API 설정
GOOGLE_SCHOLAR_API_KEY = os.getenv('SEARCHAPI_API_KEY')