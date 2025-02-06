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
    'chat_completion_endpoint': '/serviceapp/v1/chat-completions/HCX-003',
    'chat_completion_endpoint-light': '/testapp/v1/chat-completions/HCX-DASH-001'
}

# OCR 설정
OCR_CONFIG = {
    'host': os.getenv('OCR_HOST'),
    'secret_key': os.getenv('OCR_SECRET_KEY')
}

# AI API 설정
AI_CONFIG = {
    'claude': os.getenv('CLAUDE'),
    'openai': os.getenv('OPEN_AI'),
    'secret_key': os.getenv('OCR_SECRET_KEY'),
    'layout_model_path': 'pdf2text/models/doclayout_yolo_docstructbench_imgsz1024.pt'
}


# Google Scholar API 설정
GOOGLE_SCHOLAR_API_KEY = os.getenv('SEARCHAPI_API_KEY')

# 네이버 VOICE 설정
VOICE_CONFIG = {
    'voice_client_id': os.getenv('VOICE-CLIENT-ID'),
    'voice_client_secret': os.getenv('VOICE-CLIENT-SECRET')
}

# Object Storage 설정
STORAGE_CONFIG = {
    'ncp_access_key': os.getenv('NCP_ACCESS_KEY'),
    'ncp_secret_key': os.getenv('NCP_SECRET_KEY'),
    'ncp_endpoint': os.getenv('NCP_ENDPOINT'),
    'ncp_region': os.getenv('NCP_REGION'),
    'ncp_bucket_name': os.getenv('NCP_BUCKET_NAME')
}