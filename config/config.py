import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'port': os.getenv('DB_PORT')
}

API_CONFIG = {
    'host': os.getenv('NCP_HOST'),
    'api_key': os.getenv('NCP_API_KEY'),
    'api_key_primary_val': os.getenv('NCP_API_KEY_PRIMARY_VAL'),
    'request_id': os.getenv('REQUEST_ID')
}

OCR_CONFIG = {
    'host': os.getenv('OCR_HOST'),
    'secret_key': os.getenv('OCR_SECRET_KEY')
}