from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from pdf2text import Pdf2Text
from config.config import AI_CONFIG, API_CONFIG

# 모델 초기화
class ModelManager:
    def __init__(self):
        self.sentence_transformer = SentenceTransformer("dragonkue/bge-m3-ko")
        self.reranker_model = CrossEncoder("jinaai/jina-reranker-v2-base-multilingual", trust_remote_code=True)
        self.pdf2text = Pdf2Text(AI_CONFIG["layout_model_path"])

# 싱글톤 방식
model_manager = ModelManager()