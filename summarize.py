from summarizer import Summarizer

from transformers import *
from utils import images_to_text, clean_text, chunkify_with_overlap, query_and_respond, MultiChatManager, abstractive_summarization

from config.config import AI_CONFIG, API_CONFIG

from api import EmbeddingAPI, ChatCompletionsExecutor

import pytextrank
import spacy
import requests
import networkx as nx
import sys
import os

completion_executor = ChatCompletionsExecutor(
            host=API_CONFIG['host'],
            api_key=API_CONFIG['api_key'],
            request_id=API_CONFIG['request_id']
        )

# Load model, model config and tokenizer via Transformers
custom_config = AutoConfig.from_pretrained('allenai/scibert_scivocab_uncased')
custom_config.output_hidden_states=True
custom_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
custom_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', config=custom_config)

model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)
# PDF에서 텍스트 추출





# 텍스트에서 줄바꿈을 기준으로 분리하고 공백이나 특수 문자가 있는 경우 합침
cleaned_text = ' '.join(text.splitlines())

# 중간에 끊어진 부분이 있다면, 공백으로 연결하여 정리된 텍스트를 출력
# print(cleaned_text)
res = model(cleaned_text,  num_sentences=30)
print(res)

res2 = extractive_summarization(cleaned_text, '')
result = abstractive_summarization(res, completion_executor)
print(result)
