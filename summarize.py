from langchain.document_loaders import PyPDFLoader
from summarizer import Summarizer

from transformers import *
from utils import abstractive_summarization, extract_keywords, timeline_str, abstractive_timeline

from config.config import API_CONFIG

from api import ChatCompletionsExecutor
import json
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

# PDF 파일 경로 설정
pdf_filepath = '/data/ephemeral/home/lexxsh/level4-cv-finalproject-hackathon-cv-12-lv3/eval/dataset_paper/social/SOCIAL CAPITAL Its Origins and.pdf'

# PyPDFLoader를 사용하여 PDF 파일 로드
loader = PyPDFLoader(pdf_filepath)
pages = loader.load()

# 각 페이지의 내용을 page와 page_label에 맞게 저장
page_dict = {}
for page in pages:
    page_number = page.metadata.get('page', 'Unknown')
    page_label = page.metadata.get('page_label', 'Unknown')
    page_content = page.page_content.replace('\n', ' ')  # \n을 공백으로 대체
    page_dict[page_number] = {
        'page_label': page_label,
        'content': page_content
    }

# 모든 페이지를 리스트 형태로 저장
page_list = []
for page_number, page_info in page_dict.items():
    page_list.append({
        'page_number': page_number,
        'page_label': page_info['page_label'],
        'content': page_info['content']
    })

# 모든 페이지 출력
for page in page_list:
    print(f"Page Number: {page['page_number']}")
    print(f"Page Label: {page['page_label']}")
    print(f"Content: {page['content']}")
    print("-" * 40)  # 페이지 구분을 위한 구분선

# 각 페이지의 page_label별로 요약 수행
label_summaries = {}
for page in page_list:
    page_label = page['page_label']
    content = page['content']
    
    if page_label not in label_summaries:
        label_summaries[page_label] = ""
    
    summary = model(content, num_sentences=10)
    label_summaries[page_label] += summary + " "

# label_summaries에 저장된 각 page_label별 요약 결과 출력
for label, summary in label_summaries.items():
    print(f"Page Label: {label}")
    print(f"Summary: {summary}")
    print("-" * 40)  # 구분선

# 모든 page_label별 요약을 하나로 합침
combined_summary = " ".join(label_summaries.values())
final_summary = model(combined_summary, num_sentences=30)

# 최종 요약 결과 출력
print("Final Summary:")
print(final_summary)

result = abstractive_summarization(final_summary, completion_executor)

##추출된 요약이라고 가정 (키워드 포함)
query_list = extract_keywords(result)
print(f"태그 : {query_list}")

## 키워드 검색, JSON 파일 저장
# str= timeline_str(query_list)
# print(f"str : {str}")

final_result = f"키워드 : {query_list}\n" + str

response = abstractive_timeline(query_list)

