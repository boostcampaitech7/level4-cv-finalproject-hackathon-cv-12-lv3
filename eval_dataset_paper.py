from datasets import load_dataset
from tqdm import tqdm
from config.config import OCR_CONFIG, API_CONFIG, AI_CONFIG
from utils import pdf_to_image, images_to_text, clean_text, chunkify_to_num_token, query_and_respond, MultiChatManager
from utils import llm_refine, query_and_respond_reranker_compare
from api import EmbeddingAPI, ChatCompletionsExecutor, SummarizationExecutor
from datebase import DatabaseConnection, DocumentUploader, SessionManager, PaperManager, ChatHistoryManager
from sentence_transformers import SentenceTransformer
import os, csv
from difflib import SequenceMatcher
import sys
import pandas as pd

# 로그 파일 설정
log_filename = "log_paper.txt"
log_file = open(log_filename, "w", encoding="utf-8")
sys.stdout = log_file

def find_matching_file(domain, target_file_name, base_dir="/data/ephemeral/home/lexxsh/level4-cv-finalproject-hackathon-cv-12-lv3/eval/dataset_paper"):
    domain_dir = os.path.join(base_dir, domain)
    if not os.path.exists(domain_dir):
        raise FileNotFoundError(f"Domain directory not found: {domain_dir}")
    
    best_match = None
    best_ratio = 0
    
    for file_name in os.listdir(domain_dir):
        ratio = SequenceMatcher(None, file_name, target_file_name).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = file_name
    
    if best_ratio >= 0.6:
        return os.path.join(domain_dir, best_match)
    else:
        return None

def save_to_csv(data, filename="generated_answers_paper.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Domain", "Question", "Target Answer", "Generated Answer", "Target File"])
        writer.writerows(data)

def process_csv_data(csv_path):
    # CSV 파일 읽기
    df = pd.read_csv(csv_path)
    print(f"총 {len(df)}개의 데이터를 로드했습니다.")
    
    generated_data = []
    model = SentenceTransformer("dragonkue/bge-m3-ko")
    db_connection = DatabaseConnection()
    conn = db_connection.connect()
    
    try:
        embedding_api = EmbeddingAPI(
            host=API_CONFIG['host2'],
            api_key=API_CONFIG['api_key'],
            request_id=API_CONFIG['request_id']
        )
        completion_executor = ChatCompletionsExecutor(
            host=API_CONFIG['host'],
            api_key=API_CONFIG['api_key'],
            request_id=API_CONFIG['request_id']
        )
        session_manager = SessionManager(conn)
        session_id = session_manager.create_session()
        
        # DataFrame을 순회하면서 처리
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing documents"):
            domain = row['domain']
            question = row['question']
            target_answer = row['answer']
            file_name = row['name']
            
            print(f"\nID: {row['id']}")
            print(f"Domain: {domain}")
            print(f"Question: {question}")
            print(f"File Name: {file_name}")
            print(f"Question Type: {row['question_type']}")
            print(f"PDF Language: {row['pdf_language']}")
            print(f"Answer Language: {row['answer_language']}")
            
            file_path = find_matching_file(domain, file_name)
            if not file_path:
                print(f"No matching file found for {file_name} in domain {domain}. Skipping this question.")
                continue
            
            print(f"Matched File: {file_path}")
            
            images = pdf_to_image(file_path)
            print("PDF를 이미지로 변환하였습니다.")
            chunked_documents = []
            
            for i, image in tqdm(enumerate(images), desc="이미지 처리 중"):
                raw_text = images_to_text(image, OCR_CONFIG['host'], OCR_CONFIG['secret_key'])
                cleaned_text = clean_text(raw_text)
                chunks = chunkify_to_num_token(cleaned_text, 256)
                for chunk in chunks:
                    chunked_documents.append({"page": int(i + 1), "chunk": chunk})
            
            for i in tqdm(chunked_documents, desc="Generating Embeddings", total=len(chunked_documents)):
                embedding = model.encode(i["chunk"])
                i["embedding"] = embedding.tolist()
            
            uploader = DocumentUploader(conn)
            uploader.upload_documents(chunked_documents, session_id)
            print("문서 업로드가 완료되었습니다.")
            
            multichat = MultiChatManager()
            multichat.initialize_chat("안녕하세요! 저는 논문 도우미 SummarAI입니다.")
            
            relevant_response = query_and_respond(
                query=question,
                conn=conn,
                model=model,
                session_id=session_id,
                top_k=5
            )
            
            request_data = multichat.prepare_chat_request(question, context=relevant_response)
            
            try:
                response = completion_executor.execute(request_data, stream=True)
                if response:
                    multichat.process_response(response)
                    gen_answer = response['content']
                    print(f"\nGenerated Answer:\n{gen_answer}\n")
                    
                    gen_answer = response['content'].replace("\n", " ")
                    generated_data.append([domain, question, target_answer, gen_answer, file_name])
                    multichat.initialize_chat("")
                    
                    # 각 응답 후 CSV 파일 업데이트
                    save_to_csv(generated_data)
            except Exception as e:
                print(f"대화 중 오류 발생: {str(e)}")

    except Exception as e:
        print(f"처리 중 오류 발생: {str(e)}")
    finally:
        if conn:
            db_connection.close()
            print("DB connection is closed")
        log_file.close()

if __name__ == '__main__':
    csv_path = "/data/ephemeral/home/lexxsh/level4-cv-finalproject-hackathon-cv-12-lv3/eval/dataset_paper/dataset_2.csv"  # CSV 파일 경로를 지정하세요
    process_csv_data(csv_path)