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
from collections import defaultdict
import sys

# 로그 파일 설정
log_filename = "log.txt"
log_file = open(log_filename, "w", encoding="utf-8")
sys.stdout = log_file  # 표준 출력을 파일로 리디렉션


def find_matching_file(domain, target_file_name, base_dir="/data/ephemeral/home/lexxsh/level4-cv-finalproject-hackathon-cv-12-lv3/eval/dataset"):
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

def save_to_csv(data, filename="generated_answers.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Domain", "Question", "Target Answer", "Generated Answer", "Target File"])
        writer.writerows(data)

if __name__ == '__main__':
    dataset = load_dataset("allganize/RAG-Evaluation-Dataset-KO")
    ds = dataset['test'].remove_columns(dataset['test'].column_names[4:])
    
    domains = ds['domain']
    questions = ds['question']
    target_answers = ds['target_answer']
    target_file_names = ds['target_file_name']
    
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
        
        for domain, question, target_answer, target_file_name in zip(domains, questions, target_answers, target_file_names):
            print(f"\nDomain: {domain}")
            print(f"Question: {question}")
            print(f"Target File Name: {target_file_name}")
            
            file_path = find_matching_file(domain, target_file_name)
            if not file_path:
                print(f"No matching file found for {target_file_name} in domain {domain}. Skipping this question.")
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
                    
                    # 출력 추가
                    print(f"\nGenerated Answer:\n{gen_answer}\n")
                    
                    gen_answer = response['content'].replace("\n", " ")
                    generated_data.append([domain, question, target_answer, gen_answer, target_file_name])
                    multichat.initialize_chat("")
                    
                    save_to_csv(generated_data)
            except Exception as e:
                print(f"대화 중 오류 발생: {str(e)}")
    except Exception as e:
        print(f"대화 전 오류 발생: {str(e)}")
    finally:
        print("END.")
        if conn:
            db_connection.close()
            print("DB connection is closed")
        log_file.close()  # 로그 파일 닫기