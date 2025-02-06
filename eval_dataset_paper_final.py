from datasets import load_dataset
from tqdm import tqdm
from config.config import OCR_CONFIG, API_CONFIG, AI_CONFIG
from utils import pdf_to_image, images_to_text, clean_text, chunkify_to_num_token, query_and_respond, MultiChatManager
from utils import llm_refine, process_query_with_reranking_compare, extractive_summarization, split_sentences, extract_paper_metadata, group_academic_paragraphs
from api import EmbeddingAPI, ChatCompletionsExecutor, SummarizationExecutor
from datebase import DatabaseConnection, DocumentUploader, SessionManager, PaperManager, ChatHistoryManager
from sentence_transformers import SentenceTransformer, CrossEncoder
from pdf2text import Pdf2Text, pdf_to_image
import os, csv
from difflib import SequenceMatcher
import sys, json
import subprocess
import pandas as pd

# 로그 파일 설정
log_filename = "log_paper.txt"
log_file = open(log_filename, "w", encoding="utf-8")
sys.stdout = log_file

def run_translate(file_name):
    command = ["python", "utils/translate.py", file_name]
    subprocess.run(command, capture_output=True, text=True)

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
    reranker_model = CrossEncoder("jinaai/jina-reranker-v2-base-multilingual", trust_remote_code=True)
    pdf2text = Pdf2Text(AI_CONFIG["layout_model_path"])
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
            
            images, lang = pdf_to_image(file_path)
            print("PDF를 이미지로 변환하였습니다.")
            chunked_documents = []
            
            run_translate(file_path)
            with open("new.json", "r", encoding="utf-8") as f:
                new_data = json.load(f)
                
            chunked_documents = []
            last_three_sentences = []
            for i, image in tqdm(enumerate(images), desc="이미지 처리 중"):
                raw_text = pdf2text.recognize_only_text(image, lang)
                sentences = split_sentences(raw_text)
                if last_three_sentences:
                    sentences = last_three_sentences + sentences
                chunks = chunkify_to_num_token(sentences, 256)
                
                for chunk in chunks:
                        chunked_documents.append({"page": int(i + 1), "chunk": chunk})
                last_three_sentences = sentences[-3:]
                
                if str(i) in new_data:
                    new_text = new_data[str(i)]
                    new_chunks = chunkify_to_num_token(new_text, 256)
                    for chunk in new_chunks:
                        chunked_documents.append({
                            "page": int(i + 1),
                            "chunk": chunk
                        })
                matched_res, unmatched_res = pdf2text.recognize_only_table_figure(image, lang)
                
                #매칭된 테이블 그대로 넣어주기.
                for table_data in matched_res["table"]:
                    table_caption = table_data["caption_text"]
                    table_text = table_data["obj"] if table_data["obj"] else "Table OCR Failed"
                    chunked_documents.append({
                        "page": int(i + 1),
                        "chunk": f"{table_caption}: {table_text}"
                    })
                #매칭된 Figure 중 caption_text만 받아오기.
                for figure_data in matched_res["figure"]:
                    caption_text = figure_data["caption_text"]
                    caption_number = figure_data["caption_number"]
                    if caption_text:  # ✅ caption이 존재할 때만 저장
                        figure_prefix = f"Figure{caption_number}:" if caption_number else "Figure:"
                        chunked_documents.append({
                            "page": int(i + 1),
                            "chunk": f"{figure_prefix}{caption_text}"
                        })
                #매칭이 실패한 친구들중 Table 만 저장한다.
                for obj in unmatched_res["obj"]:
                    obj_type = obj["type"]
                    obj_item = obj["item"]
                    if obj_type == "Table":
                        chunked_documents.append({
                        "page": int(i + 1),
                        "chunk": f"Table: {obj_item}"
                    })
                #매칭이 실패한 친구들중 caption은 다 저장한다.
                for caption in unmatched_res["caption"]:
                    caption_item = caption["item"]
                    chunked_documents.append({
                        "page": int(i + 1),
                        "chunk": f"{caption_item}:"
                    })
                        

            for i in tqdm(chunked_documents, desc="Generating Embeddings", total=len(chunked_documents)):
                embedding = model.encode(i["chunk"])
                i["embedding"] = embedding.tolist()
            
            uploader = DocumentUploader(conn)
            uploader.upload_documents(chunked_documents, session_id)
            print("문서 업로드가 완료되었습니다.")
            
            multichat = MultiChatManager()
            multichat.initialize_chat("안녕하세요! 저는 논문 도우미 SummarAI입니다.")
            
            relevant_response = process_query_with_reranking_compare(
                query=question,
                conn=conn,
                model=model,
                reranker=reranker_model,
                completion_executor=completion_executor,
                session_id=session_id,
                top_k= 3,
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
    csv_path = "/data/ephemeral/home/lexxsh/level4-cv-finalproject-hackathon-cv-12-lv3/eval/dataset_paper/dataset_2.csv" 
    process_csv_data(csv_path)