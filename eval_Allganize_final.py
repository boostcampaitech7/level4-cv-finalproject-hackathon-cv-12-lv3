from datasets import load_dataset
from tqdm import tqdm
from config.config import API_CONFIG, AI_CONFIG
from utils import MultiChatManager, chunkify_to_num_token, conversation_with_images
from utils import process_query_with_reranking_compare, split_sentences, conversation_with_images
from api import EmbeddingAPI, ChatCompletionsExecutor
from datebase import DatabaseConnection, DocumentUploader, SessionManager
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
import csv
from pdf2text import Pdf2Text, pdf_to_image
from difflib import SequenceMatcher
import sys
import subprocess

# 로그 파일 설정
log_filename = "log.txt"
log_file = open(log_filename, "w", encoding="utf-8")
sys.stdout = log_file  # 표준 출력을 파일로 리디렉션


def run_translate(file_name):
    command = ["python", "utils/translate.py", file_name]
    subprocess.run(command, capture_output=True, text=True)


def find_matching_file(domain, target_file_name, base_dir="/data/ephemeral/home/lexxsh/level4-cv-finalproject-hackathon-cv-12-lv3/eval/dataset_Allganize"):
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


def save_to_csv(data, filename="generated_answers_final.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Domain", "Question", "Target Answer",
                        "Generated Answer", "Target File"])
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
    reranker_model = CrossEncoder(
        "jinaai/jina-reranker-v2-base-multilingual", trust_remote_code=True)
    pdf2text = Pdf2Text(AI_CONFIG["layout_model_path"])

    CHUNK_SIZE = 256

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
                print(
                    f"No matching file found for {target_file_name} in domain {domain}. Skipping this question.")
                continue

            print(f"Matched File: {file_path}")

            images, lang = pdf_to_image(file_path)
            print("PDF를 이미지로 변환하였습니다.")
            lang = 'korean'

            chunked_documents = []
            last_three_sentences = []
            for i, image in tqdm(enumerate(images), desc="이미지 처리 중"):
                raw_text = pdf2text.recognize_only_text(image, lang)
                sentences = split_sentences(raw_text)
                if last_three_sentences:
                    sentences = last_three_sentences + sentences
                chunks = chunkify_to_num_token(sentences, 256)

                for chunk in chunks:
                    chunked_documents.append(
                        {"page": int(i + 1), "chunk": chunk})
                last_three_sentences = sentences[-3:]

                matched_res, unmatched_res = pdf2text.recognize_only_table_figure(
                    image, lang)

                # 매칭된 테이블 그대로 넣어주기.
                for table_data in matched_res["table"]:
                    table_caption = table_data["caption_text"]
                    table_text = table_data["obj"] if table_data["obj"] else "Table OCR Failed"
                    chunked_documents.append({
                        "page": int(i + 1),
                        "chunk": f"{table_caption}: {table_text}"
                    })
                # 1. 매칭된 Figure 중 caption_text만 받아오기.
                # 2.
                for figure_data in matched_res["figure"]:
                    figure_img = figure_data["obj"]
                    caption_text = figure_data["caption_text"]
                    caption_number = figure_data["caption_number"]
                    # 이미지가 존재하는 경우에 딥식에 보내기
                    if figure_img:
                        res = conversation_with_images(
                            "deepseek-ai/deepseek-vl-7b-chat", figure_img, image_description=caption_text)
                        chunked_documents.append({
                            "page": int(i + 1),
                            "chunk": f"{res}"
                        })
                # 1. 매칭이 실패한 친구들중 Table 저장
                # 2. 매칭이 실패한 친구들 중 이미지 딥식 보내기
                for obj in unmatched_res["obj"]:
                    obj_type = obj["type"]
                    obj_item = obj["item"]
                    if obj_type == "Table":
                        chunked_documents.append({
                            "page": int(i + 1),
                            "chunk": f"Table: {obj_item}"
                        })
                    if obj_type == "Figure":
                        res = conversation_with_images(
                            "deepseek-ai/deepseek-vl-7b-chat", obj_item)
                        chunked_documents.append({
                            "page": int(i + 1),
                            "chunk": f"Figure: {res}"
                        })
                # 매칭이 실패한 친구들중 caption은 다 저장한다.
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
                top_k=3,
            )
            request_data = multichat.prepare_chat_request(
                question, context=relevant_response)

            try:
                response = completion_executor.execute(
                    request_data, stream=True)
                if response:
                    multichat.process_response(response)
                    gen_answer = response['content']

                    # 출력 추가
                    print(f"\nGenerated Answer:\n{gen_answer}\n")

                    gen_answer = response['content'].replace("\n", " ")
                    generated_data.append(
                        [domain, question, target_answer, gen_answer, target_file_name])
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
