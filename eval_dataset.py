from datasets import load_dataset
from tqdm import tqdm
from config.config import OCR_CONFIG, API_CONFIG
from utils import pdf_to_image, images_to_text, clean_text, chunkify_to_num_token, query_and_respond, MultiChatManager
from api import EmbeddingAPI, ChatCompletionsExecutor, SummarizationExecutor
from datebase import DatabaseConnection, DocumentUploader, SessionManager, PaperManager, ChatHistoryManager
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
from difflib import SequenceMatcher
from collections import defaultdict
import pandas as pd

def find_matching_file(domain, target_file_name, base_dir="/data/ephemeral/home/lexxsh/Finalproject/eval/dataset/pdf"):
    """
    domain 폴더 내에서 target_file_name과 절반 정도 일치하는 파일을 찾습니다.
    """
    domain_dir = os.path.join(base_dir, domain)
    if not os.path.exists(domain_dir):
        raise FileNotFoundError(f"Domain directory not found: {domain_dir}")

    best_match = None
    best_ratio = 0

    for file_name in os.listdir(domain_dir):
        # 파일 이름과 target_file_name의 유사도 계산
        ratio = SequenceMatcher(None, file_name, target_file_name).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = file_name

    # 절반 정도 일치하는지 확인 (유사도가 0.5 이상인 경우)
    if best_ratio >= 0.6:
        return os.path.join(domain_dir, best_match)
    else:
        return None

if __name__ == '__main__':
    # 데이터셋 로드 및 불필요한 컬럼 제거
    dataset = load_dataset("allganize/RAG-Evaluation-Dataset-KO")
    ds = dataset['test'].remove_columns(dataset['test'].column_names[4:])  # 처음 4개 컬럼만 남김

    # 데이터셋에서 필요한 컬럼 추출
    domains = ds['domain']
    questions = ds['question']
    target_answers = ds['target_answer']
    target_file_names = ds['target_file_name']

    # 도메인별 통계를 저장할 딕셔너리
    domain_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    # 전체 통계를 저장할 변수
    total_questions = 0
    total_correct = 0

    # 질문, 생성답변, 정답을 저장할 리스트
    question_list = []
    generated_answers_list = []
    target_answers_list = [] 
    
    CHUNK_SIZE = 256
    TOKEN_LIMIT = 4096
    SYSTEM_MESSAGE = """안녕하세요! 저는 논문 도우미 SummarAI입니다. 
    논문을 이해하고 분석하는 데 도움을 드릴 수 있어요. 
    어떤 것이든 물어보세요!"""

    # 모델 초기화
    model = SentenceTransformer("dragonkue/bge-m3-ko")
    reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # 데이터베이스 연결
    db_connection = DatabaseConnection()
    conn = db_connection.connect()

    try:
        # API 초기화
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
        summarization_executor = SummarizationExecutor(
            host=API_CONFIG['host2'],
            api_key=API_CONFIG['api_key'],
            request_id=API_CONFIG['request_id']
        )

        # 세션 생성
        session_manager = SessionManager(conn)
        session_id = session_manager.create_session()

        # 모든 도메인에 대해 처리
        for domain, question, target_answer, target_file_name in zip(domains, questions, target_answers, target_file_names):
            print(f"\nDomain: {domain}")
            print(f"Question: {question}")
            print(f"Target File Name: {target_file_name}")

            # 파일 매칭
            file_path = find_matching_file(domain, target_file_name)
            if not file_path:
                print(f"No matching file found for {target_file_name} in domain {domain}. Skipping this question.")
                continue  # 파일을 찾지 못하면 다음 질문으로 넘어감

            print(f"Matched File: {file_path}")

            # PDF 처리 및 임베딩
            chunked_documents = []
            images = pdf_to_image(file_path)
            print("PDF를 이미지로 변환하였습니다.")

            for i, image in tqdm(enumerate(images), desc="이미지 처리 중"):
                raw_text = images_to_text(image, OCR_CONFIG['host'], OCR_CONFIG['secret_key'])
                cleaned_text = clean_text(raw_text)
                chunks = chunkify_to_num_token(cleaned_text, CHUNK_SIZE)
                for chunk in chunks:
                    chunked_documents.append({
                        "page": int(i + 1),
                        "chunk": chunk
                    })

            for i in tqdm(chunked_documents, desc="Generating Embeddings", total=len(chunked_documents)):
                embedding = model.encode(i["chunk"])
                i["embedding"] = embedding.tolist()

            # 논문 정보 저장
            paper_manager = PaperManager(conn)
            paper_id = paper_manager.store_paper_info(
                session_id=session_id,
                title=file_path,
                authors=None
            )

            # 문서 업로드
            uploader = DocumentUploader(conn)
            uploader.upload_documents(chunked_documents, session_id)
            print("문서 업로드가 완료되었습니다.")

            # 멀티챗 매니저 초기화
            chat_manager = ChatHistoryManager(conn)
            multichat = MultiChatManager()
            multichat.initialize_chat(SYSTEM_MESSAGE)

            # 질문 처리 및 답변 생성
            user_input = question
            relevant_response = query_and_respond(
                query=user_input,
                conn=conn,
                model=model,
                session_id=session_id,
                top_k=5
            )

            request_data = multichat.prepare_chat_request(user_input, context=relevant_response)

            try:
                response = completion_executor.execute(request_data, stream=True)

                if response:
                    # 응답 처리
                    multichat.process_response(response)
                    gen_answer = response['content']

                    # DB에 대화 저장
                    chat_manager.store_conversation(
                        session_id=session_id,
                        user_message=user_input,
                        llm_response=gen_answer
                    )

                    # 토큰 제한 체크
                    if multichat.check_token_limit(request_data["maxTokens"]):
                        print("토큰 제한 도달. 요약을 시작합니다.")
                        summary_text = summarization_executor.execute({
                            "texts": [msg["content"] for msg in multichat.session_state["chat_log"]],
                            "autoSentenceSplitter": True,
                            "segCount": -1
                        })
                        print(f"\n AI: {summary_text}\n")

                        # 요약본 저장
                        chat_manager.store_summary(
                            session_id=session_id,
                            summary=summary_text,
                            summarized_chat_ids=[msg["chat_id"] for msg in multichat.session_state["chat_log"]]
                        )
                    else:
                        print(f"\nAI: {gen_answer}\n")

                    multichat.initialize_chat("")

                    question_list.append(question)
                    generated_answers_list.append(gen_answer)
                    target_answers_list.append(target_answer)
                    
            except Exception as e:
                print(f"대화 중 오류 발생: {str(e)}")

    except Exception as e:
        print(f"대화 전 오류 발생: {str(e)}")
    finally:
        if conn:
            db_connection.close()
            print("DB connection is closed")
            
    # 모든 질문에 대한 처리가 끝난 후 CSV로 저장
    if question_list and generated_answers_list and target_answers_list:
        # DataFrame 생성
        results_df = pd.DataFrame({
            "question": question_list,
            "generated_answer": generated_answers_list,
            "target_answer": target_answers_list
        })

        # CSV 파일로 저장
        output_csv_path = "output/results.csv"
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)  # 디렉토리 생성
        results_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")  # UTF-8 인코딩으로 저장
        print(f"결과가 {output_csv_path}에 저장되었습니다.")