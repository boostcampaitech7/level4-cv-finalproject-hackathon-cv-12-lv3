import argparse,re
from tqdm import tqdm
from config.config import AI_CONFIG, API_CONFIG
from utils import MultiChatManager, FileManager, PaperSummarizer
from utils import process_query_with_reranking_compare, split_sentences, group_academic_paragraphs, write_full_script, script_to_speech
from utils import chunkify_to_num_token, chunkify_with_overlap
from api import EmbeddingAPI, ChatCompletionsExecutor, SummarizationExecutor
from datebase import DatabaseConnection, DocumentUploader, SessionManager, PaperManager, ChatHistoryManager
from pdf2text import Pdf2Text, pdf_to_image
from sentence_transformers import SentenceTransformer, CrossEncoder
from hashlib import sha256
from utils.pdf2zh import translate
import traceback
import subprocess
import json
import os
from PIL import Image

def run_translate(file_name):
    command = ["python", "utils/translate.py", file_name]
    subprocess.run(command, capture_output=True, text=True)

    filename = os.path.splitext(os.path.basename(file_name))[0]
    mono_pdf_path = f"{filename}-mono.pdf"

    with open('new.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    return {
        'translated_pdfs': mono_pdf_path,
        'translated_json': json_data
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Using Pdf Path")
    args = parser.parse_args()

    # 파일 설정 부분
    FILE_NAME = args.path
    CHUNK_SIZE = 256

    TOKEN_LIMIT = 4096
    conn = None
    SYSTEM_MESSAGE = """안녕하세요! 저는 논문 도우미 SummarAI입니다. 
    논문을 이해하고 분석하는 데 도움을 드릴 수 있어요. 
    어떤 것이든 물어보세요!"""

    user_id = 'admin'

    model = SentenceTransformer("dragonkue/bge-m3-ko")
    reranker_model = CrossEncoder("jinaai/jina-reranker-v2-base-multilingual", trust_remote_code=True)
    pdf2text = Pdf2Text(AI_CONFIG["layout_model_path"])

    chunked_documents = []
    summarized_documents = []
    summary_list = []
    last_three_sentences = []
    all_match_res = []

    # 데이터베이스 연결
    db_connection = DatabaseConnection()
    conn = db_connection.connect()

    try:
        # 2. API 초기화
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

        # 3. PDF 처리 및 임베딩
        images, lang = pdf_to_image(FILE_NAME)
        print("PDF를 이미지로 변환하였습니다.")

        translation_result = run_translate(FILE_NAME)
        mono_pdf_path = translation_result['translated_pdfs']
        new_data = translation_result['translated_json']

        for i, image in tqdm(enumerate(images), desc="이미지 처리 중"):
            raw_text, match_res, unmatch_res = pdf2text.recognize(image, lang)

            if match_res:
                match_res['page'] = i + 1
                all_match_res.append(match_res)
            sentences = split_sentences(raw_text)
            if last_three_sentences:
                sentences = last_three_sentences + sentences
            chunks = chunkify_to_num_token(sentences, 256)
            for chunk in chunks:
                    chunked_documents.append({"page": int(i + 1), "chunk": chunk})
            last_three_sentences = sentences[-3:]

            if str(i) in new_data:
                new_text = new_data[str(i)]
                new_chunks = chunkify_with_overlap(new_text, CHUNK_SIZE)
                for chunk in new_chunks:
                    chunked_documents.append({
                        "page": int(i + 1),
                        "chunk": chunk
                    })

        for doc in tqdm(chunked_documents, desc="Generating Embeddings", total=len(chunked_documents)):
            embedding = model.encode(doc["chunk"])
            doc["embedding"] = embedding.tolist()
            
        db_connection = DatabaseConnection()
        conn = db_connection.connect()

        # 1. 세션 생성
        # session_manager = SessionManager(conn)
        # session_id = session_manager.create_session()

        file_manager = FileManager(conn)

        paper_info = {
            'user_id': user_id,
            'title': FILE_NAME,
            'authors': None,
            'abstract': None,
            'year': None
        }

        paper_id = file_manager.store_paper(FILE_NAME, paper_info, user_id)

        # [o] TODO 2. 논문 번역 및 저장
        file_manager.update_translated_paper(mono_pdf_path, user_id, paper_id)

        # [o] TODO 3. Figure & Table 저장 (Figure & table 추출 들어가있는지)
        file_manager.store_figures_and_tables(all_match_res, user_id, paper_id,
                                              model, completion_executor)

        # [o] TODO 4. 요약 추출 및 저장 (요약은 summary로 하는 거 같은데 txt로 반환이 안되는 거 같음)
        # [o] TODO 5. 요약을 통한 오디오 파일 생성 및 저장 (요약이 되는대로 함수만 엮으면 될듯)
        # [o] TODO 6. 태그 추출 저장
        # [o] TODO 7. 타임라인 정보 생성 및 저장 (타임라인은 요약 시 바로 생성되는 것으로 확인)
        summarizer = PaperSummarizer()
        final_summary = summarizer.generate_summary(pdf_filepath=FILE_NAME)
        file_manager.extract_summary_content(
            final_summary=final_summary,
            completion_executor=completion_executor,
            user_id=user_id,
            paper_id=paper_id
        )

        # 8. 문서 업로드
        uploader = DocumentUploader(conn)
        uploader.upload_documents(chunked_documents, user_id, paper_id)

        print("문서 업로드가 완료되었습니다.")

        # 6. 멀티챗 매니저 초기화
        chat_manager = ChatHistoryManager(
            conn, embedding_api, completion_executor)
        multichat = MultiChatManager()

        # 7. 시스템 메시지 설정
        multichat.initialize_chat(SYSTEM_MESSAGE)

        # 8. 대화 루프
        while True:
            context = ""
            user_input = input("사용자: ")
            if user_input.lower().replace(" ", "") in ['exit', 'quit', '대화종료']:
                break

            relevant_response = process_query_with_reranking_compare(
                query=user_input,
                conn=conn,
                model=model,
                reranker=reranker_model,
                completion_executor=completion_executor,
                user_id=user_id,
                paper_id=paper_id,
                top_k= 3 if lang == 'en' else 2,
                chat_manager=chat_manager
            )

            context_result = chat_manager.handle_question(
                user_input, user_id=user_id, paper_id=paper_id)
            
            if relevant_response is not None:
                if relevant_response['type'] == "reference":
                    context = {
                        "type": relevant_response['type'],
                        "content": f"{context_result['content']}\n{relevant_response['content']}" if
                        context_result else relevant_response['content']
                    }
                elif relevant_response['type'] in ["unrelated", "no_result"]:
                    context = {
                        "type": relevant_response['type'],
                        "content": relevant_response['message']
                    }
                else:
                    context = {
                        "type": relevant_response['type'],
                        "content": relevant_response['message']
                    }
            else:
                print("응답이 없습니다.")

            cache_key = f"{user_id}:{paper_id}:{user_input}:{sha256(context['content'].encode()).hexdigest()}"

            if cache_key in chat_manager.cache:
                cached_response = chat_manager.cache[cache_key]
                request_data = multichat.prepare_chat_request(
                    user_input,
                    context=f"캐시된 응답:\n{cached_response}"
                )
            else:
                request_data = multichat.prepare_chat_request(
                    user_input,
                    context=context
                )

            try:
                response = completion_executor.execute(
                    request_data, stream=True)

                if response:
                    # 응답 처리
                    multichat.process_response(response)

                    current_embedding = embedding_api.get_embedding(user_input)

                    # DB에 대화 저장
                    chat_manager.store_conversation(
                        user_id=user_id,
                        paper_id=paper_id,
                        user_message=user_input,
                        llm_response=response['content'],
                        embedding=current_embedding,
                        chat_type=context['type']
                    )

                    print(f"\nAI: {response['content']}")
                    chat_manager.add_to_cache(
                        user_id, paper_id, user_input, context, response['content'])

                    multichat.initialize_chat("")

            except Exception as e:
                print(f"대화 중 오류 발생: {str(e)}")
    except Exception as e:
        traceback.print_exc()
        print(f"대화 전 오류 발생: {str(e)}")
    finally:
        if conn:
            db_connection.close()
            print("DB connection is closed")