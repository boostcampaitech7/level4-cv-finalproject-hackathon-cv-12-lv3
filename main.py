from tqdm import tqdm
from config.config import OCR_CONFIG, API_CONFIG
from utils import pdf_to_image, images_to_text, clean_text, chunkify_to_num_token, query_and_respond, MultiChatManager
from utils import llm_refine, query_and_respond_reranker_compare
from api import EmbeddingAPI, ChatCompletionsExecutor, SummarizationExecutor
from datebase import DatabaseConnection, DocumentUploader, SessionManager, PaperManager, ChatHistoryManager
from sentence_transformers import SentenceTransformer, CrossEncoder

if __name__ == '__main__':

    # 파일 설정 부분
    FILE_NAME = "transformer.pdf"
    CHUNK_SIZE = 256

    TOKEN_LIMIT = 4096
    conn = None
    SYSTEM_MESSAGE = """안녕하세요! 저는 논문 도우미 SummarAI입니다. 
    논문을 이해하고 분석하는 데 도움을 드릴 수 있어요. 
    어떤 것이든 물어보세요!"""
    
    model = SentenceTransformer("dragonkue/bge-m3-ko")
    reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    ## 데이터베이스 연결
    db_connection = DatabaseConnection()
    conn = db_connection.connect()
    
    try:
        # 2. API 초기화
        embedding_api = EmbeddingAPI(
            host = API_CONFIG['host2'],
            api_key=API_CONFIG['api_key'],
            request_id=API_CONFIG['request_id']
        ) 
        completion_executor = ChatCompletionsExecutor(
            host = API_CONFIG['host'],
            api_key=API_CONFIG['api_key'],
            request_id=API_CONFIG['request_id']
        )

        summarization_executor = SummarizationExecutor(
            host = API_CONFIG['host2'],
            api_key=API_CONFIG['api_key'],
            request_id=API_CONFIG['request_id']
        )

        # 3. PDF 처리 및 임베딩
        chunked_documents = []
        images = pdf_to_image(FILE_NAME)
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
            # embedding = embedding_api.get_embedding(i["chunk"])
            #  i["embedding"] = embedding
            embedding = model.encode(i["chunk"])
            i["embedding"] = embedding.tolist()

        db_connection = DatabaseConnection()
        conn = db_connection.connect()

        # 1. 세션 생성
        session_manager = SessionManager(conn)
        session_id = session_manager.create_session()

        # 4. 논문 정보 저장
        paper_manager = PaperManager(conn)
        paper_id = paper_manager.store_paper_info(
            session_id=session_id,
            title=FILE_NAME,
            authors=None
        )

        # 5. 문서 업로드
        uploader = DocumentUploader(conn)
        uploader.upload_documents(chunked_documents)
        
        print("문서 업로드가 완료되었습니다.")

        # 6. 멀티챗 매니저 초기화
        chat_manager = ChatHistoryManager(conn)
        multichat = MultiChatManager()

        # 7. 시스템 메시지 설정
        # system_message = input("시스템 메시지를 입력하세요: ")
        multichat.initialize_chat(SYSTEM_MESSAGE)

        # 8. 대화 루프
        while True:
            user_input = input("사용자: ")
            if user_input.lower() in ['exit', 'quit', '대화종료']:
                break
            
            # 8-1. 질의를 강화하기
            enhaced_query = llm_refine(user_input, completion_executor)
            if enhaced_query:
                print(f"질의강화 검색문: {enhaced_query}")
                user_input = enhaced_query
            else:
                print("질의 강화 쿼리가 존재하지 않습니다.")
            # relevant_response = query_and_respond_reranker_compare(
            #     query=user_input,
            #     conn=conn,
            #     model=model,
            #     reranker_model=reranker_model,  # Cross-Encoder 모델 전달
            #     session_id=session_id,
            #     top_k=5
            # )
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

                    # DB에 대화 저장
                    chat_manager.store_conversation(
                        session_id=session_id,
                        user_message=user_input,
                        llm_response=response['content']
                    )
                    
                    # 토큰 제한 체크
                    if multichat.check_token_limit(request_data["maxTokens"]):
                        print("토큰 제한 도달. 요약을 시작합니다.")
                        summary_text = summarization_executor.execute({
                            "texts": [msg["content"] for msg in
                                      multichat.session_state["chat_log"]],
                            "autoSentenceSplitter": True,
                            "segCount": -1
                        })
                        print(f"\n AI: {summary_text}\n")

                        # 요약본 저장
                        chat_manager.store_summary(
                            session_id=session_id,
                            summary=summary_text,
                            summarized_chat_ids=[msg["chat_id"] for msg in
                                                multichat.session_state["chat_log"]]
                        )
                    else:
                        print(f"\nAI: {response['content']}\n")

                    multichat.initialize_chat("")

            except Exception as e:
                print(f"대화 중 오류 발생: {str(e)}")
    except Exception as e:
        print(f"대화 전 오류 발생: {str(e)}")
    finally:
        if conn:
            db_connection.close()
            print("DB connection is closed")