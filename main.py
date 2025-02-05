import argparse,re

from tqdm import tqdm
from config.config import AI_CONFIG, API_CONFIG
from utils import images_to_text, clean_text, chunkify_with_overlap, query_and_respond, MultiChatManager, abstractive_summarization
from utils import llm_refine, process_query_with_reranking_compare, extractive_summarization, split_sentences, extract_paper_metadata, group_academic_paragraphs
from api import EmbeddingAPI, ChatCompletionsExecutor, SummarizationExecutor
from datebase import DatabaseConnection, DocumentUploader, SessionManager, PaperManager, ChatHistoryManager
from pdf2text import Pdf2Text, pdf_to_image
from sentence_transformers import SentenceTransformer, CrossEncoder
from hashlib import sha256
import traceback
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
    lang = "en"

    model = SentenceTransformer("dragonkue/bge-m3-ko")
    # reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker_model = CrossEncoder("jinaai/jina-reranker-v2-base-multilingual", trust_remote_code=True)
    pdf2text = Pdf2Text(AI_CONFIG["layout_model_path"], lang=lang)

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
        chunked_documents = []
        summarized_documents = []
        summary_list = []
        last_two_sentences = []
        full_text = ""
        images = pdf_to_image(FILE_NAME)
        print("PDF를 이미지로 변환하였습니다.")

        for i, image in tqdm(enumerate(images), desc="이미지 처리 중"):
            # raw_text = images_to_text(
            #     image, OCR_CONFIG['host'], OCR_CONFIG['secret_key'])
            # cleaned_text = clean_text(raw_text)
            raw_text = pdf2text.recognize(image)
            full_text += raw_text + "\n"
            # print(raw_text)
            
            # 문장단위 분할
            sentences = split_sentences(raw_text)
            # sentence_embeddings = model.encode(sentences)
            if last_two_sentences:
                sentences = last_two_sentences + sentences
            # chunks, chunk_embeddings = semantic_chunking(
            #     sentences=sentences,
            #     sentence_embeddings=sentence_embeddings,
            #     threshold=0.4,
            #     target_chunk_size=7,
            #     dynamic_threshold_increment=0.1
            # )
            
            # for chunk, embedding in zip(chunks, chunk_embeddings):
            #     sentence_chunk = split_sentences(chunk)
            #     summary = extractive_summarization(sentence_chunk, model=model, top_n=2)
            #     chunked_documents.append({
            #         "page": int(i + 1),
            #         "chunk": chunk,
            #         "embedding": embedding
            #     })
            #     summarized_documents.append({
            #         "page": int(i + 1),
            #         "summary": summary
            #     })

            # chunks = chunkify_with_overlap(sentences, CHUNK_SIZE)
            chunks = group_academic_paragraphs(sentences)
            for chunk in chunks:
                # sentence_chunk = split_sentences(chunk)
                # summary = extractive_summarization(sentence_chunk, model=model, top_n=7)
                # summary_list.append(summary)
                chunked_documents.append({
                    "page": int(i + 1),
                    "chunk": chunk
                })
                # summarized_documents.append({
                #     "page": int(i + 1),
                #     "summary": summary
                # })
            last_two_sentences = sentences[-2:]
        for i in tqdm(chunked_documents, desc="Generating Embeddings", total=len(chunked_documents)):
        #     # embedding = embedding_api.get_embedding(i["chunk"])
        #     #  i["embedding"] = embedding
            embedding = model.encode(i["chunk"])
            i["embedding"] = embedding.tolist()
            
        import json
        output_path = "chunked_documents"
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(chunked_documents, json_file, ensure_ascii=False, indent=4)
        print(f"청크 데이터를 JSON 파일로 저장했습니다: {output_path}")
        
        # flattened_sentences = [item for sublist in summary_list for item in sublist]
        # summaries = extractive_summarization(flattened_sentences, model=model, top_n=10)
        
        # final_summary = " ".join([item for sublist in summaries for item in sublist])
        
        # print(final_summary)
        # summary_result = abstractive_summarization(final_summary,completion_executor)
        # print(summary_result)
        
        db_connection = DatabaseConnection()
        conn = db_connection.connect()

        # 1. 세션 생성
        session_manager = SessionManager(conn)
        session_id = session_manager.create_session()

        # metadata = extract_paper_metadata(full_text, completion_executor, lang)

        # if metadata is None:
        #     metadata = {
        #         'title': FILE_NAME,
        #         'authors': None,
        #         'abstract': None,
        #         'year': None
        #     }
        #     print("메타데이터 추출 실패. 기본값을 사용합니다.")
        # else:
        #     print("메타데이터 추출 성공:")
        #     print(f"제목: {metadata['title']}")
        #     print(f"저자: {metadata['authors']}")
        #     print(f"연도: {metadata['year']}")

        metadata = {
                'title': FILE_NAME,
                'authors': None,
                'abstract': None,
                'year': None
        }

        # 4. 논문 정보 저장
        paper_manager = PaperManager(conn)
        paper_id = paper_manager.store_paper_info(
            session_id=session_id,
            title=metadata['title'],
            authors=metadata['authors'],
            abstract=metadata['abstract'],
            year=metadata['year']
        )

        # 5. 문서 업로드
        uploader = DocumentUploader(conn)
        uploader.upload_documents(chunked_documents, session_id)

        print("문서 업로드가 완료되었습니다.")

        # 6. 멀티챗 매니저 초기화
        chat_manager = ChatHistoryManager(
            conn, embedding_api, completion_executor)
        multichat = MultiChatManager()

        # 7. 시스템 메시지 설정
        # system_message = input("시스템 메시지를 입력하세요: ")
        multichat.initialize_chat(SYSTEM_MESSAGE)

        # 8. 대화 루프
        while True:
            context = ""
            user_input = input("사용자: ")
            if user_input.lower().replace(" ", "") in ['exit', 'quit', '대화종료']:
                break

            # 8-1. 질의를 강화하기
            # enhaced_query = llm_refine(user_input, completion_executor)
            # if enhaced_query:
            #     print(f"질의강화 검색문: {enhaced_query}")
            #     user_input = enhaced_query
            # else:
            #     print("질의 강화 쿼리가 존재하지 않습니다.")
            
            # relevant_response = query_and_respond(
            #     query=user_input,
            #     conn=conn,
            #     model=model,
            #     session_id=session_id,
            #     top_k=5,
            #     chat_manager=chat_manager
            # )

            relevant_response = process_query_with_reranking_compare(
                query=user_input,
                conn=conn,
                model=model,
                reranker=reranker_model,
                completion_executor=completion_executor,
                session_id=session_id,
                top_k= 3 if lang == 'en' else 2,
                chat_manager=chat_manager
            )

            context_result = chat_manager.handle_question(
                user_input, session_id)
            
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

            cache_key = f"{session_id}:{user_input}:{sha256(context['content'].encode()).hexdigest()}"

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
                        session_id=session_id,
                        user_message=user_input,
                        llm_response=response['content'],
                        embedding=current_embedding,
                        chat_type=context['type']
                    )

                    # 토큰 제한 체크
                    # if multichat.check_token_limit(request_data["maxTokens"]):
                    #     print("토큰 제한 도달. 요약을 시작합니다.")
                    #     summary_text = summarization_executor.execute({
                    #         "texts": [msg["content"] for msg in
                    #                   multichat.session_state["chat_log"]],
                    #         "autoSentenceSplitter": True,
                    #         "segCount": -1
                    #     })
                    #     print(f"\n AI: {summary_text}\n")

                    #     # 요약본 저장
                    #     chat_manager.store_summary(
                    #         session_id=session_id,
                    #         summary=summary_text,
                    #         summarized_chat_ids=[msg["chat_id"] for msg in
                    #                              multichat.session_state["chat_log"]]
                    #     )

                    #     chat_manager.add_to_cache(
                    #         session_id, user_input, context, summary_text)
                    # else:
                    print(f"\nAI: {response['content']}")
                    chat_manager.add_to_cache(
                        session_id, user_input, context, response['content'])

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