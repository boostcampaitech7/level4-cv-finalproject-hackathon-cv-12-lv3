from datasets import load_dataset
from tqdm import tqdm
from config.config import OCR_CONFIG, API_CONFIG, AI_CONFIG
from utils import pdf_to_image, images_to_text, clean_text, chunkify_to_num_token, query_and_respond, MultiChatManager
from api import EmbeddingAPI, ChatCompletionsExecutor
from datebase import DatabaseConnection, DocumentUploader, SessionManager
from sentence_transformers import SentenceTransformer
import os

def process_single_pdf(pdf_path, questions):
    """
    단일 PDF 파일을 처리하고 주어진 질문들에 대한 답변을 생성합니다.
    
    Args:
        pdf_path (str): PDF 파일의 경로
        questions (list): 질문 리스트
    """
    print(f"Processing PDF: {pdf_path}")
    
    # 모델과 데이터베이스 연결 초기화
    model = SentenceTransformer("dragonkue/bge-m3-ko")
    db_connection = DatabaseConnection()
    conn = db_connection.connect()
    
    try:
        # API 초기화
        completion_executor = ChatCompletionsExecutor(
            host=API_CONFIG['host'],
            api_key=API_CONFIG['api_key'],
            request_id=API_CONFIG['request_id']
        )
        
        # 세션 생성
        session_manager = SessionManager(conn)
        session_id = session_manager.create_session()
        
        # PDF를 이미지로 변환
        print("PDF를 이미지로 변환 중...")
        images = pdf_to_image(pdf_path)
        chunked_documents = []
        
        # 이미지에서 텍스트 추출 및 청킹
        print("텍스트 추출 및 처리 중...")
        for i, image in enumerate(images):
            raw_text = images_to_text(image, OCR_CONFIG['host'], OCR_CONFIG['secret_key'])
            cleaned_text = clean_text(raw_text)
            chunks = chunkify_to_num_token(cleaned_text, 256)
            for chunk in chunks:
                chunked_documents.append({"page": int(i + 1), "chunk": chunk})
        
        # 임베딩 생성
        print("임베딩 생성 중...")
        for i in chunked_documents:
            embedding = model.encode(i["chunk"])
            i["embedding"] = embedding.tolist()
        
        # 문서 업로드
        uploader = DocumentUploader(conn)
        uploader.upload_documents(chunked_documents, session_id)
        print("문서 처리가 완료되었습니다.")
        
        # 각 질문에 대한 답변 생성
        multichat = MultiChatManager()
        
        print("\n질문에 대한 답변을 생성합니다:")
        for idx, question in enumerate(questions, 1):
            print(f"\n질문 {idx}: {question}")
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
                    answer = response['content']
                    print(f"\n답변:\n{answer}\n")
                    print("-" * 80)
            except Exception as e:
                print(f"답변 생성 중 오류 발생: {str(e)}")
                
    except Exception as e:
        print(f"처리 중 오류 발생: {str(e)}")
    finally:
        if conn:
            db_connection.close()
            print("DB 연결이 종료되었습니다.")

if __name__ == '__main__':
    # PDF 파일 경로와 질문들을 직접 지정
    pdf_path = "/data/ephemeral/home/lexxsh/level4-cv-finalproject-hackathon-cv-12-lv3/eval/dataset_paper/physics/Observation of a new boson at a mass of 125 GeV with the.pdf"  # PDF 파일 경로를 여기에 입력하세요
    
    # 질문 리스트
    questions = [
        "What are the key differences between the multivariate analysis and the cross-check analysis used for the H → γγ search, and how do their sensitivities compare?",
    ]
    
    # 실행
    process_single_pdf(pdf_path, questions)