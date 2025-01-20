from tqdm import tqdm
from config.config import OCR_CONFIG, API_CONFIG
from utils import pdf_to_image, images_to_text, clean_text, chunkify_to_num_token, query_and_respond
from api import EmbeddingAPI, ChatCompletionAPI
from datebase import DatabaseConnection, DocumentUploader

if __name__ == '__main__':

    # 파일 설정 부분
    File_name = "transformer.pdf"
    CHUNK_SIZE = 256
    
    # API들
    embedding_api = EmbeddingAPI(
        host=API_CONFIG['host'],
        api_key=API_CONFIG['api_key'],
        request_id=API_CONFIG['request_id']
    )
    chat_api = ChatCompletionAPI(
        host=API_CONFIG['host'],
        api_key=API_CONFIG['api_key'],
        request_id=API_CONFIG['request_id']
    )
    
    # 주요 변수 선언
    chunked_documents = []
    
    # 로직 시작
    images = pdf_to_image(File_name)
    print("파일을 이미지로 변경 하였습니다.")
    
    ## PDF -> IMG -> OCR -> CLEAN -> CHUNK
    for i, image in tqdm(enumerate(images), desc="Processing images", total=len(images)):
        raw_text = images_to_text(image, OCR_CONFIG['host'], OCR_CONFIG['secret_key'])
        cleaned_text = clean_text(raw_text)
        chunks = chunkify_to_num_token(cleaned_text, CHUNK_SIZE)
        for chunk in chunks:
            chunked_documents.append(
                {
                    "page": int(i + 1),  # 페이지 번호
                    "chunk": chunk,  # 청크된 텍스트
                }
            )
    print("파일을 chunk 완료하였습니다.")
    
    ## CHUNK + EMBEDDING
    for i in tqdm(chunked_documents, desc="Generating Embeddings", total=len(chunked_documents)):
        embedding = embedding_api.get_embedding(i["chunk"])
        i["embedding"] = embedding
    
    ## 데이터베이스 연결
    db_connection = DatabaseConnection()
    conn = db_connection.connect()
    
    try:
        ## UPLOAD DATA
        uploader = DocumentUploader(conn)
        uploader.upload_documents(chunked_documents)
        print("데이터 업로드가 완료되었습니다.")
        
        ## 쿼리 및 응답
        result = query_and_respond("트랜스포머는 RNN에서 무엇이 개선된거야?", conn, embedding_api, chat_api, top_k=5)
        print(result)
        
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        ## 연결 닫기
        if 'conn' in locals():
            db_connection.close()
        print("데이터베이스 연결이 닫혔습니다.")