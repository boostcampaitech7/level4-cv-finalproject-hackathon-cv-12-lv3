import os
import io
from PIL import Image
from fitz import Rect
from collections import defaultdict
from functools import lru_cache

from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException

from config.config import AI_CONFIG
from pdf2text import Pdf2Text, pdf_to_image
from utils import FileManager
from utils import model_manager
from utils import split_sentences, chunkify_to_num_token, chunkify_with_overlap

from datebase.connection import DatabaseConnection
from datebase.operations import PaperManager, SessionManager, DocumentUploader

from sentence_transformers import SentenceTransformer

@lru_cache()
def get_db_connection():
    db_connection = DatabaseConnection()
    conn = db_connection.connect()
    try:
        yield conn
    finally:
        conn.close()

def get_file_manager(conn = Depends(get_db_connection)):
    return FileManager(conn)

def get_paper_manager(conn = Depends(get_db_connection)):
    return PaperManager(conn)

app = FastAPI()
# db_connection = DatabaseConnection()
# conn = db_connection.connect()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React 앱의 주소
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# text ocr, table and figure ocr -> text ocr 결과 임베딩하고 vector db 저장, table 하고 figure같이 나오는데,

# table은 바로 db 에 저장
# figure는 deepseek 으로 전달  -> figure에 대한 deepseek의 설명 -> 한번에 저장 or 먼저 저장하고서 update하는 방법
# vector db에 저장이 끝나면 해당 요청에 대한 response 반환 -> 챗봇은 간단한 대화 준비 끝

# table or figure에 대한 deepseek 결과값이 저장되고 response 반환되면 table과 figure에 대한 답변도 가능

# 필요 변수 선언
user_id = 'admin'

class PdfRequest(BaseModel):
    pdf_id: int

@app.get("/", status_code=status.HTTP_200_OK)
def test():
    return {"success": True, "message": "테스트 성공"}

@app.post("/pdf", status_code=status.HTTP_201_CREATED)
async def upload_pdf(file: UploadFile,
                     file_manager: FileManager = Depends(get_file_manager)):
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="PDF 파일만 업로드 가능합니다.")

        # Byte로 변경
        stream = io.BytesIO(await file.read())

        paper_info = {
            'user_id': user_id,
            'title': file.filename,
            'authors': None,
            'abstract': None,
            'year': None
        }

        # TODO pdf를 Object Storage에 저장하는 코드
        paper_id = file_manager.store_paper(file, paper_info, user_id)

        # NOTE 현재는 기구현된 코드로 대체
        # pm = PaperManager(conn)
        # id = pm.store_paper_info()
        # if id is None:
        #     return {"error": "PDF upload failed"}

        return {"success": True, "message": "PDF uploaded successfully", "data": {"filename": paper_info['title'], "file_id": paper_id}}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"PDF 업로드 중 오류 발생: {str(e)}")


# NOTE
# 해당 기능을 API로 두는게 어색하다고 느껴짐.
# 차라리 일반 함수로 작성하고 챗봇을 준비하는 API에서 호출하는 방식이 더 자연스럽게 느껴진다.
@app.post("/pdf/text", status_code=status.HTTP_200_OK)
async def pdf2text_only_text(req: PdfRequest,
                             conn = Depends(get_db_connection)):
    print(req.pdf_id)

    # NOTE 테스트 부분
    if req.pdf_id != 1:
        return {"success": False, "message": "예끼 이놈아~ 틀렸다잉"}

    # TODO pdf_id와 user 정보를 바탕으로 pdf 가져오기

    # NOTE 기구현된 pdf가져오는 코드로 대체
    # pm = PaperManager(conn)
    # pdf = pm.get_paper_info(pdf_id)

    # if pdf is None:
    #     return {"success" : False, "message" : f"PDF with ID {pdf_id} not found or not uploaded yet"}

    pdf_path = "/data/ephemeral/home/ohs/level4-cv-finalproject-hackathon-cv-12-lv3/pdf_folder/1706.03762v7.pdf"
    p2t = Pdf2Text(AI_CONFIG['layout_model_path'])
    pdf_images, lang = pdf_to_image(pdf_path)

    result = [p2t.recognize_only_text(page, lang) for page in pdf_images]

    del p2t

    sentences = [split_sentences(raw_text) for raw_text in result]

    # NOTE 해당 값을 반환하는 함수로 마무리하는 것이 좋아보인다.
    for idx in range(1, len(sentences)):
        sentences[idx] = sentences[idx - 1][-3:] + sentences[idx]

    # return sentences
    """ ============ 아래부터는 분리하면 좋은 부분 ================ """
    # NOTE chunking 부분
    total_chunks = [chunkify_to_num_token(sentence, 256)
                    for sentence in sentences]

    chunked_documents = [{"page": idx + 1, "chunk": chunk}
                         for idx, chunks in enumerate(total_chunks) for chunk in chunks]

    # NOTE main.py의 new_data가 어떤 역할인지 잘 몰라 주석처리
    # CHUNK_SIZE = 256
    # for idx in range(len(result)):
    #     str_i = str(idx)
    #     if str_i in new_data:
    #         new_text = new_data[str_i]
    #         new_chunks = chunkify_with_overlap(new_text, CHUNK_SIZE)
    #         chunked_documents.extend(
    #             [{"page": idx + 1, "chunk": chunk} for chunk in new_chunks])

    # NOTE 임베딩 과정
    model = SentenceTransformer("dragonkue/bge-m3-ko")
    chunked_documents = list(map(lambda x: {
                             **x, 'embedding': model.encode(x['chunk']).tolist()}, chunked_documents))

    del model
    # return chunked_documents

    """ ============ 아래부터는 분리하면 좋은 부분 ================ """

    try:
        uploader = DocumentUploader(conn)
        uploader.upload_documents(chunked_documents, user_id, req.pdf_id)

    except Exception as e:
        # return False, str(e)
        return {"success": False, "message": f"Fail.. {e}"}
    # return True
    return {"success": True, "message": "Ready for Chat"}


@app.post("/pdf/table-figure", status_code=status.HTTP_200_OK)
async def pdf2text_table_figure(req: PdfRequest,
                                conn = Depends(get_db_connection),
                                file_manager: FileManager = Depends(get_file_manager)):
    # NOTE 테스트 부분
    if req.pdf_id != 1:
        return {"success": False, "message": "예끼 이놈아~ 틀렸다잉"}

    # obj_list, caption_list = [], []
    pdf_path = "/data/ephemeral/home/ohs/level4-cv-finalproject-hackathon-cv-12-lv3/pdf_folder/1706.03762v7.pdf"
    p2t = Pdf2Text(AI_CONFIG['layout_model_path'])
    pdf_images, lang = pdf_to_image(pdf_path)

    total_match_res = defaultdict(list)
    total_unmatch_res = defaultdict(list)
    for page in pdf_images:
        match_res, unmatch_res = p2t.recognize_only_table_figure(page, lang)

        for key, val in match_res.items():
            total_match_res[key].extend(val)
        for key, val in unmatch_res.items():
            total_unmatch_res[key].extend(val)

    del p2t

    # TODO 매칭된 Table은 Vector DB에 저장하는 코드
    file_manager.store_figures_and_tables(match_res, user_id, req.pdf_id)

    # TODO 매칭된 Figure은 정제하여 DeepSeek으로 전달하는 코드

    # TODO DeepSeek 결과물 Vector DB 저장 후

    # TODO 두 작업 모두 종료되면 response 반환

    return {"success": True, "message": "여기까지면 정상적으로 온거야 ㅇㅈ? 저장된거 확인해보셈"}
