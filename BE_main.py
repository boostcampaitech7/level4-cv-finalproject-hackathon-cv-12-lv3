import os
import io
from PIL import Image
from fitz import Rect

from fastapi import FastAPI, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from config.config import AI_CONFIG
from pdf2text import Pdf2Text, pdf_to_image
from utils import split_sentences, chunkify_to_num_token, chunkify_with_overlap

from datebase.connection import DatabaseConnection
from datebase.operations import PaperManager, SessionManager, DocumentUploader

from sentence_transformers import SentenceTransformer

app = FastAPI()
db_connection = DatabaseConnection()
conn = db_connection.conn

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 앱의 주소
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# CHECKLIST
# [ ] pdf 업로드
# [ ] 업로드된 pdf가 object storage에 저장
# [ ] 여기서 반환된 pdf id를 통해서 이후 요청에 대한 pdf 가져온다음 image 변환 반복 수행

#

# pdf 업로드 -> 저장 -> id
# response 반환 (id 반환)
# 반환된 id로 다시 요청
#!! summary, translate -> 일단 여기는 내알바 아님!!
# text ocr, table and figure ocr -> text ocr 결과 임베딩하고 vector db 저장, table 하고 figure같이 나오는데,

# table은 바로 db 에 저장
# figure는 deepseek 으로 전달  -> figure에 대한 deepseek의 설명 -> 한번에 저장 or 먼저 저장하고서 update하는 방법
# vector db에 저장이 끝나면 해당 요청에 대한 response 반환 -> 챗봇은 간단한 대화 준비 끝

# table or figure에 대한 deepseek 결과값이 저장되고 response 반환되면 table과 figure에 대한 답변도 가능


@app.get("/", status_code=status.HTTP_200_OK)
def test():
    return {"message": "테스트 성공"}


@app.post("/pdf", status_code=status.HTTP_201_CREATED)
async def upload_pdf(file: UploadFile):
    global conn

    # Byte로 변경
    stream = io.BytesIO(await file.read())

    # TODO pdf를 Object Storage에 저장하는 코드

    # NOTE 현재는 기구현된 코드로 대체
    # pm = PaperManager(conn)
    # id = pm.store_paper_info()
    if id is None:
        return {"error": "PDF upload failed"}

    # user 관련 코드는 어떤 방식으로 처리할지 몰라 보류
    id = 1
    return {"message": "PDF uploaded successfully", "filename": file.filename, "file_id": id}


# NOTE
# 해당 기능을 API로 두는게 어색하다고 느껴짐.
# 차라리 일반 함수로 작성하고 챗봇을 준비하는 API에서 호출하는 방식이 더 자연스럽게 느껴진다.
@app.get("/pdf/text/{pdf_id}", status_code=status.HTTP_200_OK)
async def pdf2text_only_text(pdf_id: int):
    global conn

    # TODO pdf_id와 user 정보를 바탕으로 pdf 가져오기

    # NOTE 기구현된 pdf가져오는 코드로 대체
    # pm = PaperManager(conn)
    # pdf = pm.get_paper_info(pdf_id)

    # if pdf is None:
    #     return {"error" : f"PDF with ID {pdf_id} not found or not uploaded yet"}

    pdf_path = "/data/ephemeral/home/ohs/level4-cv-finalproject-hackathon-cv-12-lv3/pdf_folder/1706.03762v7.pdf"
    model = Pdf2Text(AI_CONFIG['layout_model_path'])
    pdf_images, lang = pdf_to_image(pdf_path)

    result = [model.recognize_only_text(page, lang) for page in pdf_images]

    sentences = [split_sentences(raw_text) for raw_text in result]

    # NOTE 해당 값을 반환하는 함수로 마무리하는 것이 좋아보인다.
    for idx in range(1, len(sentences)):
        sentences[idx] = sentences[idx - 1][-3:] + sentences[idx]

    # return sentences
    """ ============ 아래부터는 분리하면 좋은 부분 ================ """
    # NOTE chunking 부분
    chunks = [chunkify_to_num_token(sentence, 256)
              for sentence in sentences]

    chunked_documents = [{"page": idx + 1, "chunk": chunk}
                         for idx, chunk in enumerate(chunks)]

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
                             **x, 'embedding': model.encode(x['chunk'].tolist())}, chunked_documents))

    # return chunked_documents

    """ ============ 아래부터는 분리하면 좋은 부분 ================ """

    try:
        session_manager = SessionManager(conn)
        session_id = session_manager.create_session()

        uploader = DocumentUploader(conn)
        uploader.upload_documents(chunked_documents, session_id)

    except Exception as e:
        return False, str(e)
    return True


@app.get("/pdf/table-figure", status_code=status.HTTP_200_OK)
async def pdf2text_table_figure():
    global cur_pdf, lang, model
    if cur_pdf is None or lang is None:
        return {"error": "No PDF uploaded yet or Unknown PDF Language"}

    # obj_list, caption_list = [], []
    path = "/data/ephemeral/home/ohs/level4-cv-finalproject-hackathon-cv-12-lv3/api_test"
    for page in cur_pdf:
        match_res, unmatch_res = model.recognize_only_table_figure(page, lang)

        for key in match_res:
            path += f"/{key}"
            os.makedirs(path, exist_ok=True)
            for obj in match_res[key]:
                path += f"/{obj['caption_number']}.png"
                rect = Rect(obj['obj_bbox']).include_rect(
                    Rect(obj['caption_bbox']))
                bb = (rect.top_left[0], rect.top_left[1],
                      rect.bottom_right[0], rect.bottom_right[1])
                page.crop(bb).save(path, "png")
                path = os.path.dirname(path)
            path = os.path.dirname(path)

        path += "/unmatched"
        os.makedirs(path, exist_ok=True)
        for key in unmatch_res:
            if key == 'caption':
                for caption in unmatch_res[key]:
                    print(caption['item'] + f" {caption['type']}")
            else:
                for idx, obj in enumerate(unmatch_res[key]):
                    if isinstance(obj['item'], Image.Image):
                        obj['item'].save(
                            path + f"/{obj['type']}_{idx}.png", 'png')
                    else:
                        print(obj['item'])

    return {"message": "여기까지면 정상적으로 온거야 ㅇㅈ? 저장된거 확인해보셈"}
