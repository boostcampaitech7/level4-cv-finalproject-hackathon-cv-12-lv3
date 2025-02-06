from collections import defaultdict
from functools import lru_cache

from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, status, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from config.config import AI_CONFIG, API_CONFIG
from pdf2text import Pdf2Text, pdf_to_image
from utils import FileManager, MultiChatManager, PaperSummarizer
from utils import model_manager
from utils import split_sentences, chunkify_to_num_token, chunkify_with_overlap
from api import EmbeddingAPI, ChatCompletionsExecutor, SummarizationExecutor

from datebase.connection import DatabaseConnection
from datebase.operations import PaperManager, DocumentUploader, ChatHistoryManager, AdditionalFileUploader

from sentence_transformers import SentenceTransformer


@lru_cache()
def get_db_connection():
    db_connection = DatabaseConnection()
    conn = db_connection.connect()
    try:
        yield conn
    finally:
        conn.close()


def get_file_manager(conn=Depends(get_db_connection)):
    return FileManager(conn)


def get_paper_manager(conn=Depends(get_db_connection)):
    return PaperManager(conn)


def get_document_manager(conn=Depends(get_db_connection)):
    return DocumentUploader(conn)


def get_add_file_uploader(conn=Depends(get_db_connection)):
    return AdditionalFileUploader(conn)


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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React 앱의 주소
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

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
    global user_id
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="PDF 파일만 업로드 가능합니다.")

        # Byte로 변경
        # stream = io.BytesIO(await file.read())

        paper_info = {
            'user_id': user_id,
            'title': file.filename,
            'authors': None,
            'abstract': None,
            'year': None
        }

        paper_id = file_manager.store_paper(file, paper_info, user_id)

        return {"success": True, "message": "PDF uploaded successfully", "data": {"filename": paper_info['title'], "file_id": paper_id}}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"PDF 업로드 중 오류 발생: {str(e)}")


@app.post("/chat-bot", status_code=status.HTTP_200_OK)
async def prepare_chatbot_base(req: PdfRequest,
                               file_manager: FileManager = Depends(
                                   get_file_manager),
                               document_mannager: DocumentUploader = Depends(get_document_manager)):
    global user_id

    pdf_id = req.pdf_id
    pdf = file_manager.get_paper(user_id, pdf_id)

    if pdf is None:
        return {"success": False, "message": f"PDF with ID {pdf_id} not found or not uploaded yet"}

    pdf_path = "/data/ephemeral/home/ohs/level4-cv-finalproject-hackathon-cv-12-lv3/pdf_folder/1706.03762v7.pdf"

    sentences = pdf2text_recognize(pdf_path, key="text")

    chunked_documents = chunking_embedding(sentences)

    try:
        document_mannager.upload_documents(
            chunked_documents, user_id, req.pdf_id)

    except Exception as e:
        return {"success": False, "message": f"Fail.. {e}"}
    return {"success": True, "message": "Ready for Chat"}


@app.post("/table-figure", status_code=status.HTTP_200_OK)
async def pdf2text_table_figure(req: PdfRequest,
                                file_manager: FileManager = Depends(get_file_manager)):
    global user_id

    pdf_id = req.pdf_id
    pdf = file_manager.get_paper(user_id, pdf_id)

    if pdf is None:
        return {"success": False, "message": f"PDF with ID {pdf_id} not found or not uploaded yet"}

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
    table_flag = file_manager.store_figures_and_tables(
        match_res, user_id, req.pdf_id)

    # TODO 매칭된 Figure은 정제하여 DeepSeek으로 전달하는 코드

    # TODO DeepSeek 결과물 Vector DB 저장 후
    # figure_flag =

    # TODO 두 작업 모두 종료되면 response 반환

    return {"success": True, "message": "여기까지면 정상적으로 온거야 ㅇㅈ? 저장된거 확인해보셈"}

# 요약 및 오디오, 태그, 타임라인 파일 생성하기


@app.post("/pdf/summarize", status_code=status.HTTP_200_OK)
async def summarize_and_get_files(req: PdfRequest,
                                  conn=Depends(get_db_connection),
                                  file_manager: FileManager = Depends(get_file_manager)):
    summarizer = PaperSummarizer()

    paper_file = file_manager.get_paper(user_id, req.pdf_id)

    final_summary = summarizer.generate_summary(paper_file)

    file_manager.extract_summary_content(
        final_summary=final_summary,
        completion_executor=completion_executor,
        user_id=user_id,
        paper_id=req.pdf_id
    )

    return {"success": True, "message": "이야 이거 파일 개잘만든다 바로 storage 확인해라 ㅏㅡㅑ"}


@app.get("/pdf/get_paper")
async def get_paper(req: PdfRequest,
                    file_manager: FileManager = Depends(get_file_manager)):
    pdf_path = file_manager.get_paper(user_id, req.pdf_id)

    if pdf_path:
        return FileResponse(pdf_path, media_type="application/pdf")
    return {"error": "Paper not found"}


@app.get("/pdf/get_figure")
async def get_figure(req: PdfRequest,
                     file_manager: FileManager = Depends(get_file_manager)):
    fig_paths = file_manager.get_figure(user_id, req.pdf_id)

    if fig_paths:
        return {"status": "success", "figures": fig_paths}
    return {"error": "Figures not found"}


@app.get("/pdf/get_timeline")
async def get_timeline(req: PdfRequest,
                       file_manager: FileManager = Depends(get_file_manager)):
    timeline_path = file_manager.get_timeline(user_id, req.pdf_id)

    if timeline_path:
        return FileResponse(timeline_path, media_type="application/json")
    return {"error": "Timeline not found"}


@app.get("/pdf/get_audio")
async def get_audio(req: PdfRequest,
                    file_manager: FileManager = Depends(get_file_manager)):
    audio_path = file_manager.get_audio(user_id, req.pdf_id)

    if audio_path:
        return FileResponse(audio_path, media_type="audio/mpeg")
    return {"error": "Audio not found"}


@app.get("/pdf/get_thumbnail")
async def get_thumbnail(req: PdfRequest,
                        file_manager: FileManager = Depends(get_file_manager)):
    thumbnail_path = file_manager.get_thumbnail(user_id, req.pdf_id)

    if thumbnail_path:
        return FileResponse(thumbnail_path, media_type="image/png")
    return {"error": "Thumbnail not found"}


@app.get("/pdf/get_script")
async def get_script(req: PdfRequest,
                     file_manager: FileManager = Depends(get_file_manager)):
    script_path = file_manager.get_script(user_id, req.pdf_id)

    if script_path:
        return FileResponse(script_path, media_type="application/json")
    return {"error": "Thumbnail not found"}


@app.get("/pdf/get_table")
async def get_table(req: PdfRequest,
                    additional_uploader: AdditionalFileUploader = Depends(get_add_file_uploader)):
    table_info = additional_uploader.search_table_file(user_id, req.pdf_id)

    if table_info:
        return {"status": "success", "tables": table_info}
    return {"status": "error", "message": "Table information not found"}


def pdf2text_recognize(pdf):
    p2t = Pdf2Text(AI_CONFIG['layout_model_path'])

    pdf_images, lang = pdf_to_image(pdf)

    result = [p2t.recognize_only_text(page, lang) for page in pdf_images]

    del p2t

    sentences = [split_sentences(raw_text) for raw_text in result]

    for idx in range(1, len(sentences)):
        sentences[idx] = sentences[idx - 1][-3:] + sentences[idx]

    return sentences


def chunking_embedding(sentences, size=256):
    total_chunks = [chunkify_to_num_token(
        sentence, size) for sentence in sentences]

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

    model = SentenceTransformer("dragonkue/bge-m3-ko")
    chunked_documents = list(map(lambda x: {
                             **x, 'embedding': model.encode(x['chunk']).tolist()}, chunked_documents))

    del model

    return chunked_documents
