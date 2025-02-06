import os
import torch

from typing import Dict, Any, Optional
from collections import defaultdict
from functools import lru_cache
from base64 import b64encode
from hashlib import sha256

from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, status, Depends, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from config.config import AI_CONFIG, API_CONFIG
from pdf2text import Pdf2Text, pdf_to_image
from utils import FileManager, MultiChatManager, PaperSummarizer
# from utils.model_manager import model_manager
from utils import split_sentences, chunkify_to_num_token, chunkify_with_overlap
from utils import process_query_with_reranking_compare
from api import EmbeddingAPI, ChatCompletionsExecutor, SummarizationExecutor

from datebase.connection import DatabaseConnection
from datebase.operations import PaperManager, DocumentUploader, ChatHistoryManager, AdditionalFileUploader

from sentence_transformers import SentenceTransformer, CrossEncoder
from summarizer import Summarizer

import torch
import gc
import os

@lru_cache()
def get_db_connection():
    db_connection = DatabaseConnection()
    conn = db_connection.connect()
    return conn


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

multi_chat_manager = MultiChatManager()
chat_history_manager = ChatHistoryManager(
    get_db_connection(), embedding_api, completion_executor
)

def get_chat_manager(conn=Depends(get_db_connection)):
    return ChatHistoryManager(conn, embedding_api, completion_executor)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React 앱의 주소
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)


class BaseRequest(BaseModel):
    pdf_id: int
    user_id: str = "admin"


class BaseResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class PdfRequest(BaseRequest):
    @classmethod
    def as_form(
        cls,
        user_id: str = Form('admin'),
        pdf_id: int = Form(0)
    ):
        return cls(user_id=user_id, pdf_id=pdf_id)


class ChatRequest(BaseRequest):
    message: str = ""


@app.get("/", response_model=BaseResponse, response_model_exclude_unset=True)
def test():
    return {'success': True, 'message': "테스트 성공"}


@app.post("/pdf", response_model=BaseResponse, response_model_exclude_unset=True)
async def upload_pdf(file: UploadFile,
                     req: PdfRequest = Depends(PdfRequest.as_form),
                     file_manager: FileManager = Depends(get_file_manager)):
    pdf_id, user_id = req.pdf_id, req.user_id
    try:
        print("=== Debug Info ===")
        print(f"File received: {file.filename}")
        print(f"Request data: {req}")

        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="PDF 파일만 업로드 가능합니다.")

        # Byte로 변경
        # stream = io.BytesIO(await file.read())

        file_content = await file.read()
        print(f"File content length: {len(file_content)}")

        paper_info = {
            'user_id': user_id,
            'title': file.filename,
            'authors': None,
            'abstract': None,
            'year': None
        }
        print(f"Paper info: {paper_info}")

        paper_id = file_manager.store_paper(
            file_content, paper_info, user_id)

        return {"success": True, "message": "PDF uploaded successfully", "data": {"filename": paper_info['title'], "file_id": paper_id}}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"PDF 업로드 중 오류 발생: {str(e)}")


@app.post("/chat-bot", response_model=BaseResponse, response_model_exclude_unset=True)
async def prepare_chatbot_base(req: PdfRequest,
                               file_manager: FileManager = Depends(
                                   get_file_manager),
                               document_mannager: DocumentUploader = Depends(get_document_manager)):
    pdf_id, user_id = req.pdf_id, req.user_id
    pdf_path = file_manager.get_paper(user_id, pdf_id)

    if pdf_path is None:
        return {"success": False, "message": f"PDF with ID {pdf_id} not found or not uploaded yet"}

    sentences, lang = pdf2text_recognize(pdf_path)

    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    chunked_documents = chunking_embedding(sentences)

    try:
        document_mannager.upload_documents(
            chunked_documents, user_id, pdf_id)

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Document 업로드 중에 오류 발생 : {str(e)}")
    return {"success": True, "message": "Ready for Chat"}


@app.post("/chat-bot/message", response_model=BaseResponse, response_model_exclude_unset=True)
async def chat_message(req: ChatRequest,
                       conn=Depends(get_db_connection),
                       paper_manager: PaperManager = Depends(get_paper_manager)):
    pdf_id, user_id, user_input = req.pdf_id, req.user_id, req.message

    multi_chat_manager.initialize_chat("")

    model = SentenceTransformer("dragonkue/bge-m3-ko")
    reranker_model = CrossEncoder(
        "jinaai/jina-reranker-v2-base-multilingual", trust_remote_code=True)

    # paper_info = paper_manager.get_paper_info(user_id, pdf_id)

    relevant_response = process_query_with_reranking_compare(
        query=user_input,
        conn=conn,
        model=model,
        reranker=reranker_model,
        completion_executor=completion_executor,
        user_id=user_id,
        paper_id=pdf_id,
        top_k=3,  # NOTE 함수 변경되면 아래 주석으로 진행
        # top_k=3 if paper_info['lang'] == 'en' else 2,
        chat_manager=chat_history_manager
    )

    context_result = chat_history_manager.handle_question(
        user_input, user_id=user_id, paper_id=pdf_id)

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
        context = {"type": "unrelated",
                   "content": "이 질문은 논문과 관련이 없어요. 논문에 대한 질문을 해주시면 도와드릴게요!"}

    cache_key = f"{user_id}:{pdf_id}:{user_input}:{sha256(context['content'].encode()).hexdigest()}"

    if cache_key in chat_history_manager.cache:
        cached_response = chat_history_manager.cache[cache_key]
        request_data = multi_chat_manager.prepare_chat_request(
            user_input,
            f"캐시된 응답:\n{cached_response}"
        )
    else:
        request_data = multi_chat_manager.prepare_chat_request(
            user_input,
            context
        )
    try:
        response = completion_executor.execute(
            request_data, stream=True
        )

        if response:
            multi_chat_manager.process_response(response)
            current_embedding = embedding_api.get_embedding(user_input)

            chat_history_manager.store_conversation(
                user_id=user_id,
                paper_id=pdf_id,
                user_message=user_input,
                llm_response=response['content'],
                embedding=current_embedding,
                chat_type=context['type']
            )

            chat_history_manager.add_to_cache(
                user_id, pdf_id, user_input, context, response['content']
            )
            return {"success": True, "data": {"message": response}}

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"답변 생성 중 오류 발생 : {str(e)}")


@app.post("/table-figure", response_model=BaseResponse, response_model_exclude_unset=True)
async def pdf2text_table_figure(req: PdfRequest,
                                file_manager: FileManager = Depends(get_file_manager)):
    pdf_id, user_id = req.pdf_id, req.user_id

    pdf_path = file_manager.get_paper(user_id, pdf_id)

    if pdf_path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"PDF with ID {pdf_id} not found or not uploaded yet")

    p2t = Pdf2Text(AI_CONFIG['layout_model_path'])
    model = SentenceTransformer("dragonkue/bge-m3-ko")
    pdf_images, lang = pdf_to_image(pdf_path)

    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    total_match_res = defaultdict(list)
    total_unmatch_res = defaultdict(list)
    for page in pdf_images:
        match_res, unmatch_res = p2t.recognize_only_table_figure(page, lang)

        for key, val in match_res.items():
            total_match_res[key].extend(val)
        for key, val in unmatch_res.items():
            total_unmatch_res[key].extend(val)

    del p2t
    torch.cuda.empty_cache()

    # match_res = {'figure': [{}, {}, {}], 'table': [{}, {}, {}]}
    # TODO 매칭된 Table은 Vector DB에 저장하는 코드
    # TODO 매칭된 Figure은 정제하여 DeepSeek으로 전달하는 코드
    # TODO DeepSeek 결과물 Vector DB 저장 후
    table_flag = file_manager.store_figures_and_tables(
        total_match_res, user_id, req.pdf_id, 
        model, completion_executor)
    
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # TODO 두 작업 모두 종료되면 response 반환
    if table_flag:
        return {"success": True, "message": "여기까지면 정상적으로 온거야 ㅇㅈ? 저장된거 확인해보셈"}
    return {"success": False, "message": "Embedding 값 저장 중 오류 발생"}

# 요약 및 오디오, 태그, 타임라인 파일 생성하기


@app.post("/pdf/summarize", response_model=BaseResponse, response_model_exclude_unset=True)
async def summarize_and_get_files(req: PdfRequest,
                                  file_manager: FileManager = Depends(get_file_manager)):
    pdf_id, user_id = req.pdf_id, req.user_id
    model = Summarizer()

    paper_file = file_manager.get_paper(pdf_id, user_id)

    results = pdf2text_recognize(paper_file, key="summary")

    label_summaries = []
    for result in results:
        summary = model(result, num_sentences=10)
        label_summaries.append(summary)

    combined_summary = " ".join(label_summaries)
    final_summary = model(combined_summary, num_sentences=30)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    file_manager.extract_summary_content(
        final_summary=final_summary,
        completion_executor=completion_executor,
        user_id=user_id,
        paper_id=pdf_id
    )

    return {"success": True, "message": "이야 이거 파일 개잘만든다 바로 storage 확인해라 ㅏㅡㅑ"}


@app.get("/pdf/get_paper")
async def get_paper(req: PdfRequest,
                    file_manager: FileManager = Depends(get_file_manager)):
    pdf_id, user_id = req.pdf_id, req.user_id

    pdf_path = file_manager.get_paper(user_id, pdf_id)

    if pdf_path:
        return FileResponse(pdf_path, media_type="application/pdf")
    return {'success': False, "message": "Paper not found"}

@app.get("/pdf/get_translate_paper")
async def get_translate_paper(req: PdfRequest,
                              file_manager: FileManager = Depends(get_file_manager)):
    pdf_path = file_manager.get_trans_paper(req.user_id, req.pdf_id)

    if pdf_path:
        return FileResponse(pdf_path, media_type="application/pdf")
    return {'success': False, "message": "Paper not found"}

@app.get("/pdf/get_figure")
async def get_figure(req: PdfRequest,
                     file_manager: FileManager = Depends(get_file_manager)):
    pdf_id, user_id = req.pdf_id, req.user_id
    fig_paths = file_manager.get_figure(user_id, pdf_id)

    if fig_paths:
        figures_data = []
        for fig in fig_paths:
            with open(fig['path'], 'rb') as img_file:
                img_data = b64encode(img_file.read()).decode('utf-8')
                figures_data.append({
                    'image': img_data,
                    'figure_number': fig['figure_number'],
                    'caption': fig['caption']
                })
            os.remove(fig['path'])

        return JSONResponse({
            "status": "success",
            "figures": figures_data
        })
    return {'success': False, "message": "Figures not found"}


@app.get("/pdf/get_timeline")
async def get_timeline(req: PdfRequest,
                       file_manager: FileManager = Depends(get_file_manager)):
    pdf_id, user_id = req.pdf_id, req.user_id
    timeline_path = file_manager.get_timeline(user_id, pdf_id)

    if timeline_path:
        return FileResponse(timeline_path, media_type="application/json")
    return {'success': False, "message": "Timeline not found"}


@app.get("/pdf/get_audio")
async def get_audio(req: PdfRequest,
                    file_manager: FileManager = Depends(get_file_manager)):
    pdf_id, user_id = req.pdf_id, req.user_id
    audio_path = file_manager.get_audio(user_id, pdf_id)

    if audio_path:
        return FileResponse(audio_path, media_type="audio/mpeg")
    return {'success': False, "message": "Audio not found"}


@app.get("/pdf/get_thumbnail")
async def get_thumbnail(req: PdfRequest,
                        file_manager: FileManager = Depends(get_file_manager)):
    pdf_id, user_id = req.pdf_id, req.user_id
    thumbnail_path = file_manager.get_thumbnail(user_id, pdf_id)

    if thumbnail_path:
        return FileResponse(thumbnail_path, media_type="image/png")
    return {'success': False, "message": "Thumbnail not found"}


@app.get("/pdf/get_script")
async def get_script(req: PdfRequest,
                     file_manager: FileManager = Depends(get_file_manager)):
    pdf_id, user_id = req.pdf_id, req.user_id
    script_path = file_manager.get_script(user_id, pdf_id)

    if script_path:
        return FileResponse(script_path, media_type="application/json")
    return {'success': False, "message": "Thumbnail not found"}


@app.get("/pdf/get_table")
async def get_table(req: PdfRequest,
                    additional_uploader: AdditionalFileUploader = Depends(get_add_file_uploader)):
    pdf_id, user_id = req.pdf_id, req.user_id
    table_info = additional_uploader.search_table_file(user_id, pdf_id)

    if table_info:
        return {'success': True, "data": {"tables": table_info}}
    return {'success': False, "message": "Table information not found"}


# TODO 프론트 메인 화면에서 시작하기를 눌렀을 때 user_id의 history에서 pdf title을 전송해주는 API

# TODO history에서 해당 pdf를 눌렀을 때 pdf와 번역본, chat history를 전송해주는 API
@app.get("/pdf/get_chat_hist")
async def get_chat_hist(req: PdfRequest,
                        chat_history_manager: ChatHistoryManager = Depends(get_chat_manager)):
    chat_hist = chat_history_manager.get_chat_history(req.user_id, req.pdf_id)

    if chat_hist:
        return {"success": True, "chat_hist": chat_hist}
    return {"success": False, "message": "채팅 기록 불러오기 중 에러 발생"}

def pdf2text_recognize(pdf, key="text"):
    p2t = Pdf2Text(AI_CONFIG['layout_model_path'])

    pdf_images, lang = pdf_to_image(pdf)

    result = [p2t.recognize_only_text(page, lang) for page in pdf_images]

    del p2t
    torch.cuda.empty_cache()
    gc.collect()

    if key=="summary":
        return result
    
    sentences = [split_sentences(raw_text) for raw_text in result]

    for idx in range(1, len(sentences)):
        sentences[idx] = sentences[idx - 1][-3:] + sentences[idx]

    return sentences, lang


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
    torch.cuda.empty_cache()

    return chunked_documents
