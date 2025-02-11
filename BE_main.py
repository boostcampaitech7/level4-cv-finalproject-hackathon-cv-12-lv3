import gc
import os
import json
import torch
import subprocess
import logging

from typing import List
from typing import Dict, Any, Optional
from collections import defaultdict
from base64 import b64encode
from hashlib import sha256
from enum import Enum

from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, status, Depends, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from config.config import AI_CONFIG, API_CONFIG
from pdf2text import Pdf2Text, pdf_to_image
from utils import FileManager, MultiChatManager, PaperSummarizer
from utils import split_sentences, chunkify_to_num_token
from utils import process_query_with_reranking_compare, query_and_respond
from api import EmbeddingAPI, ChatCompletionsExecutor, SummarizationExecutor

from datebase.connection import DatabaseConnection
from datebase.operations import PaperManager, DocumentUploader, ChatHistoryManager, AdditionalFileUploader

from sentence_transformers import SentenceTransformer, CrossEncoder

logger = logging.getLogger(__name__)


def get_db_connection():
    db_connection = DatabaseConnection()
    return db_connection.connect()


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


class InfoType(str, Enum):
    all = "all"
    head = "head"


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


class MultiPdfRequest(BaseModel):
    pdf_ids: List[int]
    user_id: str = "admin"

    @classmethod
    def as_form(
        cls,
        user_id: str = Form('admin'),
        pdf_ids: str = Form(...)
    ):
        pdf_id_list = [int(id_) for id_ in pdf_ids.split(',')]
        return cls(user_id=user_id, pdf_ids=pdf_id_list)


class ChatRequest(BaseRequest):
    message: str = ""


@app.get("/", response_model=BaseResponse, response_model_exclude_unset=True)
def test():
    return {'success': True, 'message': "테스트 성공"}


@app.post("/pdf", response_model=BaseResponse, response_model_exclude_unset=True)
async def upload_pdf(file: UploadFile,
                     req: PdfRequest = Depends(PdfRequest.as_form),
                     file_manager: FileManager = Depends(get_file_manager),
                     chat_manager: ChatHistoryManager = Depends(get_chat_manager)):
    user_id = req.user_id
    try:
        print("=== Debug Info ===")
        print(f"File received: {file.filename}")
        print(f"Request data: {req}")

        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="PDF 파일만 업로드 가능합니다.")

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

        # 임시 대화 1개 생성
        chat_manager.store_chat(user_id, paper_id, 'assistant',
                                "안녕하세요 당신의 논문 공부를 도와드릴 SummarAI입니다! 무엇을 도와드릴까요?",
                                None, False, None, None, None, None)

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

    translation_result = run_translate(pdf_path)
    mono_pdf_path = translation_result['translated_pdfs']
    new_data = translation_result['translated_json']

    file_manager.update_translated_paper(mono_pdf_path, user_id, pdf_id)

    sentences = pdf2text_recognize(pdf_path)

    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    chunked_documents = chunking_embedding(sentences, new_data)

    try:
        document_mannager.upload_documents(
            chunked_documents, user_id, pdf_id)

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Document 업로드 중에 오류 발생 : {str(e)}")
    return {"success": True, "message": "대화를 할 준비가 되었습니다!! 궁금하신 점을 질문주세요!"}


@app.post("/chat-bot/message", response_model=BaseResponse, response_model_exclude_unset=True)
async def chat_message(req: ChatRequest,
                       conn=Depends(get_db_connection)):
    pdf_id, user_id, user_input = req.pdf_id, req.user_id, req.message

    fig_table_keywords = ['figure', '피규어', 'table', '테이블']
    is_figure_query = any(keyword in user_input.lower()
                          for keyword in fig_table_keywords)

    multi_chat_manager.initialize_chat("")

    model = SentenceTransformer("dragonkue/bge-m3-ko")
    reranker_model = CrossEncoder(
        "jinaai/jina-reranker-v2-base-multilingual", trust_remote_code=True)

    if is_figure_query:
        relevant_response = query_and_respond(
            query=user_input,
            conn=conn,
            model=model,
            user_id=user_id,
            paper_id=pdf_id,
            top_k=3,
            chat_manager=chat_history_manager
        )
    else:
        relevant_response = process_query_with_reranking_compare(
            query=user_input,
            conn=conn,
            model=model,
            reranker=reranker_model,
            completion_executor=completion_executor,
            user_id=user_id,
            paper_id=pdf_id,
            top_k=3,
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
            return {"success": True, "data": {"message": response['content']}}

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

    table_flag = file_manager.store_figures_and_tables(
        total_match_res, user_id, req.pdf_id,
        model, completion_executor)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    if table_flag:
        return {"success": True, "message": "Table과 Figure에 대한 처리가 완료되었습니다. 이제 해당 부분에 대한 답변도 가능합니다~!"}
    return {"success": False, "message": "Embedding 값 저장 중 오류 발생"}


@app.post("/pdf/summarize", response_model=BaseResponse, response_model_exclude_unset=True)
async def summarize_and_get_files(req: PdfRequest,
                                  file_manager: FileManager = Depends(get_file_manager)):
    pdf_id, user_id = req.pdf_id, req.user_id
    paper_summarizer = PaperSummarizer()

    paper_file = file_manager.get_paper(user_id, pdf_id)

    final_summary = paper_summarizer.generate_summary(paper_file)

    file_flag = file_manager.extract_summary_content(
        final_summary=final_summary,
        completion_executor=completion_executor,
        user_id=user_id,
        paper_id=pdf_id
    )

    if file_flag:
        return {"success": True, "message": "요약 및 오디오, 태그, 타임라인 생성이 완료되었습니다."}
    return {"success": False, "message": "Summarize 중 에러 발생"}


@app.post("/pdf/get_paper")
async def get_paper(req: PdfRequest,
                    file_manager: FileManager = Depends(get_file_manager)):
    user_id, pdf_id = req.user_id, req.pdf_id
    pdf_path = file_manager.get_paper(user_id, pdf_id)

    if pdf_path:
        return FileResponse(pdf_path, media_type="application/pdf")
    return {'success': False, "message": "Paper not found"}


@app.post("/pdf/get_translate_paper")
async def get_translate_paper(req: PdfRequest,
                              file_manager: FileManager = Depends(get_file_manager)):
    user_id, pdf_id = req.user_id, req.pdf_id
    pdf_path = file_manager.get_trans_paper(user_id, pdf_id)

    if pdf_path:
        return FileResponse(pdf_path, media_type="application/pdf")
    return {'success': False, "message": "Translate Paper not found"}


@app.post("/pdf/get_figure")
async def get_figure(req: PdfRequest,
                     file_manager: FileManager = Depends(get_file_manager)):
    pdf_id, user_id = req.pdf_id, req.user_id
    fig_paths = file_manager.get_figure(user_id, pdf_id)

    if fig_paths:
        figures_data = []
        for fig in fig_paths:
            data = {}

            with open(fig['path'], 'rb') as img_file:
                img_data = b64encode(img_file.read()).decode('utf-8')
                data['image'] = img_data

            with open(fig['caption_path'], 'rb') as img_file:
                img_data = b64encode(img_file.read()).decode('utf-8')
                data['caption_image'] = img_data

            figures_data.append({
                **data,
                'figure_number': fig['figure_number'],
                'caption_info': fig['caption_info'],
                'description': fig['description'],
            })

            os.remove(fig['path'])
            os.remove(fig['caption_path'])

        return JSONResponse({
            "status": "success",
            "figures": figures_data
        })
    return {'success': False, "message": "Figures not found"}


@app.post("/pdf/get_timeline")
async def get_timeline(req: PdfRequest,
                       file_manager: FileManager = Depends(get_file_manager)):
    pdf_id, user_id = req.pdf_id, req.user_id
    timeline_path = file_manager.get_timeline(user_id, pdf_id)

    if timeline_path:
        return FileResponse(timeline_path, media_type="application/json")
    return {'success': False, "message": "Timeline not found"}


@app.post("/pdf/get_audio")
async def get_audio(req: PdfRequest,
                    file_manager: FileManager = Depends(get_file_manager)):
    pdf_id, user_id = req.pdf_id, req.user_id
    audio_path = file_manager.get_audio(user_id, pdf_id)

    if audio_path:
        return FileResponse(audio_path, media_type="audio/mpeg")
    return {'success': False, "message": "Audio not found"}


@app.post("/pdf/get_thumbnail")
async def get_thumbnail(req: PdfRequest,
                        file_manager: FileManager = Depends(get_file_manager)):
    pdf_id, user_id = req.pdf_id, req.user_id
    thumbnail_path = file_manager.get_thumbnail(user_id, pdf_id)

    if thumbnail_path:
        return FileResponse(thumbnail_path, media_type="image/png")
    return {'success': False, "message": "Thumbnail not found"}


@app.post("/pdf/get_script")
async def get_script(req: PdfRequest,
                     file_manager: FileManager = Depends(get_file_manager)):
    pdf_id, user_id = req.pdf_id, req.user_id
    script_path = file_manager.get_script(user_id, pdf_id)

    if script_path:
        return FileResponse(script_path, media_type="application/json")
    return {'success': False, "message": "Thumbnail not found"}


@app.post("/pdf/get_table")
async def get_table(req: PdfRequest,
                    file_manager: FileManager = Depends(get_file_manager)):
    pdf_id, user_id = req.pdf_id, req.user_id
    table_paths = file_manager.get_table(user_id, pdf_id)

    if table_paths:
        table_data = []
        for table in table_paths:
            data = {}
            with open(table['path'], 'rb') as img_file:
                img_data = b64encode(img_file.read()).decode('utf-8')
                data['image'] = img_data

            with open(table['caption_path'], 'rb') as img_file:
                img_data = b64encode(img_file.read()).decode('utf-8')
                data['caption_image'] = img_data

            table_data.append({
                **data,
                'table_number': table['table_number'],
                'description': table['description']
            })

            if os.path.exists(table['path']):
                os.remove(table['path'])
            if os.path.exists(table['caption_path']):
                os.remove(table['caption_path'])

        return JSONResponse({
            "success": True,
            "data": {
                "tables": table_data
            }
        })

    return {'success': False, "message": "Table information not found"}


@app.post("/pdf/get_users_hist")
async def get_users_hist(req: PdfRequest,
                         additional_uploader: AdditionalFileUploader = Depends(get_add_file_uploader)):
    user_id = req.user_id
    hist_info = additional_uploader.search_users_hist(user_id)

    if hist_info:
        return {'success': True, "data": hist_info}
    return {'success': False, 'message': "No chat history found"}


@app.post("/pdf/get_chat_hist")
async def get_chat_hist(req: PdfRequest,
                        chat_history_manager: ChatHistoryManager = Depends(get_chat_manager)):
    user_id, pdf_id = req.user_id, req.pdf_id
    chat_hist = chat_history_manager.get_chat_history(user_id, pdf_id)

    if chat_hist:
        return {"success": True, "chat_hist": chat_hist}
    return {"success": False, "message": "채팅 기록 불러오기 중 에러 발생"}


@app.post("/pdf/get_summary")
async def get_summary(req: PdfRequest,
                      paper_manager: PaperManager = Depends(get_paper_manager)):
    user_id, pdf_id = req.user_id, req.pdf_id
    summary_info = paper_manager.get_paper_info(user_id, pdf_id)

    if summary_info:
        return {
            "success": True,
            "data": {"long_summary": summary_info['long_summary'] if summary_info else None}
        }
    return {"success": False, "message": "요약 불러오기 중 에러 발생"}


@app.post("/pdf/get_tags")
async def get_tags(req: PdfRequest,
                   additional_uploader: AdditionalFileUploader = Depends(get_add_file_uploader)):
    user_id, pdf_id = req.user_id, req.pdf_id
    tag_info = additional_uploader.search_tag_text(user_id, pdf_id)

    if tag_info:
        return {
            "success": True,
            "data": {"tag_text": tag_info}
        }
    return {"success": False, "message": "태그 정보를 찾을 수 없습니다."}


@app.post("/pdf/get_summary_pdf_id")
async def get_summary_pdf_id(req: PdfRequest,
                             paper_manager: PaperManager = Depends(get_paper_manager)):
    user_id = req.user_id
    summary_pdf_id = paper_manager.get_summary_pdf_id(user_id)

    if summary_pdf_id:
        return {
            "success": True,
            "data": {"paper_ids": summary_pdf_id}
        }
    return {"success": False, "message": "요약된 PDF ID를 찾을 수 없습니다."}


@app.post("/pdf/get_all_summary_info/{info_type}")
async def get_all_summary_info(info_type: InfoType,
                               req: MultiPdfRequest,
                               paper_manager: PaperManager = Depends(
                                   get_paper_manager),
                               file_manager: FileManager = Depends(
                                   get_file_manager),
                               additional_uploader: AdditionalFileUploader = Depends(get_add_file_uploader)):
    user_id = req.user_id
    all_pdf_data = []
    temp_files = []

    try:
        for pdf_id in req.pdf_ids:
            response_data = {"json_data": {}, "files": {}}

            # 1. Paper 기본 정보 가져오기
            paper_info = paper_manager.get_paper_info(user_id, pdf_id)
            if not paper_info:
                return {"success": False, "message": "Paper 정보를 찾을 수 없습니다[1]"}
            response_data["json_data"].update({
                "paper_info": {
                    "paper_id": paper_info['paper_id'],
                    "title": paper_info['title'],
                    "long_summary": paper_info['long_summary']
                },
                # 2. 태그 정보 가져오기
                "tags": additional_uploader.search_tag_text(user_id, pdf_id)
            })

            # 3. 썸네일 가져오기
            thumbnail_path = file_manager.get_thumbnail(user_id, pdf_id)
            if thumbnail_path:
                temp_files.append(thumbnail_path)
                with open(thumbnail_path, 'rb') as f:
                    response_data["files"]["thumbnail"] = b64encode(
                        f.read()).decode('utf-8')
                os.remove(thumbnail_path)

            # 상세 보기 페이지를 들어가는 경우 (all)
            if info_type == 'all':
                # 4. Figure 정보 가져오기
                fig_paths = file_manager.get_figure(user_id, pdf_id)
                figures_data = []
                if fig_paths:
                    for fig in fig_paths:
                        with open(fig['path'], 'rb') as img_file:
                            img_data = b64encode(
                                img_file.read()).decode('utf-8')
                            figures_data.append({
                                'image': img_data,
                                'figure_number': fig['figure_number'],
                                'caption_info': fig['caption_info'],
                                'description': fig['description']
                            })
                        os.remove(fig['path'])
                    response_data["json_data"]["figures"] = figures_data

                # 5. 타임라인 정보 가져오기
                timeline_path = None
                timeline_path = file_manager.get_timeline(user_id, pdf_id)

                if timeline_path:
                    temp_files.append(timeline_path)
                    with open(timeline_path, 'rb') as f:
                        response_data["files"]["timeline"] = f.read().decode(
                            'utf-8')
                    os.remove(timeline_path)

                # 6. Table 정보 가져오기
                table_info = additional_uploader.search_table_file(
                    user_id, pdf_id)
                response_data["json_data"]["tables"] = table_info

            all_pdf_data.append(response_data)

        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.error(f"임시 파일 삭제 실패: {str(e)}")

        return JSONResponse({"status": "success", "content": all_pdf_data})
    except FileNotFoundError as e:
        return JSONResponse({
            "status": "error",
            "content": f"파일을 찾을 수 없습니다: {str(e)}"
        })
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse({
            "status": "error",
            "content": f"정보 조회 중 오류가 발생했습니다: {str(e)}"
        })


def pdf2text_recognize(pdf):
    p2t = Pdf2Text(AI_CONFIG['layout_model_path'])

    pdf_images, lang = pdf_to_image(pdf)

    result = [p2t.recognize_only_text(page, lang) for page in pdf_images]

    del p2t
    torch.cuda.empty_cache()
    gc.collect()

    sentences = [split_sentences(raw_text) for raw_text in result]

    for idx in range(1, len(sentences)):
        sentences[idx] = sentences[idx - 1][-3:] + sentences[idx]

    return sentences


def chunking_embedding(sentences, new_data, size=256):
    total_chunks = [chunkify_to_num_token(
        sentence, size) for sentence in sentences]

    chunked_documents = [{"page": idx + 1, "chunk": chunk}
                         for idx, chunks in enumerate(total_chunks) for chunk in chunks]

    for idx in range(len(sentences)):
        str_i = str(idx)
        if str_i in new_data:
            new_text = new_data[str_i]
            new_chunks = chunkify_to_num_token(new_text, size)
            chunked_documents.extend(
                [{"page": idx + 1, "chunk": chunk} for chunk in new_chunks])

    model = SentenceTransformer("dragonkue/bge-m3-ko")

    chunked_documents = list(map(lambda x: {
                             **x, 'embedding': model.encode(x['chunk']).tolist()}, chunked_documents))

    del model
    torch.cuda.empty_cache()

    return chunked_documents


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
