from storage import ObjectStorageManager
from datebase import PaperManager, AdditionalFileUploader, DocumentUploader
from typing import Dict
from dotenv import load_dotenv
# from .model_manager import model_manager
from .summary_short import abstractive_summarization
from .timeline import extract_keywords, timeline_str, abstractive_timeline
from .script import write_full_script
from .audiobook_test import script_to_speech
from .vlm import conversation_with_images, translate_clova
import os
import json
import random
import tempfile
from PIL import Image
import traceback
load_dotenv()

# TODO 이거 임시 파일 만들고 삭제하는 로직은 나중에 추가하기
# save_files.py


class FileManager:
    def __init__(self, conn):
        self.conn = conn
        self.storage_manager = ObjectStorageManager()
        self.paper_manager = PaperManager(conn)
        self.additional_manager = AdditionalFileUploader(conn)
        self.document_manager = DocumentUploader(conn)

    def store_paper(self, file_input, paper_info: Dict, user_id) -> str:
        if isinstance(file_input, bytes):
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix='.pdf', mode='wb')
            temp_file.write(file_input)
            temp_file.close()
            file_path = temp_file.name
        # 문자열(파일 경로)인 경우
        elif isinstance(file_input, str):
            file_path = file_input
        else:
            raise TypeError("file_input must be either bytes or str")

        try:
            # Object Storage에 PDF 저장
            storage_info = self.storage_manager.upload_pdf(
                file_path=file_path,
                bucket_name=os.getenv('NCP_BUCKET_NAME')
            )

            if not storage_info:
                raise Exception("PDF 저장 실패")

            if isinstance(file_input, bytes):
                os.unlink(file_path)

            # DB에 논문 정보 저장
            paper_id = self.paper_manager.store_paper_info(
                user_id=user_id,
                title=paper_info['title'],
                author=paper_info['authors'],
                pdf_file_path=storage_info['path']
            )

            return paper_id
        except Exception as e:
            if isinstance(file_input, bytes) and 'file_path' in locals():
                os.unlink(file_path)
            raise e

    def update_translated_paper(self, file_path, user_id, paper_id):
        """번역 PDF 저장"""
        storage_info = self.storage_manager.upload_pdf(
            file_path=file_path,
            bucket_name=os.getenv('NCP_BUCKET_NAME')
        )

        if not storage_info:
            raise Exception("번역 PDF 저장 실패")

        self.paper_manager.update_tran_pdf_file(
            user_id=user_id,
            paper_id=paper_id,
            tran_pdf_file_path=storage_info['path']
        )

    def store_figures_and_tables(self, match_res, user_id, paper_id,
                                 model, completion_executor):
        """Figure랑 Table 저장 후 Vector DB 저장까지"""
        try:
            chunked_documents = []
            # figure 처리
            if 'figure' in match_res:
                for figure in match_res['figure']:
                    path_dict = {
                        "figure_path": f"temp_figure_{paper_id}_{figure['caption_number']}.png",
                        "caption_path": f"temp_caption_{paper_id}_figure_{figure['caption_number']}.png",
                    }

                    figure['obj_image'].save(path_dict['figure_path'])
                    figure['caption_image'].save(path_dict['caption_path'])

                    figure_storage_info = self.storage_manager.upload_figure(
                        file_path=path_dict['figure_path'],
                        bucket_name=os.getenv('NCP_BUCKET_NAME')
                    )

                    caption_storage_info = self.storage_manager.upload_figure(
                        file_path=path_dict['caption_path'],
                        bucket_name=os.getenv('NCP_BUCKET_NAME')
                    )

                    # DeepSeek 처리
                    image = Image.open(path_dict['figure_path'])
                    caption = "This is Transformer Acheitecture img"
                    response = conversation_with_images("deepseek-ai/deepseek-vl-7b-chat",
                                                        [image],
                                                        image_description=figure['caption_text']
                                                        if figure['caption_text'] else caption)
                    trans_response = translate_clova(
                        response, completion_executor)

                    # 컬럼 추가
                    self.additional_manager.insert_figure_file(
                        user_id=user_id,
                        paper_id=paper_id,
                        storage_path=figure_storage_info['path'],
                        caption_number=figure['caption_number'],
                        caption_info=figure['caption_text'],
                        description=trans_response,
                        caption_path=caption_storage_info['path']
                    )

                    # embedding
                    figure_doc = {
                        "page": figure['caption_number'],
                        "chunk": figure['caption_text'],
                        "type": "table"
                    }
                    if figure_doc["chunk"] and isinstance(figure_doc["chunk"], str):
                        figure_doc["embedding"] = model.encode(
                            figure_doc["chunk"]).tolist()
                    chunked_documents.append(figure_doc)

                    for path in path_dict:
                        os.remove(path_dict[path])

            # table 처리
            if 'table' in match_res:
                for table in match_res['table']:
                    path_dict = {
                        "table_path": f"temp_table_{paper_id}_{table['caption_number']}.png",
                        "caption_path": f"temp_caption_{paper_id}_table_{table['caption_number']}.png",
                    }

                    table['obj_image'].save(path_dict['table_path'])
                    table['caption_image'].save(path_dict['caption_path'])

                    table_storage_info = self.storage_manager.upload_table(
                        file_path=path_dict['table_path'],
                        bucket_name=os.getenv('NCP_BUCKET_NAME')
                    )

                    caption_storage_info = self.storage_manager.upload_caption(
                        file_path=path_dict['caption_path'],
                        bucket_name=os.getenv('NCP_BUCKET_NAME')
                    )

                    self.additional_manager.insert_table_file(
                        user_id=user_id,
                        paper_id=paper_id,
                        table_obj=table['obj'],
                        caption_number=table['caption_number'],
                        description=table['caption_text'],
                        storage_path=table_storage_info['path'],
                        caption_path=caption_storage_info['path']
                    )

                    table_doc = {
                        "page": table['caption_number'],
                        "chunk": table['caption_text'],
                        "type": "table"
                    }
                    if table_doc["chunk"] and isinstance(table_doc["chunk"], str):
                        table_doc["embedding"] = model.encode(
                            table_doc["chunk"]).tolist()
                    chunked_documents.append(table_doc)

                    for path in path_dict:
                        os.remove(path_dict[path])

            if chunked_documents:
                self.document_manager.upload_documents(
                    chunked_documents, user_id, paper_id)
            return True
        except Exception as e:
            print(f"Figure/Table 저장 중 에러 발생: {str(e)}")
            return False

    def extract_summary_content(self, final_summary, completion_executor,
                                user_id, paper_id):
        """요약 ~ 오디오 생성까지 일괄로 진행하는 코드"""
        try:
            # 1. 요약 생성 및 저장
            result = abstractive_summarization(
                final_summary, completion_executor)
            res1, res2 = result.split("\n", 1)

            # 요약 정보 업데이트
            self.paper_manager.update_summary(
                user_id,
                paper_id,
                short_summary=res1,
                long_summary=result
            )

            # 2. 키워드/태그 저장
            query_list = extract_keywords(result)
            for tag in query_list:
                self.additional_manager.insert_tag_file(
                    user_id=user_id,
                    paper_id=paper_id,
                    tag_text=tag
                )

            # 3. 타임라인 저장
            # timeline_data = timeline_str(query_list)
            timeline_data = abstractive_timeline(query_list)
            temp_timeline_path = f"temp_timeline_{paper_id}.json"
            with open(temp_timeline_path, 'w', encoding='utf-8') as f:
                json.dump(timeline_data, f, ensure_ascii=False, indent=4)

            timeline_storage_info = self.storage_manager.upload_timeline(
                file_path=temp_timeline_path,
                bucket_name=os.getenv('NCP_BUCKET_NAME')
            )

            self.additional_manager.insert_timeline_file(
                user_id=user_id,
                paper_id=paper_id,
                storage_path=timeline_storage_info['path'],
                timeline_name=f"Timeline_{paper_id}",
                description=f"{paper_id} Timeline"
            )

            # 4. 오디오북 생성/저장
            conversations = write_full_script(res1)
            final_audio = script_to_speech(conversations)

            temp_script_path = f"temp_script_{paper_id}.json"
            with open(temp_script_path, 'w', encoding='utf-8') as f:
                json.dump(conversations, f, ensure_ascii=False)

            temp_audio_path = f"temp_audio_{paper_id}.mp3"
            final_audio.export(temp_audio_path, format="mp3")

            temp_thumbnail_path = self.make_thumbnail(user_id, paper_id)

            if temp_thumbnail_path:
                thumbnail_storage_info = self.storage_manager.upload_thumbnail(
                    file_path=temp_thumbnail_path,
                    bucket_name=os.getenv('NCP_BUCKET_NAME')
                )

            conversation_storage_info = self.storage_manager.upload_script(
                file_path=temp_script_path,
                bucket_name=os.getenv('NCP_BUCKET_NAME')
            )

            audio_storage_info = self.storage_manager.upload_audio(
                file_path=temp_audio_path,
                bucket_name=os.getenv('NCP_BUCKET_NAME')
            )

            self.additional_manager.insert_audio(
                user_id=user_id,
                paper_id=paper_id,
                audio_file_path=audio_storage_info['path'],
                thumbnail_path=thumbnail_storage_info['path'] if temp_thumbnail_path else None,
                audio_title=f"Audio Summary of Paper {paper_id}",
                script=conversation_storage_info['path']
            )

            os.remove(temp_audio_path)
            if temp_thumbnail_path:
                os.remove(temp_thumbnail_path)
            os.remove(temp_timeline_path)
            os.remove(temp_script_path)

            return True
        except Exception as e:
            print(f"콘텐츠 처리 및 저장 중 에러 발생: {str(e)}")
            print("Traceback:", traceback.format_exc())
            return False

    # TODO Paper부터 다른 가져오는 기능 임시 파일 삭제하는거 만들기
    def get_paper(self, user_id: str, paper_id: int) -> str:
        """Storage에서 PDF 파일 가져오는 메서드"""
        try:
            paper_info = self.paper_manager.get_paper_info(user_id, paper_id)

            if not paper_info:
                raise Exception("Paper not found")

            temp_path = f"temp_paper_{paper_id}.pdf"
            if not os.path.exists(temp_path):
                downloaded = self.storage_manager.download_file(
                    file_url=paper_info['pdf_file_path'],
                    local_path=temp_path,
                    bucket_name=os.getenv('NCP_BUCKET_NAME')
                )

                if not downloaded:
                    raise Exception("Paper 다운로드 실패")

            return temp_path
        except Exception as e:
            print(f"Paper 가져오기 실패: {str(e)}")
            return None

    def get_trans_paper(self, user_id: str, paper_id: int) -> str:
        """Storage에서 PDF 파일 가져오는 메서드"""
        try:
            paper_info = self.paper_manager.get_paper_info(user_id, paper_id)

            if not paper_info:
                raise Exception("Trans Paper not found")

            temp_path = f"temp_trans_paper_{paper_id}.pdf"
            if not os.path.exists(temp_path):
                downloaded = self.storage_manager.download_file(
                    file_url=paper_info['tran_pdf_file_path'],
                    local_path=temp_path,
                    bucket_name=os.getenv('NCP_BUCKET_NAME')
                )

                if not downloaded:
                    raise Exception("Trans Paper 다운로드 실패")

            return temp_path
        except Exception as e:
            print(f"Trans Paper 가져오기 실패: {str(e)}")
            return None

    def get_figure(self, user_id: str, paper_id: str):
        """ Figure 가져오기"""
        try:
            figure_info = self.additional_manager.search_figure_file(
                user_id, paper_id)
            if not figure_info:
                raise Exception("Figure not found")

            figure_paths = []
            for figure in figure_info:
                path_dict = {
                    "figure_path": f"temp_figure_{paper_id}_{figure['caption_number']}.png",
                    "caption_path": f"temp_caption_{paper_id}_figure_{figure['caption_number']}.png"
                }

                figure_downloaded = self.storage_manager.download_file(
                    file_url=figure['storage_path'],
                    local_path=path_dict['figure_path'],
                    bucket_name=os.getenv('NCP_BUCKET_NAME')
                )

                caption_downloaded = self.storage_manager.download_file(
                    file_url=figure['caption_path'],
                    local_path=path_dict['caption_path'],
                    bucket_name=os.getenv('NCP_BUCKET_NAME')
                )

                if not figure_downloaded:
                    print(f"Figure {figure['caption_number']} 다운로드 실패")
                    continue

                if not caption_downloaded:
                    print(
                        f"Figure {figure['caption_number']}번의 Caption 다운로드 실패")
                    continue

                figure_paths.append({
                    'path': path_dict['figure_path'],
                    'figure_number': figure['caption_number'],
                    'caption_info': figure.get('caption_info', ''),
                    'description': figure.get('description', ''),
                    'caption_path': path_dict['caption_path']
                })

            return figure_paths
        except Exception as e:
            print(f"Figure 가져오기 실패: {str(e)}")
            return None

    def get_timeline(self, user_id: str, paper_id: str):
        """ Timeline 가져오기 """
        try:
            timeline_info = self.additional_manager.search_timeline_file(
                user_id, paper_id)
            if not timeline_info:
                raise Exception("Timeline not found")

            temp_path = f"temp_timeline_{paper_id}.json"
            downloaded = self.storage_manager.download_file(
                file_url=timeline_info['storage_path'],
                local_path=temp_path,
                bucket_name=os.getenv('NCP_BUCKET_NAME')
            )

            if not downloaded:
                raise Exception("Timeline 다운로드 실패")

            return temp_path
        except Exception as e:
            print(f"Timeline 가져오기 실패: {str(e)}")
            return None

    def get_audio(self, user_id: str, paper_id: str):
        """ audio 가져오기 """
        try:
            audio_info = self.additional_manager.search_audio_file(
                user_id, paper_id)
            if not audio_info:
                raise Exception("Audio not found")

            temp_path = f"temp_audio_{paper_id}.mp3"
            downloaded = self.storage_manager.download_file(
                file_url=audio_info['audio_file_path'],
                local_path=temp_path,
                bucket_name=os.getenv('NCP_BUCKET_NAME')
            )

            if not downloaded:
                raise Exception("Audio 다운로드 실패")

            return temp_path
        except Exception as e:
            print(f"audio 가져오기 실패: {str(e)}")
            return None

    def get_thumbnail(self, user_id: str, paper_id: str):
        """ Thumbnail 가져오기 """
        try:
            thumbnail_info = self.additional_manager.search_audio_file(
                user_id, paper_id)
            if not thumbnail_info:
                raise Exception("Thumbnail not found")

            temp_path = f"temp_thumbnail_{paper_id}.png"
            downloaded = self.storage_manager.download_file(
                file_url=thumbnail_info['thumbnail_path'],
                local_path=temp_path,
                bucket_name=os.getenv('NCP_BUCKET_NAME')
            )

            if not downloaded:
                raise Exception("Thumbnail 다운로드 실패")

            return temp_path
        except Exception as e:
            print(f"Thumbnail 가져오기 실패: {str(e)}")
            return None

    def get_script(self, user_id: str, paper_id: str):
        """ Script 가져오기 """
        try:
            script_info = self.additional_manager.search_audio_file(
                user_id, paper_id)
            if not script_info:
                raise Exception("Script not found")

            temp_path = f"temp_script_{paper_id}.json"
            downloaded = self.storage_manager.download_file(
                file_url=script_info['script'],
                local_path=temp_path,
                bucket_name=os.getenv('NCP_BUCKET_NAME')
            )

            if not downloaded:
                raise Exception("Script 다운로드 실패")

            return temp_path
        except Exception as e:
            print(f"Script 가져오기 실패: {str(e)}")
            return None

    def make_thumbnail(self, user_id: str, paper_id: str) -> str:
        """ Thumbnail 추출하기 """
        try:
            figure_info = self.additional_manager.search_figure_file(
                user_id, paper_id)

            if figure_info and len(figure_info) > 0:
                selected_figure = random.choice(figure_info)
                temp_thumbnail_path = f"temp_thumbnail_{paper_id}_{selected_figure['caption_number']}.png"

                if self.storage_manager.download_file(
                    file_url=selected_figure['storage_path'],
                    local_path=temp_thumbnail_path,
                    bucket_name=os.getenv('NCP_BUCKET_NAME')
                ):
                    return temp_thumbnail_path

            # 기본 썸네일 다운로드 시도
            default_thumbnail_path = "my_friend.png"
            if self.storage_manager.download_file(
                file_url="my_friend.png",
                local_path=default_thumbnail_path,
                bucket_name=os.getenv('NCP_BUCKET_NAME')
            ):
                return default_thumbnail_path

            return None  # 다운로드 실패시 None 반환
        except Exception as e:
            print(f"Thumbnail 생성 실패: {str(e)}")
            return None
