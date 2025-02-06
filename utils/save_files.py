from storage import ObjectStorageManager
from datebase import PaperManager, AdditionalFileUploader
from typing import Dict
from dotenv import load_dotenv
from .summary_short import abstractive_summarization
from .timeline import timeline, extract_keywords
from .script import write_full_script
from .audiobook_test import script_to_speech
import os
import json

load_dotenv()

# save_files.py
class FileManager:
    def __init__(self, conn):
        self.conn = conn
        self.storage_manager = ObjectStorageManager()
        self.paper_manager = PaperManager(conn)
        self.additional_manager= AdditionalFileUploader(conn)

    def store_paper(self, file_path: str, paper_info: Dict, user_id) -> str:
        # Object Storage에 PDF 저장
        storage_info = self.storage_manager.upload_pdf(
            file_path=file_path,
            bucket_name=os.getenv('NCP_BUCKET_NAME')
        )
        
        if not storage_info:
            raise Exception("PDF 저장 실패")
        
        # DB에 논문 정보 저장
        paper_id = self.paper_manager.store_paper_info(
            user_id=user_id,
            title=paper_info['title'],
            author=paper_info['authors'],
            pdf_file_path=storage_info['url']
        )
        
        return paper_id
    
    def update_translated_paper(self, file_path, user_id, paper_id):
        storage_info = self.storage_manager.upload_pdf(
            file_path=file_path,
            bucket_name=os.getenv('NCP_BUCKET_NAME')
        )

        if not storage_info:
            raise Exception("번역 PDF 저장 실패")
        
        self.paper_manager.update_tran_pdf_file(
            user_id=user_id,
            paper_id=paper_id,
            tran_pdf_file_path=storage_info['url']
        )

    def store_figures_and_tables(self, match_res, user_id, paper_id):
        try:
            for match_res in match_res:  # 리스트를 순회
                if 'figure' in match_res:
                    for figure in match_res['figure']:
                        temp_path = f"temp_figure_{paper_id}_{figure['caption_number']}.png"
                        figure['obj'].save(temp_path)

                        storage_info = self.storage_manager.upload_figure(
                            file_path=temp_path,
                            bucket_name=os.getenv('NCP_BUCKET_NAME')
                        )

                        self.additional_manager.insert_figure_file(
                            user_id=user_id,
                            paper_id=paper_id,
                            storage_path=storage_info['url'],
                            caption_number=figure['caption_number'],
                            description=figure['caption_text']
                        )

                        os.remove(temp_path)

                if 'table' in match_res:
                    for table in match_res['table']:
                        self.additional_manager.insert_table_file(
                            user_id=user_id,
                            paper_id=paper_id,
                            table_obj=table['obj'],
                            caption_number=table['caption_number'],
                            description=table['caption_text']
                        )

            return True
        except Exception as e:
            print(f"Figure/Table 저장 중 에러 발생: {str(e)}")
            return False
        
    def extract_summary_content(self, final_summary, completion_executor,
                                user_id, paper_id):
        try:
            # 1. 요약 생성 및 저장
            result = abstractive_summarization(final_summary, completion_executor)
            res1, res2 = result.split("\n", 1)

            # 요약 정보 업데이트
            self.paper_manager.update_summary(
                user_id,
                paper_id,
                short_summary=res1,
                result=result
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
            timeline_data = timeline(query_list)
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
                storage_path=timeline_storage_info['url'],
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
            # temp_thumbnail_path = f"temp_thumbnail_{paper_id}.png"
            final_audio.export(temp_audio_path, format="mp3")

            conversation_storage_info = self.storage_manager.upload_script(
                file_path=temp_script_path,
                bucket_name=os.getenv('NCP_BUCKET_NAME')
            )

            audio_storage_info = self.storage_manager.upload_audio(
                file_path=temp_audio_path,
                bucket_name=os.getenv('NCP_BUCKET_NAME')
            )

            # thumbnal_storage_info = self.storage_manager.upload_thumbnail(
            #     file_path=temp_thumbnail_path,
            #     bucket_name=os.getenv('NCP_BUCKET_NAME')
            # )

            self.additional_manager.insert_audio(
                user_id=user_id,
                paper_id=paper_id,
                audio_file_path=audio_storage_info['url'],
                thumbnail_path=None,
                audio_title=f"Audio Summary of Paper {paper_id}",
                script=conversation_storage_info['url']
            )

            os.remove(temp_audio_path)
            # os.remove(temp_thumbnail_path)
            os.remove(temp_timeline_path)
            os.remove(temp_script_path)

            return True
        except Exception as e:
            print(f"콘텐츠 처리 및 저장 중 에러 발생: {str(e)}")
            return False