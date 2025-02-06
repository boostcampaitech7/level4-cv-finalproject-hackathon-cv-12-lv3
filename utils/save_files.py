from storage import ObjectStorageManager
from datebase import PaperManager, AdditionalFileUploader, DocumentUploader
from typing import Dict
from dotenv import load_dotenv
from .model_manager import model_manager
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
        self.document_manager = DocumentUploader(conn)

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
            pdf_file_path=storage_info['path']
        )
        
        return paper_id
    
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

    def store_figures_and_tables(self, match_res, user_id, paper_id):
        """Figure랑 Table 저장 후 Vector DB 저장까지"""
        try:
            model = model_manager.sentence_transformer
            chunked_documents = []
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
                            storage_path=storage_info['path'],
                            caption_number=figure['caption_number'],
                            description=figure['caption_text']
                        )

                        # [ ] 해당 figure를 deepseek로 보내고, deepseek에서 반환받은 값을 vector로 저장하는 로직

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

                        # [o] 해당 table을 vector로 저장하는 로직 필요
                        table_doc = {
                            "page": table['caption_number'],
                            "chunk": table['caption_text'],
                            "type": "table"
                        }
                        table_doc["embedding"] = model.encode(table_doc["chunk"]).tolist()
                        chunked_documents.append(table_doc)

                if chunked_documents:
                    self.document_manager.upload_documents(chunked_documents, user_id, paper_id)

            return True
        except Exception as e:
            print(f"Figure/Table 저장 중 에러 발생: {str(e)}")
            return False
        
    def extract_summary_content(self, final_summary, completion_executor,
                                user_id, paper_id):
        """요약 ~ 오디오 생성까지 일괄로 진행하는 코드"""
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
                audio_file_path=audio_storage_info['path'],
                thumbnail_path=None,
                audio_title=f"Audio Summary of Paper {paper_id}",
                script=conversation_storage_info['path']
            )

            os.remove(temp_audio_path)
            # os.remove(temp_thumbnail_path)
            os.remove(temp_timeline_path)
            os.remove(temp_script_path)

            return True
        except Exception as e:
            print(f"콘텐츠 처리 및 저장 중 에러 발생: {str(e)}")
            return False
        
    def get_paper(self, user_id: str, paper_id: int) -> str:
        """Storage에서 PDF 파일 가져오는 메서드"""
        try:
            paper_info = self.paper_manager.get_paper_info(user_id, paper_id)

            if not paper_info:
                raise Exception("Paper not found")
            
            temp_path = f"temp_paper_{paper_id}.pdf"
            downloaded = self.storage_manager.download_file(
                file_url=paper_info['pdf_file_path'],
                local_path=temp_path,
                bucket_name=os.getenv('NCP_BUCKET_NAME')
            )

            if not downloaded:
                raise Exception("Figure 다운로드 실패")
            
            return temp_path
        except Exception as e:
            print(f"Figure 가져오기 실패: {str(e)}")
            return None
        
    def get_figure(self, user_id: str, paper_id: str):
        """ Figure 가져오기"""
        try:
            figure_info = self.additional_manager.search_figure_file(user_id, paper_id)
            if not figure_info:
                raise Exception("Figure not found")
            
            figure_paths = []
            for figure in figure_info:
                temp_path = f"temp_figure_{paper_id}_{figure['figure_number']}.png"
                downloaded = self.storage_manager.download_file(
                    file_url=figure['storage_path'],
                    local_path=temp_path,
                    bucket_name=os.getenv('NCP_BUCKET_NAME')
                )

                if not downloaded:
                    print(f"Figure {figure['figure_number']} 다운로드 실패")
                    continue

                figure_paths.append({
                    'path': temp_path,
                    'figure_number': figure['figure_number'],
                    'caption': figure.get('description', '')
                })
            
            return temp_path
        except Exception as e:
            print(f"Figure 가져오기 실패: {str(e)}")
            return None
        
    def get_timeline(self, user_id: str, paper_id: str):
        """ Timeline 가져오기 """
        try:
            timeline_info = self.additional_manager.search_timeline_file(user_id, paper_id)
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
            audio_info = self.additional_manager.search_audio_file(user_id, paper_id)
            if not audio_info:
                raise Exception("Audio not found")
            
            temp_path = f"temp_audio_{paper_id}.json"
            downloaded = self.storage_manager.download_file(
                file_url=audio_info['storage_path'],
                local_path=temp_path,
                bucket_name=os.getenv('NCP_BUCKET_NAME')
            )

            if not downloaded:
                raise Exception("Audio 다운로드 실패")
            
            return temp_path
        except Exception as e:
            print(f"audio 가져오기 실패: {str(e)}")
            return None
        
    def get_script(self, user_id: str, paper_id: str):
        """ Script 가져오기 """
        try:
            script_info = self.additional_manager.search_audio_file(user_id, paper_id)
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