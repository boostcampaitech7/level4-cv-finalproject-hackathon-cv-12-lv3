from tqdm import tqdm
from uuid import uuid4
from datetime import datetime
from sentence_transformers import SentenceTransformer
from hashlib import sha256
from passlib.context import CryptContext
from datetime import datetime
import psycopg2

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class BaseDBHandler:
    def __init__(self, connection):
        self.conn = connection

    def execute_query(self, query, params=None):
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, params)
                self.conn.commit()
                try:
                    return cur.fetchall()
                except psycopg2.ProgrammingError:
                    return None
        except Exception as e:
            print(f"쿼리 {query} 실행 중 에러 발생: {str(e)}")
            self.conn.rollback()
            return
        
    def execute_query_one(self, query, params=None):
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, params)
                self.conn.commit()
                try:
                    return cur.fetchone()
                except psycopg2.ProgrammingError:
                    return None
        except Exception as e:
            print(f"쿼리 {query} 실행 중 에러 발생: {str(e)}")
            self.conn.rollback()
            return

class SessionManager:
    def __init__(self, connection):
        self.conn = connection

    def create_session(self):
        try:
            cur = self.conn.cursor()
            session_id = str(uuid4())

            cur.execute("""
                INSERT INTO public.sessions (session_id, created_at, is_active)
                VALUES (%s, %s, %s)
                RETURNING session_id
            """, (session_id, datetime.now(), True))

            self.conn.commit()
            return session_id
        except Exception as e:
            self.conn.rollback()
            print(f"세션 등록 중 에러 발생: {str(e)}")
            raise
        finally:
            cur.close()

class UserManager(BaseDBHandler):
    def __init__(self, connection):
        self.conn = connection

    def _hash_password(self, password: str) -> str:
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, 
                        hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)

    def create_user(self, user_id: str, user_pw: str,
                     username: str, birth:datetime):
        hashed_password = self._hash_password(user_pw)

        query = """
            INSERT INTO public.user_info
            (user_id, user_pw, username, birth)
            VALUES (%s, %s, %s, %s)
        """
        try:
            self.execute_query(query, (user_id, hashed_password, username, birth))
            return True
        except Exception as e:
            print(f"사용자 생성 중 에러 발생: {str(e)}")
            return False

    def get_user_info(self, user_id, user_pw):
        query = """
            SELECT user_id, user_pw, username, birth
            FROM public.user_info
            WHERE user_id = %s AND user_pw = %s
        """

        user = self.execute_query_one(query, (user_id,))

        if user is None:
            return None
        
        if self.verify_password(user_pw, user[1]):
            return {
                'user_id': user[0],
                'username': user[2],
                'birth': user[3]
            }
        return None
    
    def user_exists(self, user_id: str) -> bool:
        query = "SELECT 1 FROM public.user_info WHERE user_id = %s"

        result = self.execute_query_one(query, (user_id,))
        return result is not None

class PaperManager(BaseDBHandler):
    def __init__(self, connection):
        self.conn = connection

    def store_paper_info(self, user_id: str, title: str, author: str = None, 
                         pdf_file_path: str = None):
        query = """
            INSERT INTO public.papers
            (user_id, title, author, pdf_file_path)
            VALUES (%s, %s, %s, %s)
            RETURNING paper_id
        """
        return self.execute_query_one(query, (user_id, title, author, pdf_file_path))[0]
    
    def update_tran_pdf_file(self, user_id: str, paper_id: int,
                              tran_pdf_file_path: str):
        query = """
            UPDATE public.papers
            SET tran_pdf_file_path = %s
            WHERE user_id = %s
            AND paper_id = %s
        """
        self.execute_query(query, (tran_pdf_file_path, user_id, paper_id))

    def update_summary(self, user_id: str, paper_id: int,
                       short_summary: str, long_summary: str):
        query = """
            UPDATE public.papers
            SET short_summary = %s,
                long_summary = %s
            WHERE user_id = %s
            AND paper_id = %s
        """
        self.execute_query(query, (short_summary, long_summary, user_id, paper_id))
    
    def get_paper_info(self, user_id: str, paper_id: int):
        query = """
            SELECT 
                paper_id, user_id, title, author, uploaded_at,
                pdf_file_path, tran_pdf_file_path, summary
            FROM public.papers
            WHERE user_id = %s
            AND paper_id = %s
        """
        result = self.execute_query_one(query, (user_id, paper_id))
        
        if result:
            return {
                'paper_id': result[0],
                'user_id': result[1],
                'title': result[2],
                'author': result[3],
                'uploaded_at': result[4],
                'pdf_file_path': result[5],
                'tran_pdf_file_path': result[6],
                'summary': result[7]
            }
        else:
            return None

class DocumentUploader:
    def __init__(self, connection):
        self.conn = connection
    
    def upload_documents(self, chunked_documents, user_id, paper_id):
        try:
            cur = self.conn.cursor()
            count = 0
            self.conn.rollback()  # 시작할 때 클린 슬레이트로 시작

            for doc in tqdm(chunked_documents):
                vector_str = f"[{','.join(map(str, doc['embedding']))}]"
                cur.execute("""
                    INSERT INTO public.document
                    (user_id, page, paper_id, content, embedding)
                    VALUES (%s, %s, %s, %s, %s::cdb_admin.vector)
                """, (user_id, doc["page"], paper_id, doc["chunk"], vector_str))
                count += 1

            self.conn.commit()
            print(f"데이터 업로드 완료! 총 {count}개의 문서가 업로드 되었습니다.")
        except Exception as e:
            self.conn.rollback()
            print(f"업로드 중 에러 발생: {str(e)}")
            raise
        finally:
            cur.close()

class ChatHistoryManager:
    def __init__(self, connection, embedding_api, chat_api):
        self.conn = connection
        self.embedding_api = embedding_api
        self.chat_api = chat_api
        self.cache = {}
        self.model = SentenceTransformer("dragonkue/bge-m3-ko")

    def add_to_cache(self, user_id, paper_id, question, context, answer):
        """ 새로운 응답 캐시 저장 """
        cache_key = f"{user_id}:{paper_id}:{question}:{sha256(context['content'].encode()).hexdigest()}"
        self.cache[cache_key] = answer

    def store_chat(self, user_id, paper_id, role, message, parent_id=None,
                   is_summary=False, summary_for_chat_id=None, context_docs=None,
                   embedding=None, chat_type=None):
        try:
            cur = self.conn.cursor()

            cur.execute("""
                INSERT INTO public.chat_hist
                (user_id, paper_id, role, message, parent_message_id,
                 is_summary, summary_for_chat_id, context_docs,
                 embedding, chat_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING chat_id
            """, (user_id, paper_id, role, message, parent_id,
                  is_summary, summary_for_chat_id, context_docs,
                  embedding, chat_type))
            
            chat_id = cur.fetchone()[0]
            self.conn.commit()
            return chat_id
        except Exception as e:
            self.conn.rollback()
            print(f"채팅 로그 저장 중 에러 발생: {str(e)}")
            raise
        finally:
            cur.close()

    def store_conversation(self, user_id, paper_id, user_message, llm_response,
                           parent_id=None, context_docs=None, embedding=None,
                           chat_type=None):
        """ 대화 쌍 (사용자 메시지 + AI 응답) 저장 """
        try:
            user_chat_id = self.store_chat(
                user_id=user_id,
                paper_id=paper_id,
                role='user',
                message=user_message,
                parent_id=parent_id,
                context_docs=context_docs,
                embedding=embedding,
                chat_type=chat_type
            )

            assistant_chat_id = self.store_chat(
                user_id=user_id,
                paper_id=paper_id,
                role='assistant',
                message=llm_response,
                parent_id=user_chat_id,
                context_docs=context_docs,
                embedding=embedding,
                chat_type=chat_type
            )

            self.conn.commit()

            return assistant_chat_id
        except Exception as e:
            self.conn.rollback()
            print(f"대화 저장 중 에러 발생: {str(e)}")
            raise

    def store_summary(self, user_id, paper_id, summary, summarized_chat_ids):
        """ 요약본 저장 """
        try:
            summary_id =  self.store_chat(
                user_id=user_id,
                paper_id=paper_id,
                role='summary',
                message=summary,
                is_summary=True,
                summary_for_chat_id=summarized_chat_ids[-1]
            )

            self.conn.commit()

            return summary_id
        except Exception as e:
            self.conn.rollback()
            print(f"요약본 저장 중 에러 발생: {str(e)}")
            raise

    def get_chat_history(self, user_id, paper_id, limit=None):
        """ 세션 히스토리 조회 """
        try:
            cur = self.conn.cursor()

            query = """
                SELECT chat_id, role, message, created_at, is_summary
                FROM public.chat_hist
                WHERE user_id = %s
                AND paper_id = %s
                ORDER BY created_at
            """
            if limit:
                query += " LIMIT %s "
                cur.execute(query, (user_id, paper_id, limit))
            else:
                cur.execute(query, (user_id, paper_id))

            return cur.fetchall()
        except Exception as e:
            print(f"대화 불러오기 중 에러 발생: {str(e)}")
            raise
        finally:
            cur.close()

    def find_related_conversations(self, current_question, user_id,
                                    paper_id, similarity_threshold=0.85):
        """ 임베딩 유사도 기반 이전 질문 찾기 """
        # current_embedding = self.embedding_api.get_embedding(current_question)
        current_embedding = self.model.encode(current_question).tolist()

        query = """
            SELECT 
                m1.message as question,
                m2.message as answer,
                1 - cdb_admin.cosine_distance(m1.embedding, %s::cdb_admin.vector) as similarity
            FROM public.chat_hist m1
            JOIN public.chat_hist m2 ON m2.parent_message_id = m1.chat_id
            WHERE m1.user_id = %s
                AND m1.paper_id = %s
                AND m1.role = 'user'
                AND m2.role = 'assistant'
                AND m1.embedding IS NOT NULL
                AND 1 - cdb_admin.cosine_distance(m1.embedding, %s::cdb_admin.vector) > %s
            ORDER BY similarity DESC
            LIMIT 3
        """

        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query, (current_embedding, user_id, paper_id, current_embedding, 
                                       similarity_threshold))
                results = cursor.fetchall()

                if results:
                    related_conversations = []
                    for result in results:
                        question, answer, similarity = result
                        related_conversations.append({
                            'question': question,
                            'answer': answer,
                            'similarity': similarity
                        })
                        return related_conversations
                    
                return None
        except Exception as e:
            print(f"쿼리 실행 중 에러 발생: {str(e)}")
            
        return None
    
    def handle_question(self, current_question, user_id, paper_id, context=None):
        """ 질문 처리 메인 로직 """
        if context is None:
            context = ""
        else:
            context = context['content']

        cache_key = f"{user_id}:{paper_id}:{current_question}:{sha256(context.encode()).hexdigest()}"

        if cache_key in self.cache:
            return {
                'type': 'context',
                'content': f"Previous related answer:\n{self.cache[cache_key]}"
            }
        
        related_conversations = self.find_related_conversations(current_question, user_id, paper_id)

        if related_conversations:
            context = "Related previous conversations:\n"
            for question, answer, similarity in related_conversations:
                context += f"Q: {question}\nA: {answer}\n\n"

            return {
                'type': 'context',
                'content': context
            }
        
        return None
    
    def get_last_response(self, user_id, paper_id):
        """ 마지막 응답 가져오기 """
        query = """
            SELECT
                m2.message as answer,
                m2.chat_type as response_type
            FROM public.chat_hist m1
            JOIN public.chat_hist m2 ON m2.parent_message_id = m1.chat_id
            WHERE m1.user_id = %s
                AND m1.paper_id = %s
                AND m1.role = 'user'
                AND m2.role = 'assistant'
            ORDER BY m1.created_at DESC
            LIMIT 1
        """

        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query, (user_id, paper_id))
                result = cursor.fetchone()

                if result:
                    return {
                        "content": result[0],
                        "type": result[1]
                    }
                return None
        except Exception as e:
            print(f"마지막 응답 조회 중 오류 발생: {str(e)}")
            return None
    
class SearchFileText:
    def __init__(self, connection):
        self.conn = connection

    def search_similar_doc(self, query_vector, top_k=10):
        try:
            cur = self.conn.cursor()

            vector_str = f"[{','.join(map(str, query_vector))}]"

            cur.execute("""
                SELECT doc_id, page, content,
                cdb_admin.cosine_distance(embedding, %s::cdb_admin.vector) as distance
                FROM public.document
                ORDER BY cdb_admin.cosine_distance(embedding, %s::cdb_admin.vector)
                LIMIT %s
            """, (vector_str, vector_str, top_k))

            results = cur.fetchall()
            return [
                {
                    "id": row[0],
                    "page": row[1],
                    "text": row[2],
                    "score": 1 - row[3]
                } for row in results
            ]
        except Exception as e:
            print(f"조회 중 에러 발생: {str(e)}")
            return []
        finally:
            cur.close()

class SummaryFileText(BaseDBHandler):
    def __init__(self, connection):
        self.conn = connection

    def update_summary_text(self, user_id, paper_id, summary):
        query = """
            SELECT figure_id, storage_path, description
            FROM public.figure_info
            WHERE user_id = %s AND paper_id = %s
        """
        return self.execute_query_one(query, (user_id, paper_id))

class AdditionalFileUploader(BaseDBHandler):
    def __init__(self, connection):
        self.conn = connection

    def insert_figure_file(self, user_id, paper_id,
                           storage_path, caption_number, description):
        query = """
            INSERT INTO public.figure_info
            (user_id, paper_id, storage_path, 
            caption_number, description)
            VALUES (%s, %s, %s, %s, %s)
        """

        self.execute_query_one(query, (user_id, paper_id,
                                        storage_path, caption_number, description))
        
        

    def insert_table_file(self, user_id, paper_id,
                          table_obj, caption_number, description):
        query = """
            INSERT INTO public.table_info
            (user_id, paper_id, table_obj,
            caption_number, description)
            VALUES (%s, %s, %s, %s, %s)
        """

        self.execute_query_one(query, (user_id, paper_id,
                                        table_obj, caption_number, description))

    def insert_tag_file(self, user_id, paper_id,
                        tag_text):
        query = """
            INSERT INTO public.tag_info
            (user_id, paper_id, tag_text)
            VALUES (%s, %s, %s)
        """

        self.execute_query_one(query, (user_id, paper_id, tag_text))
    
    def insert_timeline_file(self, user_id, paper_id,
                             storage_path, timeline_name, description):
        query = """
            INSERT INTO public.timeline_info
            (user_id, paper_id, storage_path,
            timeline_name, description)
            VALUES (%s, %s, %s, %s, %s)
        """

        self.execute_query_one(query, (user_id, paper_id, storage_path,
                                         timeline_name, description))
        
    def insert_audio(self, user_id, paper_id, audio_file_path, 
                     thumbnail_path, audio_title, script):
        query = """
            INSERT INTO public.audio_info
            (user_id, paper_id, audio_file_path,
            thumbnail_path, audio_title, script)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        self.execute_query(query, (user_id, paper_id, audio_file_path, 
                                   thumbnail_path, audio_title, script))
        

    def search_figure_file(self, user_id, paper_id):
        query = """
            SELECT 
                storage_path, figure_number, description
            FROM public.figure_info
            WHERE user_id = user_id
            AND paper_id = paper_id
        """

        figures = self.execute_query(query, (user_id, paper_id))

        result = []
        for figure in figures:
            result.append({
                'storage_path': figure[0],
                'figure_number': figure[1],
                'description': figure[2]
            })

        return result

    def search_table_file(self, user_id, paper_id):
        query = """
            SELECT 
                storage_path, table_name, description
            FROM public.table_info
            WHERE user_id = user_id
            AND paper_id = paper_id
        """

        tables = self.execute_query(query, (user_id, paper_id))

        result = []
        for table in tables:
            result.append({
                'storage_path': table[0],
                'table_name': table[1],
                'description': table[2]
            })

        return result

    def search_tag_text(self, user_id, paper_id):
        query = """
            SELECT 
                tag_text
            FROM public.tag_info
            WHERE user_id = user_id
            AND paper_id = paper_id
        """

        tags = self.execute_query_one(query, (user_id, paper_id))

        return tags

    def search_timeline_file(self, user_id, paper_id):
        query = """
            SELECT 
                storage_path, timeline_name, description
            FROM public.timeline_info
            WHERE user_id = user_id
            AND paper_id = paper_id
        """

        timeline = self.execute_query_one(query, (user_id, paper_id))

        if timeline:
            result = {
                'storage_path': timeline[0],
                'timeline_name': timeline[1],
                'description': timeline[2]
            }
            return result
        else:
            return None
        
    def search_audio_file(self, user_id, paper_id):
        query = """
            SELECT
                audio_file_path, thumbnail_path, script
            FROM public.audio_info
            WHERE user_id = %s
            AND paper_id = %s
        """

        audio = self.execute_query_one(query, (user_id, paper_id))

        if audio:
            result = {
                'audio_file_path': audio[0],
                'thumbnail_path': audio[1],
                'script': audio[2]
            }
            return result
        else:
            return None