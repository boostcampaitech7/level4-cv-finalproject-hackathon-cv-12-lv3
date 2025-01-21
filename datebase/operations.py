from tqdm import tqdm
from uuid import uuid4
from datetime import datetime

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

class PaperManager:
    def __init__(self, connection):
        self.conn = connection

    def store_paper_info(self, session_id, title, authors=None):
        try:
            cur = self.conn.cursor()

            cur.execute("""
                INSERT INTO public.papers_info (session_id, title, authors)
                VALUES (%s, %s, %s)
                RETURNING paper_id
            """, (session_id, title, authors))

            paper_id = cur.fetchone()[0]
            self.conn.commit()
            return paper_id
        except Exception as e:
            self.conn.rollback()
            print(f"논문 정보 저장 중 에러 발생: {str(e)}")
            raise
        finally:
            cur.close()

class DocumentUploader:
    def __init__(self, connection):
        self.conn = connection

    def upload_documents(self, chunked_documents):
        try:
            cur = self.conn.cursor()
            count = 0

            for doc in tqdm(chunked_documents):
                vector_str = f"[{','.join(map(str, doc['embedding']))}]"
                cur.execute("""
                    INSERT INTO public.documents (page, content, embedding)
                    VALUES (%s, %s, %s::cdb_admin.vector)            
                """, (doc["page"], doc["chunk"], vector_str))
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
    def __init__(self, connection):
        self.conn = connection

    def store_chat(self, session_id, role, message, parent_id=None,
                   is_summary=False, summary_for_chat_id=None, context_docs=None):
        try:
            cur = self.conn.cursor()

            cur.execute("""
                INSERT INTO public.chat_history
                (session_id, role, message, parent_message_id,
                 is_summary, summary_for_chat_id, context_docs)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING chat_id
            """, (session_id, role, message, parent_id,
                  is_summary, summary_for_chat_id, context_docs))
            
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"채팅 로그 저장 중 에러 발생: {str(e)}")
            raise
        finally:
            cur.close()

    def store_conversation(self, session_id, user_message, llm_response,
                           parent_id=None, context_docs=None):
        """ 대화 쌍 (사용자 메시지 + AI 응답) 저장 """
        try:
            user_chat_id = self.store_chat(
                session_id=session_id,
                role='user',
                message=user_message,
                parent_id=parent_id,
                context_docs=context_docs
            )

            assistant_chat_id = self.store_chat(
                session_id=session_id,
                role='assistant',
                message=llm_response,
                parent_id=user_chat_id,
                context_docs=context_docs
            )

            self.conn.commit()

            return assistant_chat_id
        except Exception as e:
            self.conn.rollback()
            print(f"대화 저장 중 에러 발생: {str(e)}")
            raise

    def store_summary(self, session_id, summary, summarized_chat_ids):
        """ 요약본 저장 """
        try:
            summary_id =  self.store_chat(
                session_id=session_id,
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

    def get_chat_history(self, session_id, limit=None):
        """ 세션 히스토리 조회 """
        try:
            cur = self.conn.cursor()

            query = """
                SELECT chat_id, role, message, created_at, is_summary
                FROM public.chat_history
                WHERE session_id = %s
                ORDER BY created_at
            """
            if limit:
                query += " LIMIT %s "
                cur.execute(query, (session_id, limit))
            else:
                cur.execute(query, (session_id,))

            return cur.fetchall()
        except Exception as e:
            print(f"대화 불러오기 중 에러 발생: {str(e)}")
            raise
        finally:
            cur.close()

class SearchFileText:
    def __init__(self, connection):
        self.conn = connection

    def search_similar_doc(self, query_vector, top_k=10):
        try:
            cur = self.conn.cursor()

            vector_str = f"[{','.join(map(str, query_vector))}]"

            cur.execute("""
                SELECT id, page, content
                cdb_admin.cosine_distance(embedding, %s::cdb_admin.vector) as distance
                FROM public.documents
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

class ExternalPaperManager:
    def __init__(self, connection):
        self.conn = connection

    def store_external_paper(self, paper_id, session_id, title, author=None,
                             abstract=None, year=None, source=None, url=None):
        try:
            cur = self.conn.cursor()

            cur.execute("""
                INSERT INTO public.external_papers
                (paper_id, session_id, title, author, abstract,
                 publication_year, source, url)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING paper_id
            """, (paper_id, session_id, title, author, abstract, year,
                  source, url))
            
            self.conn.comit()
            return cur.fetchone()[0]
        except Exception as e:
            self.conn.rollback()
            print(f"외부 논문 저장 중 에러 발생: {str(e)}")
            raise
        finally:
            cur.close()

    def get_related_papers(self, session_id, embedding, top_k=5):
        try:
            cur = self.conn.cursor()
            vector_str = f"[{','.join(map(str, embedding))}]"

            cur.execute("""
                SELECT  paper_id, title, author, abstract,
                        cdb_admin.cosine_distance(embedding,
                        %s::cdb_admin.vector) as distance
                FROM external_papers
                WHERE session_id = %s
                ORDER BY distance
                LIMIT %s
            """, (vector_str, session_id, top_k))

            return cur.fetchall()
        except Exception as e:
            print(f"논문 불러오기 중 에러 발생: {str(e)}")
        finally:
            cur.close()