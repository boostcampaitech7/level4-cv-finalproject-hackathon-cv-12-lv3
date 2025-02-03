from tqdm import tqdm
from uuid import uuid4
from datetime import datetime
from sentence_transformers import SentenceTransformer
from hashlib import sha256

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

    def store_paper_info(self, session_id, title, authors=None, abstract=None,
                         year=None):
        try:
            cur = self.conn.cursor()

            cur.execute("""
                INSERT INTO public.papers_info 
                (session_id, title, authors, abstract, publication_year)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING paper_id
            """, (session_id, title, authors, abstract, year))

            paper_id = cur.fetchone()[0]
            self.conn.commit()
            return paper_id
        except Exception as e:
            self.conn.rollback()
            print(f"논문 정보 저장 중 에러 발생: {str(e)}")
            raise
        finally:
            cur.close()

    def get_paper_info(self, session_id):
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT title, authors, abstract, publication_year
                FROM papers_info
                WHERE session_id = %s
            """, (session_id,))
            return cur.fetchone()
        finally:
            cur.close()

class DocumentUploader:
    def __init__(self, connection):
        self.conn = connection

    def clean_text(self, text):
        if text is None:
            return None
        
        try:
            if isinstance(text, str):
                text = text.encode('utf-8', errors='ignore')
            return text.decode('utf-8', errors='ignore').strip()
        except Exception as e:
            print(f"텍스트 정제 중 에러 발생: {str(e)}")
            return None

    def upload_documents(self, chunked_documents, session_id):
        try:
            cur = self.conn.cursor()
            count = 0

            for doc in tqdm(chunked_documents):
                try:
                    cleaned_text = self.clean_text(doc["chunk"])
                    if not cleaned_text:
                        print(f"빈 텍스트 건너뛰기: {doc['page']} 페이지")
                        continue

                    vector_str = f"[{','.join(map(str, doc['embedding']))}]"
                    
                    cur.execute("""
                        INSERT INTO public.documents (session_id, page, content, embedding)
                        VALUES (%s, %s, %s, %s::cdb_admin.vector)            
                    """, (session_id, doc["page"], cleaned_text, vector_str))
                    count += 1
                    
                except Exception as e:
                    print(f"개별 문서 삽입 중 에러 발생 (페이지 {doc['page']}): {str(e)}")
                    continue
            
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

    def add_to_cache(self, session_id, question, context, answer):
        """ 새로운 응답 캐시 저장 """
        cache_key = f"{session_id}:{question}:{sha256(context['content'].encode()).hexdigest()}"
        self.cache[cache_key] = answer

    def store_chat(self, session_id, role, message, parent_id=None,
                   is_summary=False, summary_for_chat_id=None, context_docs=None,
                   embedding=None, chat_type=None):
        try:
            cur = self.conn.cursor()

            cur.execute("""
                INSERT INTO public.chat_history
                (session_id, role, message, parent_message_id,
                 is_summary, summary_for_chat_id, context_docs,
                 embedding, chat_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING chat_id
            """, (session_id, role, message, parent_id,
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

    def store_conversation(self, session_id, user_message, llm_response,
                           parent_id=None, context_docs=None, embedding=None,
                           chat_type=None):
        """ 대화 쌍 (사용자 메시지 + AI 응답) 저장 """
        try:
            user_chat_id = self.store_chat(
                session_id=session_id,
                role='user',
                message=user_message,
                parent_id=parent_id,
                context_docs=context_docs,
                embedding=embedding,
                chat_type=chat_type
            )

            assistant_chat_id = self.store_chat(
                session_id=session_id,
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

    def find_related_conversations(self, current_question, session_id,
                                    similarity_threshold=0.85):
        """ 임베딩 유사도 기반 이전 질문 찾기 """
        # current_embedding = self.embedding_api.get_embedding(current_question)
        current_embedding = self.model.encode(current_question).tolist()

        query = """
            SELECT 
                m1.message as question,
                m2.message as answer,
                1 - cdb_admin.cosine_distance(m1.embedding, %s::cdb_admin.vector) as similarity
            FROM public.chat_history m1
            JOIN public.chat_history m2 ON m2.parent_message_id = m1.chat_id
            WHERE m1.session_id = %s
                AND m1.role = 'user'
                AND m2.role = 'assistant'
                AND m1.embedding IS NOT NULL
                AND 1 - cdb_admin.cosine_distance(m1.embedding, %s::cdb_admin.vector) > %s
            ORDER BY similarity DESC
            LIMIT 3
        """

        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query, (current_embedding, session_id, current_embedding, 
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
    
    def handle_question(self, current_question, session_id, context=None):
        """ 질문 처리 메인 로직 """
        if context is None:
            context = ""
        else:
            context = context['content']

        cache_key = f"{session_id}:{current_question}:{sha256(context.encode()).hexdigest()}"

        if cache_key in self.cache:
            return {
                'type': 'context',
                'content': f"Previous related answer:\n{self.cache[cache_key]}"
            }
        
        related_conversations = self.find_related_conversations(current_question, session_id)

        if related_conversations:
            context = "Related previous conversations:\n"
            for question, answer, similarity in related_conversations:
                context += f"Q: {question}\nA: {answer}\n\n"

            return {
                'type': 'context',
                'content': context
            }
        
        return None
    
    def get_last_response(self, session_id):
        """ 마지막 응답 가져오기 """
        query = """
            SELECT
                m2.message as answer,
                m2.chat_type as response_type
            FROM public.chat_history m1
            JOIN public.chat_history m2 ON m2.parent_message_id = m1.chat_id
            WHERE m1.session_id = %s
                AND m1.role = 'user'
                AND m2.role = 'assistant'
            ORDER BY m1.created_at DESC
            LIMIT 1
        """

        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query, (session_id,))
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