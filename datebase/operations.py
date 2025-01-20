from tqdm import tqdm

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