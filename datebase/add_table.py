from db_test import get_db_connection

try:
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SET search_path TO public, cdb_admin;")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS public.documents (
            id SERIAL PRIMARY KEY,
            page INTEGER,
            content TEXT,
            embedding cdb_admin.vector(1536)
        );
    """)
    conn.commit()

    cur.execute("""
        CREATE INDEX ON public.documents
        USING hnsw (embedding cdb_admin.vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)
    conn.commit()

    print("설정이 모두 완료되었습니다!")
except Exception as e:
    print(f"Error {str(e)}")
finally:
    if 'cur' in locals():
        cur.close()
    if 'conn' in locals():
        conn.close()