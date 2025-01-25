from connection import get_db_connection

try:
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SET search_path TO public, cdb_admin;")

    # 1. Sessions table
    cur.execute("""
            CREATE TABLE IF NOT EXISTS public.sessions (
                session_id VARCHAR(36) PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            );
        """)
    conn.commit()

    # 2. Papers_info table
    cur.execute("""
            CREATE TABLE IF NOT EXISTS public.papers_info (
                paper_id SERIAL PRIMARY KEY,
                session_id VARCHAR(36) REFERENCES sessions(session_id),
                title TEXT NOT NULL,
                authors TEXT,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
    conn.commit()

    # 3. Documents table
    cur.execute("""
            CREATE TABLE IF NOT EXISTS public.documents (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(36) REFERENCES sessions(session_id),
                paper_id INTEGER REFERENCES papers_info(paper_id),
                page INTEGER,
                content TEXT,
                embedding cdb_admin.vector(1024),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
    conn.commit()

    # 4. Chat_history table
    cur.execute("""
            CREATE TABLE IF NOT EXISTS public.chat_history (
                chat_id SERIAL PRIMARY KEY,
                session_id VARCHAR(36) REFERENCES sessions(session_id),
                role VARCHAR(20),
                message TEXT,
                parent_message_id INTEGER REFERENCES chat_history(chat_id),
                context_docs INTEGER[],
                is_summary BOOLEAN DEFAULT FALSE,
                summary_for_chat_id INTEGER REFERENCES chat_history(chat_id),
                embedding cdb_admin.vector(1024),
                chat_type VARCHAR(20),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            );
        """)
    conn.commit()

    # 5. External_papers table
    cur.execute("""
            CREATE TABLE IF NOT EXISTS public.external_papers (
                paper_id INTEGER PRIMARY KEY REFERENCES papers_info(paper_id),
                session_id VARCHAR(36) REFERENCES sessions(session_id),
                title TEXT NOT NULL,
                author TEXT,
                abstract TEXT,
                publication_year INTEGER,
                source VARCHAR(50),
                url TEXT,
                embedding cdb_admin.vector(1024),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
    conn.commit()

    # create index
    cur.execute("""
            CREATE INDEX IF NOT EXISTS document_embedding_idx ON
                public.documents
            USING hnsw (embedding cdb_admin.vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)
    conn.commit()

    cur.execute("""
            CREATE INDEX IF NOT EXISTS extarnel_papers_embedding_idx ON
                public.external_papers
            USING hnsw (embedding cdb_admin.vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)
    conn.commit()

    print(f"모든 테이블 설정이 완료되었습니다!")

except Exception as e:
    print(f"Error: {str(e)}")
finally:
    if 'cur' in locals():
        cur.close()
    if 'conn' in locals():
        conn.close()