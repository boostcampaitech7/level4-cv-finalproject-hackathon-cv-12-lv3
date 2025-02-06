from connection import get_db_connection

try:
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SET search_path TO public, cdb_admin;")

    # 1. Users table
    cur.execute("""
            CREATE TABLE IF NOT EXISTS public.user_info(
                user_id VARCHAR(36) PRIMARY KEY,
                user_pw VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                username VARCHAR(50) NOT NULL,
                birth DATE NULL
            );
        """)
    conn.commit()

    # 2. 논문 정보 table
    cur.execute("""
            CREATE TABLE IF NOT EXISTS public.papers(
                paper_id SERIAL PRIMARY KEY,
                user_id VARCHAR(36) REFERENCES public.user_info(user_id),
                title VARCHAR(255) NOT NULL,
                author VARCHAR(255) NULL,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                pdf_file_path TEXT NULL,
                tran_pdf_file_path TEXT NULL,
                short_summary TEXT NULL,
                long_summary TEXT NULL
            );
        """)
    conn.commit()

    # 3. 채팅 기록 table
    cur.execute("""
            CREATE TABLE IF NOT EXISTS public.chat_hist(
                chat_id SERIAL PRIMARY KEY,
                user_id VARCHAR(36) REFERENCES public.user_info(user_id),
                paper_id INTEGER REFERENCES public.papers(paper_id),
                role VARCHAR(20),
                message TEXT,
                parent_message_id INTEGER REFERENCES public.chat_hist(chat_id),
                context_docs INTEGER[],
                is_summary BOOLEAN DEFAULT FALSE,
                summary_for_chat_id INTEGER REFERENCES public.chat_hist(chat_id),
                embedding cdb_admin.vector(1024),
                chat_type VARCHAR(20),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
    conn.commit()

    # 4. 오디오 정보 table
    cur.execute("""
            CREATE TABLE IF NOT EXISTS public.audio_info(
                content_id SERIAL PRIMARY KEY,
                paper_id INTEGER REFERENCES public.papers(paper_id),
                user_id VARCHAR(36) REFERENCES public.user_info(user_id),
                audio_file_path TEXT NULL,
                thumbnail_path TEXT NULL,
                audio_title VARCHAR(255) NULL,
                script TEXT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  
            );
        """)
    conn.commit()

    # 5. 피규어 정보 table
    cur.execute("""
            CREATE TABLE IF NOT EXISTS public.figure_info(
                figure_id SERIAL PRIMARY KEY,
                paper_id INTEGER REFERENCES public.papers(paper_id),
                user_id VARCHAR(36) REFERENCES public.user_info(user_id),
                storage_path TEXT NULL,
                caption_number VARCHAR(50) NOT NULL,
                description TEXT NULL    
            )
        """)
    conn.commit()

    # 6. 테이블 정보 table
    cur.execute("""
            CREATE TABLE IF NOT EXISTS public.table_info(
                table_id SERIAL PRIMARY KEY,
                paper_id INTEGER REFERENCES public.papers(paper_id),
                user_id VARCHAR(36) REFERENCES public.user_info(user_id),
                table_obj TEXT NULL,
                caption_number VARCHAR(255) NULL,
                description TEXT NULL
            )
        """)
    conn.commit()

    # 7. 타임라인 정보 table
    cur.execute("""
            CREATE TABLE IF NOT EXISTS public.timeline_info(
                timeline_id SERIAL PRIMARY KEY,
                paper_id INTEGER REFERENCES public.papers(paper_id),
                user_id VARCHAR(36) REFERENCES public.user_info(user_id),
                storage_path TEXT NULL,
                timeline_name VARCHAR(255) NULL,
                description TEXT NULL
            )
        """)
    conn.commit()

    # 8. 태그 정보 table
    cur.execute("""
            CREATE TABLE IF NOT EXISTS public.tag_info(
                tag_id SERIAL PRIMARY KEY,
                paper_id INTEGER REFERENCES public.papers(paper_id),
                user_id VARCHAR(36) REFERENCES public.user_info(user_id),
                tag_text VARCHAR(100) NOT NULL
            )
        """)
    conn.commit()

    # 9. 문서 정보 table
    cur.execute("""
            CREATE TABLE IF NOT EXISTS public.document(
                doc_id SERIAL PRIMARY KEY,
                paper_id INTEGER REFERENCES public.papers(paper_id),
                user_id VARCHAR(36) REFERENCES public.user_info(user_id),
                page INTEGER NULL,
                content TEXT NULL,
                embedding VECTOR(1024) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    conn.commit()

    # create index
    cur.execute("""
            CREATE INDEX IF NOT EXISTS doc_session_paper ON
                public.document(user_id, paper_id);
        """)
    conn.commit()

    cur.execute("""
            CREATE INDEX IF NOT EXISTS vector_idx ON
                public.document
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