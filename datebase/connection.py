import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.extensions import connection, cursor

load_dotenv()

def get_db_connection() -> connection:
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            port=os.getenv('DB_PORT'),
            connect_timeout=30,
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"Error connecting to database by oper: {e}")
        return None
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

class DatabaseConnection:
    def __init__(self):
        self.conn = None

    def connect(self):
        try:
            self.conn = get_db_connection()
            if self.conn is None:
                raise Exception("데이터베이스 연결이 설정되지 않았습니다.")
            return self.conn
        except psycopg2.OperationalError as e:
            print(f"데이터베이스 연결 실패 (운영 에러): {e}")
        except Exception as e:
            print(f"데이터베이스 연결 실패: {e}")
            raise

    def close(self):
        if self.conn:
            self.conn.close()