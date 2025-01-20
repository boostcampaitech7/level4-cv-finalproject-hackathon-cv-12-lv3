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
    except Exception as e:
        print(f"Error connecting to database: {e}")

def test_connection():
    try:
        print("DB 설정값 확인:")
        print(f"HOST: {os.getenv('DB_HOST')}")
        print(f"NAME: {os.getenv('DB_NAME')}")
        print(f"USER: {os.getenv('DB_USER')}")
        print(f"PORT: {os.getenv('DB_PORT')}")
        print("연결 시도 중...")
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            port=os.getenv('DB_PORT'),
            connect_timeout=10
        )
        print("연결 성공!")
        conn.close()
    except Exception as e:
        print(f"연결 실패. 에러 상세: {str(e)}")

if __name__ == '__main__':
    try:
        test_connection()
    except Exception as e:
        print(f"Error during test: {str(e)}")