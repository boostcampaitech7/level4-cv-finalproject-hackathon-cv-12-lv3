import os
from dotenv import load_dotenv
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from psycopg2.extensions import connection, cursor

load_dotenv()

class DatabaseConnection:
    _pool = None

    @classmethod
    def initialize_pool(cls):
        if cls._pool is None:
            try:
                cls._pool = pool.SimpleConnectionPool(
                    minconn=3,
                    maxconn=20,
                    host=os.getenv('DB_HOST'),
                    dbname=os.getenv('DB_NAME'),
                    user=os.getenv('DB_USER'),
                    password=os.getenv('DB_PASSWORD'),
                    port=os.getenv('DB_PORT'),
                    connect_timeout=30,
                    keepalives=1,
                    keepalives_idle=30,
                    keepalives_interval=10,
                    keepalives_count=5
                )
            except Exception as e:
                print(f"Pool 초기화 실패: {e}")
                raise

    def __init__(self):
        self.conn = None
        if DatabaseConnection._pool is None:
            DatabaseConnection.initialize_pool()

    def connect(self):
        try:
            if self.conn is None or (hasattr(self.conn, 'closed') 
                                     and self.conn.closed):
                self.conn = DatabaseConnection._pool.getconn()
            if self.conn is None:
                raise Exception("데이터베이스 연결이 설정되지 않았습니다.")
            return self.conn
        except psycopg2.OperationalError as e:
            print(f"데이터베이스 연결 실패 (운영 에러): {e}")
            if self.conn:
                DatabaseConnection._pool.putconn(self.conn)
            self.conn = DatabaseConnection._pool.getconn()
            return self.conn
        except Exception as e:
            print(f"데이터베이스 연결 실패: {e}")
            raise

    def reset_connection(self):
        if self.conn and not self.conn.closed:
            DatabaseConnection._pool.putconn(self.conn)
        return self.connect()
    
    def close(self):
        if self.conn:
            # DatabaseConnection._pool.putconn(self.conn)
            self._pool.putconn(self.conn)
            self.conn = None

    @classmethod
    def close_all(cls):
        if cls._pool:
            cls._pool.closeall()

def get_db_connection() -> connection:
    db_connection = DatabaseConnection()
    return db_connection.connect()