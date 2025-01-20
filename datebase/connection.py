import psycopg2
from config.config import DB_CONFIG

class DatabaseConnection:
    def __init__(self):
        self.conn = None

    def connect(self):
        try:
            self.conn = psycopg2.connect(
                **DB_CONFIG,
                connect_timeout=60
            )
            return self.conn
        except psycopg2.OperationalError as e:
            print(f"데이터베이스 연결 실패 (운영 에러): {e}")
        except Exception as e:
            print(f"데이터베이스 연결 실패: {e}")

    def close(self):
        if self.conn:
            self.conn.close()