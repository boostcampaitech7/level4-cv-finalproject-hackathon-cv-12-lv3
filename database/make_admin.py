from operations import UserManager
from datetime import date
import psycopg2
from dotenv import load_dotenv
import os

# 환경변수 로드
load_dotenv()

# DB 연결 먼저 설정
conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    port=os.getenv('DB_PORT')
)

# UserManager 초기화할 때 connection 전달
user_manager = UserManager(conn)

try:
    admin_info = {
        "user_id": "admin",
        "user_pw": "!cv12admin!",
        "username": "관리자",
        "birth": date.today()
    }

    if not user_manager.user_exists(admin_info["user_id"]):
        success = user_manager.create_user(
            admin_info["user_id"],
            admin_info["user_pw"],
            admin_info["username"],
            admin_info["birth"]
        )
        print(f"Admin 계정 생성 {'성공' if success else '실패'}")
    else:
        print("Admin 계정이 이미 존재합니다.")
finally:
    conn.close()
