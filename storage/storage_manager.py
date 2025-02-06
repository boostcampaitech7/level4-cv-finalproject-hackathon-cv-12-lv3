from typing import Dict
import boto3
from botocore.config import Config
import os
from dotenv import load_dotenv

load_dotenv()

class ObjectStorageManager:
    def __init__(self):
        self.service_name = 's3'
        self.endpoint = os.getenv('NCP_ENDPOINT')
        self.region = os.getenv('NCP_REGION')
        self.access_key = os.getenv('NCP_ACCESS_KEY')
        self.secret_key = os.getenv('NCP_SECRET_KEY')

        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
            config=Config(
                signature_version='s3',  # 'v4' 대신 's3v4' 사용
                s3={'addressing_style': 'virtual'}
            )
        )

    def upload_pdf(self, file_path: str, 
                   bucket_name: str) -> Dict[str, str]:
        """PDF 파일 OS에 업로드 하고 URL과 경로 반환"""
        file_name = os.path.basename(file_path)
        storage_path = f"papers/original/{file_name}"

        try:
            self.s3_client.upload_file(
                file_path,
                bucket_name,
                storage_path,
                ExtraArgs={'ACL': 'private'}
            )

            url = f"https://kr.object.ncloudstorage.com/{bucket_name}/{storage_path}"

            return {
                "url": url,
                "path": storage_path
            }
        except Exception as e:
            print(f"pdf 파일 업로드 중 에러 발생: {str(e)}")
            return None
        
    def upload_figure(self, file_path: str,
                            bucket_name: str) -> Dict[str, str]:
        """Figure 파일 업로드"""
        file_name = os.path.basename(file_path)
        storage_path = f"figures/{file_name}"

        try:
            self.s3_client.upload_file(
                file_path,
                bucket_name,
                storage_path,
                ExtraArgs={'ACL': 'private'}
            )

            url = f"https://kr.object.ncloudstorage.com/{bucket_name}/{storage_path}"

            return {
                "url": url,
                "path": storage_path
            }
        except Exception as e:
            print(f"Figure 파일 업로드 중 에러 발생: {str(e)}")
            return None
        
    def upload_timeline(self, file_path: str,
                            bucket_name: str) -> Dict[str, str]:
        """timeline 파일 업로드"""
        file_name = os.path.basename(file_path)
        storage_path = f"timeline/{file_name}"

        try:
            self.s3_client.upload_file(
                file_path,
                bucket_name,
                storage_path,
                ExtraArgs={'ACL': 'private'}
            )

            url = f"https://kr.object.ncloudstorage.com/{bucket_name}/{storage_path}"

            return {
                "url": url,
                "path": storage_path
            }
        except Exception as e:
            print(f"timeline 파일 업로드 중 에러 발생: {str(e)}")
            return None
        
    def upload_script(self, file_path: str,
                            bucket_name: str) -> Dict[str, str]:
        """script 파일 업로드"""
        file_name = os.path.basename(file_path)
        storage_path = f"script/{file_name}"

        try:
            self.s3_client.upload_file(
                file_path,
                bucket_name,
                storage_path,
                ExtraArgs={'ACL': 'private'}
            )

            url = f"https://kr.object.ncloudstorage.com/{bucket_name}/{storage_path}"

            return {
                "url": url,
                "path": storage_path
            }
        except Exception as e:
            print(f"audio 파일 업로드 중 에러 발생: {str(e)}")
            return None
        
    def upload_audio(self, file_path: str,
                            bucket_name: str) -> Dict[str, str]:
        """audio 파일 업로드"""
        file_name = os.path.basename(file_path)
        storage_path = f"audio/{file_name}"

        try:
            self.s3_client.upload_file(
                file_path,
                bucket_name,
                storage_path,
                ExtraArgs={'ACL': 'private'}
            )

            url = f"https://kr.object.ncloudstorage.com/{bucket_name}/{storage_path}"

            return {
                "url": url,
                "path": storage_path
            }
        except Exception as e:
            print(f"audio 파일 업로드 중 에러 발생: {str(e)}")
            return None
        
    def upload_thumbnail(self, file_path: str,
                            bucket_name: str) -> Dict[str, str]:
        """thumbnail 파일 업로드"""
        file_name = os.path.basename(file_path)
        storage_path = f"thumbnail/{file_name}"

        try:
            self.s3_client.upload_file(
                file_path,
                bucket_name,
                storage_path,
                ExtraArgs={'ACL': 'private'}
            )

            url = f"https://kr.object.ncloudstorage.com/{bucket_name}/{storage_path}"

            return {
                "url": url,
                "path": storage_path
            }
        except Exception as e:
            print(f"thumbnail 파일 업로드 중 에러 발생: {str(e)}")
            return None
        
    def download_file(self, file_url: str, local_path: str, bucket_name: str) -> bool:
        try:
            self.s3_client.download_file(
                Bucket=bucket_name,
                Key=file_url,
                Filename=local_path
            )
            if os.path.exists(local_path):
                return True
            return False
        except Exception as e:
            print(f"파일 다운로드 실패: {str(e)}")
            return False