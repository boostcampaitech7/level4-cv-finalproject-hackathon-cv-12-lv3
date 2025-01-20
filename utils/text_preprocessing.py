import re

def clean_text(text):
    """
    추출받은 텍스트를 정제하는 함수
    :param text: 원본 텍스트 추출파일
    :return: 정제 후 텍스트
    """
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()