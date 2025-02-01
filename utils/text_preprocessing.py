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


def split_sentences(text, min_length=100):
    # 숫자, 수식, 연속된 마침표, 여러 기호를 제외할 정규식 패턴
    math_pattern = r"\$\$(.*?)\$\$"  # 수식 구문을 포함한 부분
    number_pattern = r"\b\d+\b"  # 숫자만 있는 부분
    continuous_dot_pattern = r"\.\s*\.\s*\."  # 연속된 마침표
    multiple_punctuation_pattern = r"[!?.]{2,}"  # 연속된 구분 기호 (예: !!, .., ???)
    table_pattern = r"\|.*\|.*\|"  # 테이블 형식 구문을 포함한 부분

    # 수식, 숫자, 연속된 마침표 등을 제외한 텍스트를 분리
    text_no_math = re.sub(math_pattern, "", text)  # 수식 제거
    text_no_numbers = re.sub(number_pattern, "", text_no_math)  # 숫자 제거
    text_no_continuous_dot = re.sub(continuous_dot_pattern, "", text_no_numbers)  # 연속된 마침표 제거
    text_cleaned = re.sub(multiple_punctuation_pattern, "", text_no_continuous_dot)  # 연속된 구분 기호 제거

    # 마침표, 물음표, 느낌표 등을 기준으로 문장 나누기
    sentences = re.split(r'(?<=[.?!])\s+', text_cleaned)

    result = []
    current_sentence = ""

    for sentence in sentences:
        # 테이블 형식이나 제외할 내용이 포함된 문장은 건너뛰기
        if re.match(table_pattern, sentence.strip()):
            continue
        
        # 문장에 수식, 숫자, 연속된 마침표, 여러 기호 등이 포함된 경우 제외
        if re.search(math_pattern, sentence) or re.search(number_pattern, sentence) or \
           re.search(continuous_dot_pattern, sentence) or re.search(multiple_punctuation_pattern, sentence):
            continue
        
        # 현재 문장에 추가하기
        if len(current_sentence) + len(sentence) <= min_length:
            current_sentence += sentence + " "
        else:
            if current_sentence:
                result.append(current_sentence.strip())
            current_sentence = sentence + " "
    
    # 마지막 문장 추가
    if current_sentence.strip():
        result.append(current_sentence.strip())

    return result
