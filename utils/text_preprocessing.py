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


def split_sentences(text):
    # 1. LaTeX 수식 영역을 플레이스홀더로 치환합니다.
    # 인라인 수식: $...$
    # 디스플레이 수식: \[...\] 와 \(...\)
    math_patterns = [r'\$.*?\$', r'\\\[.*?\\\]', r'\\\(.*?\\\)']
    math_placeholders = {}
    math_counter = 0

    def math_replacer(match):
        nonlocal math_counter
        placeholder = f"__MATH{math_counter}__"
        math_placeholders[placeholder] = match.group(0)
        math_counter += 1
        return placeholder

    for pat in math_patterns:
        text = re.sub(pat, math_replacer, text)

    # 2. 괄호 내의 텍스트를 찾아서 그 안에 있는 '.'을 임시 토큰으로 변경합니다.
    #    (괄호 안의 내용은 단순히 .*? 로 처리합니다.
    def bracket_replacer(match):
        content = match.group(0)
        # 괄호 내부의 .을 치환합니다.
        return content.replace('.', '__DOT__')

    text = re.sub(r'\([^)]*\)', bracket_replacer, text)
    text = re.sub(r'\[[^\]]*\]', bracket_replacer, text)
    text = re.sub(r'\{[^}]*\}', bracket_replacer, text)

    # 3. 문장 분리:
    #    - [!?] 뒤의 공백은 무조건 분리.
    #    - '.' 뒤의 공백은 분리하되, 바로 앞이나 뒤가 숫자인 경우(예: 3.14)는 분리하지 않도록 함.
    #    - 줄바꿈(\n)도 분리 구분자로 처리.
    #
    # 정규표현식 설명:
    #   (?<=[!?])\s+         : !나 ? 뒤의 공백
    #   | (?<=[.])(?!\d)\s+   : . 뒤의 공백인데, 뒤에 숫자가 오지 않을 때
    #   | \n+                : 줄바꿈
    sentences = re.split(r'(?<=[.!?])\s+|(?<=[.])(?!\d)\s+|\n+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # 4. 플레이스홀더 복원
    # 괄호 내 __DOT__ 를 원래 점으로 복원
    sentences = [s.replace('__DOT__', '.') for s in sentences]
    # LaTeX 수식 플레이스홀더 복원
    for i, s in enumerate(sentences):
        for placeholder, original in math_placeholders.items():
            s = s.replace(placeholder, original)
        sentences[i] = s

    return sentences