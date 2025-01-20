import tiktoken, re
from typing import List

def num_tokens(text: str) -> int:
    """
    - 아직 필요한지는 미확실 -
    텍스트의 토큰 수를 계산하는 함수
    :param text: 입력 텍스트
    :return: 토큰 수
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))

def chunkify_to_num_token(text: str, chunk_size: int = 256) -> List[str]:
    """
    텍스트를 토큰 수 기준으로 분할하는 함수 (중복 방지)
    :param text: 입력 텍스트
    :param chunk_size: 청크당 최대 토큰 수
    :return: 분할된 청크 리스트
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)  # 문장 단위로 분할
    chunks = []
    current_chunk = ""
    current_token_count = 0

    for sentence in sentences:
        if not sentence:
            continue

        sentence_token_count = num_tokens(sentence)

        # 현재 청크에 문장을 추가할 경우 토큰 수 초과 여부 확인
        if current_token_count + sentence_token_count > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_token_count = 0

            # 문장 자체가 청크 크기를 초과하는 경우
            if sentence_token_count > chunk_size:
                # 문장을 더 작은 단위로 분할 (예: 단어 단위)
                words = sentence.split()
                for word in words:
                    word_token_count = num_tokens(word)
                    if current_token_count + word_token_count > chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = ""
                            current_token_count = 0
                        current_chunk += f" {word}"
                        current_token_count += word_token_count
                    else:
                        current_chunk += f" {word}"
                        current_token_count += word_token_count
            else:
                current_chunk += f" {sentence}"
                current_token_count += sentence_token_count
        else:
            current_chunk += f" {sentence}"
            current_token_count += sentence_token_count

    # 마지막 청크 추가
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
