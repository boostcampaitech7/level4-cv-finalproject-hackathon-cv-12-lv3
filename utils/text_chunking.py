import tiktoken, re
from typing import List
from sentence_transformers import util
import numpy as np

def num_tokens(text: str) -> int:
    """
    - 아직 필요한지는 미확실 -
    텍스트의 토큰 수를 계산하는 함수
    :param text: 입력 텍스트
    :return: 토큰 수
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))

def chunkify_with_overlap(sentences: List[str], chunk_size: int = 256, overlap_size: int = 50) -> List[str]:
    """
    문장 리스트를 토큰 수 기준으로 분할 (겹침 포함, 문장 단위 보장)
    
    :param sentences: 문장 리스트 (예: ["문장1", "문장2", ...])
    :param chunk_size: 최대 청크 토큰 수
    :param overlap_size: 겹칠 토큰 수
    :return: 분할된 청크 리스트
    """
    chunks = []
    current_chunk = []
    current_token_count = 0
    overlap_buffer = []

    for sentence in sentences:
        if not sentence:
            continue

        sentence_token_count = num_tokens(sentence)

        # 단일 문장이 chunk_size를 초과하는 경우 강제로 추가
        if sentence_token_count > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_token_count = 0
            chunks.append(sentence)
            continue

        # 현재 청크에 추가 시 토큰 초과 검사
        if current_token_count + sentence_token_count > chunk_size:
            chunks.append(" ".join(current_chunk))
            # 겹침을 위한 버퍼 업데이트: 현재 청크의 끝에서부터 토큰 수로 추출
            overlap_buffer = []
            overlap_count = 0
            for sent in reversed(current_chunk):
                token_count = num_tokens(sent)
                if overlap_count + token_count > overlap_size:
                    break
                overlap_buffer.insert(0, sent)
                overlap_count += token_count
            current_chunk = overlap_buffer
            current_token_count = overlap_count

        current_chunk.append(sentence)
        current_token_count += sentence_token_count

    # 마지막 청크 처리
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# def semantic_chunking(sentences, sentence_embeddings, threshold=0.7):
#     chunks = []
#     chunk_embeddings = []
#     current_chunk = []
#     current_embeddings = []
    
#     for i in range(len(sentences) - 1):
#         # 현재 문장과 다음 문장의 유사도 계산
#         similarity = util.cos_sim(sentence_embeddings[i], sentence_embeddings[i + 1])
        
#         if similarity > threshold:
#             current_chunk.append(sentences[i])
#             current_embeddings.append(sentence_embeddings[i])
#         else:
#             current_chunk.append(sentences[i])
#             current_embeddings.append(sentence_embeddings[i])
            
#             # 청크 텍스트와 임베딩 저장
#             chunks.append(". ".join(current_chunk))
#             chunk_embeddings.append(np.mean(current_embeddings, axis=0).tolist())  # 임베딩 평균
#             current_chunk = []
#             current_embeddings = []
    
#     if current_chunk:
#         chunks.append(". ".join(current_chunk))
#         chunk_embeddings.append(np.mean(current_embeddings, axis=0).tolist())
    
#     return chunks, chunk_embeddings

def semantic_chunking(sentences, sentence_embeddings, threshold=0.7, target_chunk_size=None, dynamic_threshold_increment=0.1):
    # Preprocess to identify special content (figures, tables, equations)
    special_indices = set()
    for idx, sentence in enumerate(sentences):
        if has_special_content(sentence):
            special_indices.add(idx)
    
    chunks = []
    chunk_embeddings = []
    current_chunk = []
    current_embeddings = []
    
    for i in range(len(sentences) - 1):
        current_sentence = sentences[i]
        next_sentence = sentences[i+1]
        
        # Calculate original similarity
        similarity = util.cos_sim(sentence_embeddings[i], sentence_embeddings[i+1]).item()
        
        # Dynamic threshold adjustment for chunk size control
        adjusted_threshold = threshold
        if target_chunk_size:
            projected_size = len(current_chunk) + 1  # +1 for current_sentence
            if projected_size >= target_chunk_size:
                adjusted_threshold += dynamic_threshold_increment
        
        # Special content handling (lower threshold to keep context together)
        if i in special_indices or (i+1) in special_indices:
            adjusted_threshold = max(threshold - 0.2, 0.2)  # Lower threshold for special content
        
        if similarity > adjusted_threshold:
            current_chunk.append(current_sentence)
            current_embeddings.append(sentence_embeddings[i])
        else:
            current_chunk.append(current_sentence)
            current_embeddings.append(sentence_embeddings[i])
            chunks.append(". ".join(current_chunk))
            chunk_embeddings.append(np.mean(current_embeddings, axis=0).tolist())
            current_chunk = []
            current_embeddings = []
    
    # Handle last sentence and remaining content
    if current_chunk:
        current_chunk.append(sentences[-1])
        current_embeddings.append(sentence_embeddings[-1])
        chunks.append(". ".join(current_chunk))
        chunk_embeddings.append(np.mean(current_embeddings, axis=0).tolist())
    elif len(sentences) > 0 and len(current_chunk) > 0:
        chunks.append(sentences[-1])
        chunk_embeddings.append(sentence_embeddings[-1].tolist())
    
    return chunks, chunk_embeddings

def has_special_content(sentence):
    """
    마크다운 형식으로 표시된 수식, 표, 테이블이 포함되어 있는지 확인하는 함수.
    """
    
    # ✅ 마크다운 및 LaTeX 스타일 수식 탐지
    latex_pattern = r'(\$\$.*?\$\$)|(\$.*?\$)|\\\(|\\\)|\\begin{.*?}|\\end{.*?}|\\mathrm{.*?}|\\mathbf{.*?}'
    if re.search(latex_pattern, sentence, re.DOTALL):  # re.DOTALL: 여러 줄 수식도 탐지 가능
        return True

    # ✅ 마크다운 표 탐지 ( "|" 기호로 구분되는 행 형태 )
    table_pattern = r'^\|(.+?)\|$'
    if re.search(table_pattern, sentence.strip()):
        return True
    
    # ✅ 'figure' 또는 'table' 포함 여부 확인 (대소문자 무시)
    if 'figure' in sentence.lower() or 'table' in sentence.lower():
        return True

    return False