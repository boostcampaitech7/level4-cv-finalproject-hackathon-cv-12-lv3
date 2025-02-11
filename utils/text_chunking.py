import tiktoken
import re
from typing import List
from sentence_transformers import util
import numpy as np
import torch


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


def chunkify_to_num_token(sentences, chunk_size=256):
    """
    텍스트를 토큰 수 기준으로 분할하는 함수 (중복 방지)
    :param text: 입력 텍스트
    :param chunk_size: 청크당 최대 토큰 수
    :return: 분할된 청크 리스트
    """
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


def group_academic_paragraphs(sentences, model, max_sentences=5, similarity_threshold=0.4, window_size=3):
    # 임베딩을 한 번만 계산
    embeddings = model.encode(sentences, convert_to_tensor=True)

    groups = []
    current_group = [sentences[0]]
    current_avg_embedding = embeddings[0]

    for i in range(1, len(sentences)):
        window_start = max(0, i - window_size)
        window_embeddings = embeddings[window_start:i]
        window_avg = torch.mean(window_embeddings, dim=0)

        context_similarity = util.pytorch_cos_sim(
            window_avg, embeddings[i]).item()
        direct_similarity = util.pytorch_cos_sim(
            embeddings[i-1], embeddings[i]).item()

        weighted_similarity = 0.7 * context_similarity + 0.3 * direct_similarity

        if weighted_similarity >= similarity_threshold and len(current_group) < max_sentences:
            current_group.append(sentences[i])
            current_avg_embedding = torch.mean(
                embeddings[len(current_group)-window_size:len(current_group)], dim=0)
        else:
            # 현재 그룹을 최종 그룹에 추가
            groups.append(current_group)
            # 새 그룹 시작
            current_group = [sentences[i]]

    # 마지막 그룹 처리
    if current_group:
        groups.append(current_group)

    # 모든 그룹을 max_sentences 기준으로 재분할
    final_groups = []
    for group in groups:
        # 그룹을 max_sentences 크기의 서브그룹으로 분할
        for i in range(0, len(group), max_sentences):
            final_groups.append(group[i:i+max_sentences])

    for i in range(1, len(final_groups)):  # 첫 번째 문단은 제외하고 시작
        if len(final_groups[i]) <= 2:
            # 이전 문단과 합치기
            final_groups[i-1].extend(final_groups[i])
            final_groups[i] = []

    # 빈 리스트 제거
    final_groups = [group for group in final_groups if group]

    return final_groups


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
        similarity = util.cos_sim(
            sentence_embeddings[i], sentence_embeddings[i+1]).item()

        # Dynamic threshold adjustment for chunk size control
        adjusted_threshold = threshold
        if target_chunk_size:
            projected_size = len(current_chunk) + 1  # +1 for current_sentence
            if projected_size >= target_chunk_size:
                adjusted_threshold += dynamic_threshold_increment

        # Special content handling (lower threshold to keep context together)
        if i in special_indices or (i+1) in special_indices:
            # Lower threshold for special content
            adjusted_threshold = max(threshold - 0.2, 0.2)

        if similarity > adjusted_threshold:
            current_chunk.append(current_sentence)
            current_embeddings.append(sentence_embeddings[i])
        else:
            current_chunk.append(current_sentence)
            current_embeddings.append(sentence_embeddings[i])
            chunks.append(". ".join(current_chunk))
            chunk_embeddings.append(
                np.mean(current_embeddings, axis=0).tolist())
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
