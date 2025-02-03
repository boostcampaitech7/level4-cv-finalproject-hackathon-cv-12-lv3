from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np
import time

def is_follow_up_request(query:str, model) -> bool:
    follow_up_patterns = [
        "자세히 설명해줘",
        "더 설명해줘",
        "자세하게 설명해줄래",
        "추가 설명해줄래",
        "다시 설명해줘"
    ]

    query_vector = np.array(model.encode(query)).reshape(1, -1)
    pattern_vectors = np.array([model.encode(pattern) for pattern in follow_up_patterns])

    similarities = cosine_similarity(query_vector, pattern_vectors)[0]
    print(f"자세히 설명 점수: {max(similarities)}")
    return max(similarities) > 0.7

def is_paper_info_request(query: str, model) -> bool:
    paper_info_patterns = [
        "이 논문이 뭐야",
        "이 논문에 대해 설명해줘",
        "논문 정보를 알려줘",
        "이 논문은 무슨 내용이야",
        "논문 개요를 알려줘",
        "이 논문의 주제가 뭐야",
        "논문 요약해줘"
    ]

    query_vector = np.array(model.encode(query)).reshape(1, -1)
    pattern_vectors = np.array([model.encode(pattern) for pattern in paper_info_patterns])

    similarities = cosine_similarity(query_vector, pattern_vectors)[0]
    print(f"논문 정보 요청 점수: {max(similarities)}")

    return max(similarities) > 0.7

def search_similar_doc(query_vector, conn, session_id, top_k=10):
    """
    고유한 문서만 선택하는 함수
    :param query_vector: 쿼리 벡터
    :param conn: 데이터베이스 연결 객체
    :param top_k: 상위 k개의 문서를 반환
    :return: 고유한 문서 리스트
    """
    cur = conn.cursor()
    vector_str = f"[{','.join(map(str, query_vector))}]"

    # 고유한 문서만 선택 (DISTINCT ON 사용)
    cur.execute("""
        SELECT id, page, content, 
        cdb_admin.cosine_distance(embedding, %s::cdb_admin.vector) as distance
        FROM public.documents
        WHERE session_id = %s
        ORDER BY cdb_admin.cosine_distance(embedding, %s::cdb_admin.vector), content
        LIMIT %s 
    """, (vector_str, session_id, vector_str, top_k))

    results = cur.fetchall()
    return [
        {
            "id": row[0],
            "page": row[1],
            "text": row[2],
            "score": 1 - row[3]  # 유사도 점수 (1 - 거리)
        }
        for row in results
    ]
    
    
def clean_clova_response(response_dict):
    """
    클로바의 응답에서 답변을 추출하는 함수
    :param response_dict: 클로바의 응답(JSON 형태)
    """
    if 'message' in response_dict:
        return response_dict['message']['content']
    return "응답을 가져오는데 실패했습니다."

### Multi Retrival 하는 코드 추가하기
def query_and_respond(query: str, conn, model, session_id, top_k=3,
                      chat_manager=None):
    """
    사용자의 쿼리를 임베딩하고, 검색하는 함수
    :param conn: db 접근
    :param model: 임베딩 모델
    :param session_id: 세션 채팅 ID
    :param top_k: 벡터 서치에서 추출할 Reference의 개수
    """
    try:
        if is_follow_up_request(query, model):
            last_response = chat_manager.get_last_response(session_id)
            if last_response and last_response['type'] in ['unrelated', 'no_result']:
                return {
                    "type": "unrelated",
                    "message": "죄송하지만 이전 질문이 논문과 관련이 없어 추가 설명을 드릴 수 없어요."
                }
            else:
                return {
                    "type": "details",
                    "message": query
                }
        
        query_vector = model.encode(query).tolist()
        matches = search_similar_doc(
            query_vector=query_vector,
            conn=conn,
            session_id=session_id,
            top_k=top_k,
        )

        if matches:
            score = matches[0]['score']
            print(score)
            if score > 0.5:
                return {
                    "type": "reference",
                    "content": "\n\n".join([
                        f"Reference (Page {match['page']}): {match['text']}"
                        for match in matches[:top_k]
                    ])
                }
            elif 0.35 <= score <= 0.5: 
                return {
                    "type": "insufficient",
                    "message": "제공된 Reference에서는 내용이 부족해요. 외부 자료를 통해 추가로 설명해드릴까요?"
                }
            else:
                return {
                    "type": "unrelated",
                    "message": "이 질문은 논문과 관련이 없어요. 논문에 대한 질문을 해주시면 도와드릴게요!"
                }
        else:
            return {
                "type": "no_result",
                "message": "검색 결과가 없습니다."
            }
        
    except Exception as e:
        print(f"벡터 검색 중 에러 발생: {str(e)}")
        return None
    
### 이부분도 받은 모델을 기반으로 rerank 진행후 query_and_respond를 수행할 수 있도록 수정하기
    
def rerank_with_cross_encoder(query, documents, model, top_k=10):
    """
    Cross-Encoder를 사용해 문서를 재정렬합니다.
    :param query: 질문 텍스트
    :param documents: 재정렬할 문서 리스트 (dict 형태)
    :param model: Cross-Encoder 모델
    :param top_k: 상위 k개의 문서를 반환
    :return: 재정렬된 문서 리스트
    """
    pairs = [(query, doc["text"]) for doc in documents]
    scores = model.predict(pairs)
    
    # 점수를 기반으로 문서 재정렬
    for doc, score in zip(documents, scores):
        doc["score"] = float(score)  # Cross-Encoder 점수로 업데이트
    
    # 점수 기준으로 정렬
    reranked_documents = sorted(documents, key=lambda x: x["score"], reverse=True)
    return reranked_documents[:top_k]


### 여기서 비교하는 코드에서 여러 모델을 받아서 비교할 수 있도록 수정하기.
def query_and_respond_reranker_compare(query: str, conn, model, reranker_model, session_id, top_k=3):
    """
    BGE-M3만 사용한 검색과 BGE-M3 + Cross-Encoder 리랭커를 사용한 검색을 비교합니다.
    :param conn: db 접근
    :param model: 임베딩 모델 (BGE-M3)
    :param reranker_model: Cross-Encoder 모델
    :param session_id: 세션 채팅 ID
    :param top_k: 벡터 서치에서 추출할 Reference의 개수
    :return: 재정렬된 문서 리스트
    """
    try:
        # 1. 쿼리 임베딩
        query_vector = model.encode(query).tolist()
        
        # 2. BGE-M3만 사용한 검색 (처리 시간 측정)
        start_time_bge = time.time()
        matches_bge = search_similar_doc(
            query_vector=query_vector,
            conn=conn,
            session_id=session_id,
            top_k=top_k * 2  # 초기 검색 결과를 더 많이 가져옴
        )
        end_time_bge = time.time()
        time_bge = end_time_bge - start_time_bge

        # matches_bge가 비어 있는지 확인
        if not matches_bge:
            print("BGE-M3 검색 결과가 없습니다.")
            return None

        # 3. BGE-M3 + Cross-Encoder 리랭커 (처리 시간 측정)
        start_time_rerank = time.time()
        matches_reranked = rerank_with_cross_encoder(query, matches_bge, reranker_model, top_k=top_k)
        end_time_rerank = time.time()
        time_rerank = end_time_rerank - start_time_rerank

        # matches_reranked가 비어 있는지 확인
        if not matches_reranked:
            print("리랭킹 결과가 없습니다.")
            return None

        # 4. 결과 출력
        print("=" * 50)
        print("BGE-M3만 사용한 검색 결과:")
        print("-" * 50)
        for match in matches_bge[:top_k]:  
            print(f"Page {match['page']}: {match['text']}/n(Score: {match['score']:.4f})")
        print(f"처리 시간: {time_bge:.4f} 초")
        print("=" * 50)

        print("BGE-M3 + Cross-Encoder 리랭커를 사용한 검색 결과:")
        print("-" * 50)
        for match in matches_reranked[:top_k]:  
            print(f"Page {match['page']}: {match['text']}/n(Score: {match['score']:.4f})")
        print(f"처리 시간: {time_rerank:.4f} 초")
        print("=" * 50)

        # 5. 최종 결과 반환 (리랭킹된 결과 사용)
        references = "\n\n".join([
            f"Reference (Page {match['page']}): {match['text']}"
            for match in matches_reranked[:top_k]  
        ])
        return references
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
    
def rerank_with_cross_encoder_v2(query: str, matches: List[dict], 
                               reranker: CrossEncoder, top_k: int = 3,
                               batch_size: int = 16) -> List[dict]:
    """
    Cross-encoder를 사용하여 검색 결과를 재순위화하는 함수
    
    Args:
        query: 검색 쿼리
        matches: BGE 임베딩으로 검색된 결과 리스트
        reranker: Cross-encoder 모델
        top_k: 반환할 상위 결과 수
        batch_size: 배치 처리 크기
    
    Returns:
        재순위화된 결과 리스트 (상위 top_k개)
    """
    if not matches:
        return []

    # 쿼리-문서 쌍 생성
    pairs = [(query, match['text']) for match in matches]
    
    # 배치 단위로 점수 계산
    scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        batch_scores = reranker.predict(batch)
        scores.extend(batch_scores)
    
    # 점수와 원본 데이터 결합
    scored_results = []
    for match, score in zip(matches, scores):
        result = match.copy()
        result['score'] = float(score)
        scored_results.append(result)
    
    # 점수 기준 내림차순 정렬 후 상위 k개 반환
    return sorted(scored_results, 
                 key=lambda x: x['score'], 
                 reverse=True)[:top_k]

def split_into_subqueries(query: str, completion_executor, max_subqueries: int = 3) -> List[str]:
    system_prompt = """
    주어진 질문에 대해 2-3개의 구체적이고 기술적인 하위 질문을 만들어주세요.
    
    규칙:
    1. 각 하위 질문은 '###' 구분자로 분리해서 출력
    2. 연관된 개념(예: 쿼리, 키, 값)은 절대 분리하지 말 것
    3. 전체 개념을 포괄하는 질문으로 작성
    4. 각 하위 질문은 완전한 문장이어야 함
    
    예시 입력: "Q, K, V가 뭐야?"
    예시 출력:
    Query, Key, Value의 개념과 역할은 무엇인가요?###Query, Key, Value는 어텐션 메커니즘에서 어떻게 상호작용하나요?###Query, Key, Value를 사용한 계산 과정은 어떻게 되나요?
    
    입력: {query}
    """

    response = completion_executor.execute({
        "messages": [
            {"role": "system", "content": system_prompt.format(query=query)}
        ]
    })
    
    # '###' 구분자로 서브쿼리 분리
    subqueries = response['content'].split('###')
    
    # 공백 제거 및 필터링
    subqueries = [q.strip() for q in subqueries if q.strip()]
    
    return subqueries[:max_subqueries]

def calculate_rerank_score(all_results_reranked: List[dict], subqueries: List[str]) -> float:
    if not all_results_reranked or not subqueries:
        return 0.0
    
    scores = list(r['score'] for r in all_results_reranked)
    if not scores:
        return 0.0
        
    chunk_size = len(scores) // len(subqueries)
    if chunk_size == 0:
        return 0.0
        
    subquery_scores = []
    for i in range(0, len(scores), chunk_size):
        chunk = scores[i:i + chunk_size]
        if chunk:
            subquery_scores.append(max(chunk))
    
    return sum(subquery_scores) / len(subquery_scores) if subquery_scores else 0.0

def process_query_with_reranking_compare(
        query: str,
        conn,
        model,
        reranker,
        completion_executor,
        session_id,
        top_k: int = 3,
        chat_manager = None,
        paper_manager = None,
):
    try:
        if is_follow_up_request(query, model):
            last_response = chat_manager.get_last_response(session_id)
            if last_response and last_response['type'] in ['unrelated', 'no_result']:
                return {
                    "type": "unrelated",
                    "message": "죄송하지만 이전 질문과 관련이 없어 추가 설명을 드릴 수 없어요."
                }
            else:
                return {
                    "type": "details",
                    "message": query
                }
            
        if is_paper_info_request(query, model):
            paper_info = paper_manager.get_paper_info(session_id)
            
            if paper_info:
                title, authors, abstract, year = paper_info
                paper_info_text = f"""
                제목: {title}
                저자: {authors}
                발행년도: {year}
                초록: {abstract}
                """
                return {
                    "type": "paper_info",
                    "content": paper_info_text
                }
            
        subqueries = split_into_subqueries(query, completion_executor)
        print(f"생성 서브쿼리들: {subqueries}")

        all_results_reranked = []

        for idx, subquery in enumerate(subqueries):
            print(f"\n=== 서브쿼리 {idx+1} 처리 중: {subquery} ===")

            query_vector = model.encode(subquery).tolist()
            matches_bge = search_similar_doc(
                query_vector=query_vector,
                conn=conn,
                session_id=session_id,
                top_k=top_k * 2
            )
            print(f"BGE 검색 결과 수: {len(matches_bge)}")

            if matches_bge:
                matches_reranked = rerank_with_cross_encoder_v2(
                    subquery, matches_bge, reranker, top_k=1
                )
                print(f"재순위 결과 수: {len(matches_reranked) if matches_reranked else 0}")
                if matches_reranked:
                    all_results_reranked.extend(matches_reranked)

        if all_results_reranked:
            # avg_rerank_score = sum(r['score'] for r in all_results_reranked) / len(all_results_reranked)
            rerank_score = calculate_rerank_score(all_results_reranked, subqueries)
            print(f"\n=== 최종 결과 ===")
            print(f"전체 재순위 결과 수: {len(all_results_reranked)}")
            print(f"평균 재순위 점수: {rerank_score:.4f}")

            if rerank_score <= 0.3:
                return {
                    "type": "unrelated",
                    "message": "이 질문은 논문과 관련이 없어요. 논문에 대한 질문을 해주시면 도와드릴게요!"
                }
            elif rerank_score <= 0.4:
                return {
                    "type": "insufficient",
                    "message": "제공된 Reference에서는 내용이 부족해요. 외부 자료를 통해 추가로 설명해드릴까요?"
                }
            else:
                results_per_query = []
                chunk_size = len(all_results_reranked) // len(subqueries)

                for i, subquery in enumerate(subqueries):
                    start_idx = i * chunk_size
                    end_idx = start_idx + chunk_size
                    query_results = all_results_reranked[start_idx:end_idx]
                    
                    results_per_query.append(
                        f"서브쿼리: {subquery}\n" +
                        "\n".join([f"Reference (Page {match['page']}): {match['text']}"
                                for match in query_results[:top_k]])
                    )

                return {
                    "type": "reference",
                    "content": "\n\n=== 다음 검색 결과 ===\n\n".join(results_per_query)
                }
        else:
            return {
                "type": "no_result",
                "message": "검색 결과가 없습니다."
            }
    except Exception as e:
        print(f"서브쿼리 생성 및 reranking 중 에러 발생: {str(e)}")
        return None