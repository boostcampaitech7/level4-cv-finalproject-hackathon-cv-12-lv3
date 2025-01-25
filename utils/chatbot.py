from sklearn.metrics.pairwise import cosine_similarity
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