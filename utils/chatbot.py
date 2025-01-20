def search_similar_doc(query_vector, conn, top_k=10):
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
        SELECT DISTINCT ON (content) id, page, content, 
        cdb_admin.cosine_distance(embedding, %s::cdb_admin.vector) as distance
        FROM public.documents
        ORDER BY content, cdb_admin.cosine_distance(embedding, %s::cdb_admin.vector)
        LIMIT %s 
    """, (vector_str, vector_str, top_k))

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

def query_and_respond(query: str, conn, embedding_api, chat_api, top_k=3):
    """
    사용자의 쿼리를 임베딩하고, 검색하는 함수
    :param conn: db 접근?
    :param embedding_api: 네이버 임베딩 API
    :param chat_api: 네이버 클로바 채팅 API
    :param top_k: 벡터 서치에서 추출할 Reference의 개수
    """
    try:
        query_vector = embedding_api.get_embedding(query)

        matches = search_similar_doc(query_vector=query_vector, conn=conn, top_k=top_k)

        if matches:
            print("검색된 관련 내용:")
            print("-" * 50)
            for match in matches[:top_k]:
                print(f"ID: {match['id']}")
                print(f"Score: {match['score']}")
                print(f"Text: {match['text']}")
                print(f"Page: {match['page']}")
                print("-" * 50)

            references = "\n\n".join([
                f"Reference (Page {match['page']}): {match['text']}"
                for match in matches[:top_k]
            ])

            messages = [
                {
                    "role": "system",
                    "content": "\n".join([
                        "You are an AI assistant specialized in explaining concepts from academic papers.",
                        "Include the page number from the reference in your answer.",
                        "Base your answer solely on the provided reference.",
                        "Keep your explanation clear and concise."
                    ])
                },
                {
                    "role": "system",
                    "content": references
                },
                {
                    "role": "user",
                    "content": query
                }
            ]

            response = chat_api.get_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            return clean_clova_response(response)
        else:
            return "검색 결과가 없습니다."
        
    except Exception as e:
        print(f"Error: {str(e)}")