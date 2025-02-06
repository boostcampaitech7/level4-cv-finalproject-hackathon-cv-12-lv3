import requests
import time
import json
import re

def extract_keywords(text):
    """
    주어진 텍스트에서 '#'이 앞에 붙은 키워드들을 띄어쓰기까지 포함하여 추출하는 함수.
    쉼표가 나오기 전까지를 하나의 키워드로 인식함.
    """
    hashtags = re.findall(r"#([^\n#,]+)", text)  # '#' 이후 쉼표 또는 줄바꿈 전까지 추출
    return [tag.strip() for tag in hashtags][:4]

def timeline(query_list):
    # API 엔드포인트 
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    # 최종 JSON 데이터 구조
    output_json = {
        "My Paper": {}
    }

    # 논문 제목을 요약하는 함수 (최대 30자)
    def shorten_title(title):
        return title[:30] + "..." if len(title) > 30 else title

    # 각 키워드별 논문 검색
    for query in query_list:
        print(f"\n🔍 검색 키워드: {query}")

        # API 요청 URL 구성
        url = f"{base_url}?query={query}&fields=title,year,citationCount&limit=10"
        
        # 요청 보내기
        response = requests.get(url)

        # 응답 확인
        if response.status_code == 200:
            data = response.json()
            
            # 논문을 인용 수 기준으로 내림차순 정렬 후 상위 3개 선택
            sorted_papers = sorted(data.get("data", []), key=lambda x: x["citationCount"], reverse=True)[:3]

            # 논문 데이터 저장
            paper_dict = {}
            for paper in sorted_papers:
                short_title = shorten_title(paper["title"])
                paper_dict[short_title] = {
                    "full_title": paper["title"],
                    "year": paper["year"],
                    "citation_count": paper["citationCount"]
                }

            # 키워드별 논문 저장
            output_json["My Paper"][query] = paper_dict

        else:
            print(f"API 요청 실패: {response.status_code}")

        # 속도 제한을 고려하여 1초 대기
        time.sleep(1)

    # JSON 파일 저장
    output_filename = "papers_data.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=4, ensure_ascii=False)

    # JSON 데이터 출력
    print(json.dumps(output_json, indent=4, ensure_ascii=False))
    
    return output_json

