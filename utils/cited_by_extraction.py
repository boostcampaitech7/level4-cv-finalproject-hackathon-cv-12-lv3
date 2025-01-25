import requests
import os
import json
import re
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, ".."))
from config.config import GOOGLE_SCHOLAR_API_KEY

def get_cited_by_papers(query, topk=5):
    api_key = GOOGLE_SCHOLAR_API_KEY  # SearchApi API 키 설정
    if not api_key:
        print("SEARCHAPI_API_KEY environment variable not set.")
        exit()

    url = "https://www.searchapi.io/api/v1/search"
    # 1. 논문 고유 ID를 검색하는 최초 쿼리
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": api_key,
    }

    response = requests.get(url, params=params)
    results = response.json()

    if response.status_code != 200:
        print(f"Error (Target Paper Search): {results.get('error', 'Unknown error')}")
        return None

    if "organic_results" not in results or not results["organic_results"]:
        print("No organic results found for target paper.")
        return None

    first_result = results["organic_results"][0]

    cited_by_id = None

    if "inline_links" in first_result and "cited_by" in first_result["inline_links"]:
        cited_by_url = first_result["inline_links"]["cited_by"]["link"]
        match = re.search(r"cites=([^&]+)", cited_by_url)
        if match:
            cited_by_id = match.group(1)

    if not cited_by_id:
        print("Cited by information not found for this paper.")
        return None

    # 2. 이 논문을 인용한 논문을 요청하는 쿼리
    cited_by_params = {
        "engine": "google_scholar",
        "cites": cited_by_id,
        "api_key": api_key,
    }

    cited_by_response = requests.get(url, params=cited_by_params)
    cited_by_results = cited_by_response.json()

    if cited_by_response.status_code != 200:
        print(f"Error (Cited By Search): {cited_by_results.get('error', 'Unknown error')}")
        return None
    elif "organic_results" in cited_by_results:
        return sorted(cited_by_results["organic_results"], key=lambda x: int(x.get("inline_links", {}).get("cited_by", {}).get("total", 0)), reverse=True)[:topk]
    else:
        print("No cited by results found.")
        return None

if __name__ == "__main__":
    target_paper_query = "Attention is all you need"  # 타겟 논문 검색어

    cited_by_papers = get_cited_by_papers(target_paper_query)

    if cited_by_papers:
        print("\n타겟 논문을 인용한 논문 상위 5개:")
        for paper in cited_by_papers:
            title = paper.get('title')
            cited_by_count = paper.get('inline_links', {}).get('cited_by', {}).get('total', 0)
            print(f"- {title}: {cited_by_count}회 인용")
        # JSON 파일로 저장
        with open("related_works.json", "w", encoding="utf-8") as json_file:
            json.dump(cited_by_papers, json_file, ensure_ascii=False, indent=4)
        print("\n상위 5개 논문을 JSON 파일로 저장하였습니다.")
    else:
        print("No cited by results found.")