import json

def extract_key_sections(full_text: str, lang) -> dict:
    """전체 텍스트에서 필요한 섹션들을 추출"""
    SECTION_KEYWORDS = {
        'abstract': {
            'en': ['abstract'],
            'korean': ['초록', '요약', 'abstract']
        },
        'introduction': {
            'en': ['introduction', '1.', 'i.'],
            'korean': ['서론', 'introduction', '1.', 'i.']
        }
    }

    sections = {
        'title_section': '',
        'abstract_section': '',
        'introduction_section': ''
    }

    text_lines = full_text.split('\n')
    current_section = None

    for line in text_lines:
        line_lower = line.strip().lower()

        if any(keyword in line_lower for keyword in SECTION_KEYWORDS['abstract'][lang]):
            current_section = 'abstract_section'
            continue
        elif any(keyword in line_lower for keyword in SECTION_KEYWORDS['introduction'][lang]):
            current_section = 'introduction_section'
            continue
            
        if not sections['title_section'] and len(line.strip()) > 0:
            sections['title_section'] += line + '\n'
            
        if current_section:
            sections[current_section] += line + '\n'

    return sections

def extract_paper_metadata(paper_text: str, completion_executor, lang) -> dict:
    sections = extract_key_sections(paper_text, lang)

    print("=== Sections Content ===")
    print("Title Section:")
    print(sections['title_section'])
    print("\nAbstract Section:")
    print(sections['abstract_section'])
    
    default_metadata = {
        "title": "",
        "authors": "",
        "year": 2025,
        "abstract": ""
    }

    prompt = """다음 논문 텍스트에서 제목, 저자, 연도, 초록만 추출해서 아래 형식의 JSON으로 반환하세요.
    다른 정보는 절대 포함하지 마세요.
    
    반드시 아래 형식을 지켜주세요:
    {
        "title": "논문의 제목",
        "authors": "저자 이름들",
        "year": 발행연도,
        "abstract": "논문의 초록"
    }
    
    입력 텍스트:
    제목 섹션:
    {title_section}
    
    초록 섹션:
    {abstract_section}
    """
    
    try:
        response = completion_executor.execute({
            "messages": [
                {
                    "role": "system", 
                    "content": "당신은 논문 메타데이터 추출 전문가입니다. 요청된 형식의 JSON만 정확히 반환하세요."
                },
                {
                    "role": "user", 
                    "content": prompt.format(
                        title_section=sections['title_section'],
                        abstract_section=sections['abstract_section']
                    )
                }
            ]
        })
        
        if response and 'content' in response:
            content = response['content'].strip()
            
            # 마크다운 코드 블록 제거
            if '```' in content:
                content = content.split('```')
                if 'json' in content:
                    content = content.replace('json', '', 1)
                content = content.strip()
            
            # JSON 시작과 끝 부분만 추출
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx > 0:
                content = content[start_idx:end_idx]
            
            try:
                metadata = json.loads(content)
                # 필수 필드 확인 및 기본값으로 대체
                for key in default_metadata:
                    if key not in metadata or not metadata[key]:
                        metadata[key] = default_metadata[key]
                return metadata
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 실패: {e}")
                print(f"정제된 content: {content}")
                return default_metadata
    except Exception as e:
        print(f"메타데이터 추출 중 에러 발생: {str(e)}")
        return default_metadata