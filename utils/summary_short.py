# import os
# import sys
# import networkx as nx

# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# sys.path.insert(0, os.path.abspath(
#     '/Users/haneol/Documents/Coding/level4-cv-finalproject-hackathon-cv-12-lv3/'))

# model = SentenceTransformer("dragonkue/bge-m3-ko")


# def extractive_summarization(sentences, sentence_embeddings=None, model=model, top_n=3):
#     """
#     문서의 문장들을 요약하는 함수 (추출적 요약)
#     TextRank만 사용하여 상위 n개의 문장 추출
#     이미 임베딩이 제공되면 이를 활용하고, 그렇지 않으면 내부에서 임베딩을 계산함
#     """

#     # 문장 임베딩과 TextRank 점수 계산
#     def calculate_textrank(similarity_matrix):
#         graph = nx.from_numpy_array(similarity_matrix)
#         scores = nx.pagerank(graph)
#         return scores

#     def calculate_combined_scores(sentences, sentence_embeddings):
#         # 문장 간 유사도 계산 (임베딩 벡터 간 코사인 유사도)
#         similarity_matrix = cosine_similarity(sentence_embeddings)

#         # TextRank 적용하여 문장 중요도 점수 계산
#         textrank_scores = calculate_textrank(similarity_matrix)

#         return textrank_scores

#     def extract_top_sentences(scores, sentences, n=3):
#         sorted_sentences = sorted(
#             scores.items(), key=lambda x: x[1], reverse=True)
#         return [sentences[i[0]] for i in sorted_sentences[:n]]

#     # 임베딩이 주어진 경우, 임베딩을 사용하고 그렇지 않으면 새로 계산
#     if sentence_embeddings is None:
#         sentence_embeddings = model.encode(sentences)

#     # TextRank 점수 계산
#     textrank_scores = calculate_combined_scores(sentences, sentence_embeddings)

#     # 상위 n개의 문장 추출
#     top_sentences = extract_top_sentences(textrank_scores, sentences, n=top_n)

#     return top_sentences


def abstractive_summarization(extracted_sentences, completion_executor):

    messages1 = [
        {"role": "system", "content":
                 """
           당신은 학술 논문 요약 전문가로, 아래 구조에 따라 2000토큰 이상의 상세한 분석을 생성해야 합니다. 제공된 내용뿐만 아니라 가지고 있는 지식 모두를 활용하여 작성해주세요.

            [요구사항]
            **주목할 만한 요약** (2000토큰 이상)
            - 기술적 혁신성: 핵심 알고리즘/방법론의 차별점
            - 학문적 기여: 이론적 확장성 또는 새로운 연구 패러다임 제시
            - 실용적 가치: 실제 적용 사례 및 산업계 파급효과
            - 비교 분석: 기존 연구 대비 우월성 (정량적 지표 제시)
            - 중요 태그: 논문을 기반으로한 중요 키워드로 태그를 반드시 영어로 생성

            [작성 원칙]
            - 다층적 구조: 메타 인지적 관점에서 개념 > 방법 > 결과 > 영향력 계층화
            - 학제간 연계: 타 분야 연구자도 이해할 수 있는 크로스-도메인 설명
            - 미래 지향적: 기술 발전 방향성과 잠재적 진화 경로 제시
            - 비판적 시각: 방법론의 한계와 개선 필요사항 반드시 기술
            
            예시:
            본 연구는 **심층 강화학습(Deep Reinforcement Learning, DRL)**을 기반으로 한 로봇 제어 시스템을 제안하며, 복잡한 물리적 환경에서의 로봇 운동 제어 문제를 혁신적으로 해결했습니다. 기술적 혁신성 측면에서, 제안 모델은 **계층적 강화학습(Hierarchical Reinforcement Learning, HRL)** 프레임워크를 도입하여 고차원 작업을 하위 작업으로 분해하고, 각 하위 작업을 효율적으로 학습하는 방식을 제시합니다. 특히, **메타 강화학습(Meta-Reinforcement Learning)**을 접목하여 다양한 환경에서의 일반화 능력을 극대화했습니다. 이는 기존의 단일 정책(Single Policy) 기반 접근법이 환경 변화에 취약했던 한계를 극복합니다.

            학문적 기여로는 강화학습의 이론적 체계를 확장한 점이 두드러집니다. 본 논문은 **다중 시간 학습(Multi-Timescale Learning)** 이론을 제안하며, 장기적 목표와 단기적 행동 간의 균형을 수학적으로 모델링합니다(수식 4 참조). 이를 통해 로봇이 복잡한 작업을 단계적으로 학습할 수 있는 이론적 근거를 마련했습니다. 실험 설계에서는 MuJoCo 및 OpenAI Gym 환경에서의 비교 평가를 수행했으며, 평균 작업 성공률이 기존 대비 **15.3% 향상**되었습니다(표 3).

            실용적 가치 측면에서, 이 모델은 산업용 로봇 및 자율주행 차량의 제어 시스템에 직접 적용 가능합니다. 특히, **시뮬레이션에서 실제 환경으로의 전이 학습(Sim-to-Real Transfer Learning)** 성능이 뛰어나며, 실제 로봇 팔을 이용한 물체 조작 실험에서 **92.5%의 작업 성공률**을 달성했습니다(그림 6). 이는 제조업 및 물류 분야에서의 자동화 비용을 크게 절감할 수 있는 잠재력을 시사합니다.

            비교 분석에서는 PPO(Proximal Policy Optimization), SAC(Soft Actor-Critic), TD3(Twin Delayed DDPG) 등 5가지 최신 강화학습 알고리즘과의 엄격한 비교 실험을 진행했습니다. 복잡한 작업 환경에서의 평균 보상(Reward)이 기존 대비 **18.7% 증가**했으며(그림 8), 학습 시간 또한 **30% 단축**되었습니다(표 7). 이러한 결과는 제안 모델이 효율성과 성능 면에서 우수함을 입증합니다.
            
            Keywords: #Deep Reinforcement Learning, #Robot Control, #Hierarchical Reinforcement Learning, #Meta-Reinforcement Learning, #Multi-Timescale Learning
            """
         },
        {"role": "user", "content": extracted_sentences}
    ]
    messages2 = [
        {"role": "system", "content":
                 """
          당신은 학술 논문 요약 전문가로, 아래 구조에 따라 1000토큰 이상의 상세한 분석을 생성해야 합니다. 제공된 내용뿐만 아니라 가지고 있는 지식 모두를 활용하여 작성해주세요.

           [요구사항]
            심층 분석 태그 시스템 (700토큰 이상)

            - 📚 연구 분야: [상위 분야] > [하위 분야] 계층 구조
            - 🛠 방법론: 기술적 세부사항을 3단계로 분해
            - 🔬 주요 발견: 연구의 핵심 발견을 요약하고, 그 중요성과 향후 연구에 미친 영향 설명, 실용적인 적용 가능성 및 해당 발견이 다른 분야에 어떻게 연관될 수 있는지 논의
            - 🎯 응용 분야: 실제 적용 가능한 산업 도메인
           [작성 원칙]

            - 다층적 구조: 메타 인지적 관점에서 개념 > 방법 > 결과 > 영향력 계층화
            - 학제간 연계: 타 분야 연구자도 이해할 수 있는 크로스-도메인 설명
            - 미래 지향적: 기술 발전 방향성과 잠재적 진화 경로 제시
            - 비판적 시각: 방법론의 한계와 개선 필요사항 반드시 기술
            
            예시:
                📚 연구 분야:
                - 컴퓨터 과학 > 자연어 처리 > 문서 요약
                - 인공지능 > 기계 학습 > 심층 학습
                
                🛠 방법론:
                1. 데이터 전처리:
                - 불용어 제거 및 형태소 분석을 통한 텍스트 정제
                - TF-IDF 기반 특징 추출
                2. 모델 설계:
                - LSTM(긴 단기 기억) 네트워크를 활용한 시퀀스 모델링
                - Attention Mechanism을 이용한 중요한 문장 강조
                3. 모델 학습:
                - Adam 옵티마이저를 이용한 모델 최적화
                - Cross-validation 기법을 사용하여 과적합 방지
                
                🔬 주요 발견:
                - 연구는 Transformer 기반 모델이 기존 LSTM 모델보다 문서 요약 정확도에서 우수하다는 중요한 발견을 했습니다.
                - 이 발견은 대규모 문서 요약 시스템에서 실용적인 적용 가능성을 제시하며, 추후 논문, 법률 문서 요약 등 다양한 분야에 적용할 수 있는 가능성을 여는 중요한 연구 결과입니다.
                - 또한, 이 연구는 딥러닝 기반 자연어 처리 기술이 기업의 고객 서비스나 자동화된 보고서 생성 시스템에 어떻게 영향을 미칠 수 있는지에 대한 실용적인 통찰을 제공합니다.
                
                🎯 응용 분야:
                - 법률: 계약서 요약 및 법적 문서 분석
                - 금융: 보고서 자동 생성 및 고객 응대 자동화
                - 건강 관리: 의학 논문 요약 및 환자 기록 분석
            """
         },
        {"role": "user", "content": extracted_sentences}
    ]
    request_data1 = {
        'messages': messages1,
        'topP': 0.8,
        'topK': 0,
        'maxTokens': 4096,
        'temperature': 0.5,
        'repeatPenalty': 5.0,
        'stopBefore': [],
        'includeAiFilters': True,
        'seed': 0
    }
    request_data2 = {
        'messages': messages2,
        'topP': 0.8,
        'topK': 0,
        'maxTokens': 4096,
        'temperature': 0.5,
        'repeatPenalty': 5.0,
        'stopBefore': [],
        'includeAiFilters': True,
        'seed': 0
    }
    res1 = completion_executor.execute(request_data1, stream=False)
    res2 = completion_executor.execute(request_data2, stream=False)

    res1 = res1['message']['content']
    res2 = res2['message']['content']
    return res1 + "\n" + res2
