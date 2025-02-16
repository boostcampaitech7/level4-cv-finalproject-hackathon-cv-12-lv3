# SummarAI

<div align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/b7f7a88b-0f2d-46e1-8831-cca7cbf18b0b">
        <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/c0a29b43-f7da-405a-9a26-c6319d16c6e1">
        <img alt="IMAGE" src="https://github.com/user-attachments/assets/c0a29b43-f7da-405a-9a26-c6319d16c6e1" alt="thumbnail">
    </picture>
</div>

## Demo Video

각 기능별 시연 영상을 통해 SummarAI의 주요 기능을 확인해 보세요.  
이미지를 클릭하면 유튜브로 이동합니다.

[![전체 시연영상](https://github.com/user-attachments/assets/9855951e-c19a-4afd-8d7e-dfb7ea671f46)](https://youtu.be/fq43BjR9tas)  
SummarAI의 모든 기능을 한 번에 확인하고 싶다면 위의 영상을 참고하세요.

<details>
    <summary> <h3>Chat Service</h3></summary>

[![Main Page 시연 영상](https://github.com/user-attachments/assets/d11bdc0c-789d-4577-894c-45e91cf6acd8)](https://youtu.be/bYKZcrlty_E)  
SummarAI의 챗봇 인터페이스를 통해 사용자는 논문에 대한 질문을 자유롭게 입력할 수 있습니다.

- Vector DB를 활용해 적절한 논문 정보를 검색 및 제공
- AI 모델을 통해 질문에 대한 요약 답변 생성
- AI 안전성 검사를 통해 부적절한 답변 필터링
</details>
<details>
    <summary> <h3>Summary Service</h3></summary>

[![Summary Page 시연 영상](https://github.com/user-attachments/assets/1c330dd1-457c-46cf-83cf-741a6509557f)](https://youtu.be/YGoFEJb3DwQ)  
논문의 핵심 정보를 한눈에 확인할 수 있는 요약 페이지를 제공합니다.

- 논문의 제목, 연구 분야, 방법론, 실험 결과 요약
- 연구자의 빠른 이해를 돕기 위한 자동 요약 생성
- 유사한 연구 논문 추천 기능 제공
</details>
<details>
    <summary> <h3>Audio Service</h3></summary>

[![Audio Page 시연 영상](https://github.com/user-attachments/assets/bccbbd56-4953-4638-8564-cdf0fde5064a)](https://youtu.be/O7F5wzoCZag)
논문 내용을 오디오 콘텐츠로 변환하여 제공합니다.

- 논문 요약을 TTS(Text-to-Speech)로 변환하여 제공
- 논문 내용을 음성으로 청취 가능, 이동 중에도 활용 가능
- 오디오 스크립트 제공으로 빠른 복습 가능
</details>

## Project Introduction

📌 논문 기반으로 한 챗봇 서비스 🏠

📌 논문 학습을 위한 A to Z 학습 콘텐츠 🌟

📌 기존 서비스와 차별화된 SummarAI 🙌

📌 논문 내용으로 블로그 글 작성 ☕

<br/>

## 🔍 **Background**

대학원생과 연구자, 기업 등 문서 기반 정보를 다루는 사용자들은 **방대한 논문과 문서에서 필요한 정보를 신속하게 파악해야 하지만, 기존 방식은 시간과 노력이 많이 요구됩니다**. 특히 수동으로 문서를 읽고 요약하는 것은 **비효율적이고 정보 누락 가능성이 존재합니다.**
이를 해결하기 위해 **AI를 활용한 논문 요약 및 문서 이해 챗봇**을 개발하여 **문서 자동 분석, 핵심 요약, 실시간 질의응답 기능**을 제공할 계획입니다 !

<br/>

## Service Architecture

![Service Architecture](https://github.com/user-attachments/assets/e291dc67-fb84-4ae3-81e7-1de7be0802b0)  
SummarAI는 사용자의 질의에 적절한 답변을 생성하는 AI 파이프라인을 구축하여 논문 검색과 요약을 지원합니다.

1️⃣ 사용자 질의 입력 및 전처리

- 사용자가 질문을 입력하면, BackEnd에서 SubQuery GAN과 Hyper Colva X를 활용하여 질의 강화를 수행합니다.
- 이를 통해 보다 정확한 검색 결과를 도출할 수 있도록 쿼리를 개선합니다.

2️⃣ 논문 벡터 임베딩 및 검색

- 질의는 Vector Embedding으로 변환 후, Vector DB에 저장된 논문 데이터와 유사도를 비교하여 검색됩니다.
- Vector DB 구축을 위해 논문 본문을 Text Chunking하여 저장하며, 검색 성능을 최적화합니다.

3️⃣ 검색 결과 재정렬 및 응답 생성

- 검색된 문서들은 Rerank 과정을 거쳐 가장 관련성이 높은 결과를 상위에 배치합니다.
- 최종적으로 Hyper Colva X 모델이 논문 데이터를 기반으로 사용자 질문에 대한 응답을 생성합니다.

<br/>

## Model Architecture

<div align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/c9c4adb2-1c7e-4625-936e-b27edbd7a54c">
        <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/3394684f-caeb-4d85-896a-c23d8102a9f5">
        <img alt="IMAGE" src="https://github.com/user-attachments/assets/3394684f-caeb-4d85-896a-c23d8102a9f5" alt="thumbnail">
    </picture>
</div>

SummarAI의 모델 파이프라인은 PDF-to-Text 모듈과 요약 모듈로 구성됩니다.

- PDF-to-Text 모듈은 PDF를 입력으로 받아, 텍스트, 이미지, 테이블로 분리하고, 각각을 OCR 모델을 통해 처리합니다.
- Layout Analysis: DocLayout-YOLO 모델 사용
- Text OCR: Table Transformer
- Image OCR: **DeepSeek-VL**
- Formula OCR: YOLO, PaddleOCR, Mathematical Formula Recognition(MFR)
- 이후 정확성 증가를 위해 번역 모델을 추가로 사용하며, 번역 내용도 추가적인 출력으로 제공합니다.

<br/>

## AI Safety

SummarAI는 프롬프트 엔지니어링을 활용하여 안전성을 강화하고, 사용자의 개인 정보를 보호하는 시스템을 구축하였습니다.  
아래와 같은 AI Safety 아키텍처를 통해 SummarAI는 신뢰할 수 있는 논문 요약 및 질의응답 서비스를 제공합니다.

<br/>

![image](https://github.com/user-attachments/assets/113692a7-91fe-47bb-b9f0-bc4fc69fe4a7)  
1️⃣ 부적절한 콘텐츠 필터링

- 혐오 표현, 차별적 발언, 유해 콘텐츠를 탐지 및 차단하여 안전한 AI 환경을 제공합니다.
- 프롬프트 엔지니어링 기법을 적용하여 사용자 질의를 정제하고, 부적절한 질문에 대한 응답을 제한합니다.

2️⃣ 개인정보 보호 및 데이터 안전성 확보

- AI 모델이 개인정보를 포함한 응답을 생성하지 않도록 제어하는 필터링 시스템을 적용하였습니다.

3️⃣ 할루시네이션에 대한 대비

- 잘못된 정보에 대한 Query에 대해서 정정하여 답변이 가능한가?
- RAG에 포함되어 있지 않은 내용에 대해서 Query에 정정된 답변을 제공할 수 있는가?
- 위의 두 항목을 고려하는 방식으로 할루시네이션에 대해서도 일부 해결하였습니다.

<br/>

![image](https://github.com/user-attachments/assets/e39a780f-e62b-4553-8309-8092e105186d)  
직접 구축한 AI 안전성 평가 데이터셋을 활용하여 기존 서비스와 비교한 결과,  
SummarAI가 전반적으로 더 높은 필터링 성능과 안전성을 제공함을 확인하였습니다.

<br/>

## Comparison with other services

SummarAI는 기존 논문 요약 및 분석 서비스와 차별화된 기능을 제공합니다.  
아래 비교 테이블을 통해 SummarAI의 강점을 확인해보세요!

<br/>

![Image](https://github.com/user-attachments/assets/c9775f16-7cdf-4063-8531-df62731b531e)  
기존 서비스에서 각 제공하는 기능들을 비교하는 테이블입니다.  
SummarAI는 타 서비스에서 제공하는 기능들을 한 곳에 모아 제공하는 논문 학습 A-Z 서비스입니다!

<br/>

![Image](https://github.com/user-attachments/assets/ac772ca2-42d0-40b9-a7a4-b564465ae916)  
SummarAI는 RAG 기반으로 논문 검색 및 답변을 제공합니다.  
기존 서비스 대비 정확한 검색 결과와 높은 응답 품질을 보이며, 이를 위해 다양한 방법론들을 적용하였습니다.

<br/>

![Image](https://github.com/user-attachments/assets/04133930-5152-4b58-865f-2d977ee4b084)  
SummarAI는 영어 및 한글 논문에 최적화된 RAG 평가 데이터 셋을 자체 구축하여, SummarAI의 성능을 객관적으로 평가하고자 하였습니다.  
타 서비스와 비교하였을 때 우수한 성능을 보이며, 이러한 정량적 평가를 기반으로 SummarAI의 신뢰성을 높이고자 합니다!

<br/>

## Team Member

적극적인 열정과 개인의 개성을 우선시하는 **TEAM 데만추**🚀입니다!

우리가 만들어가는 프로젝트가 단순한 결과물이 아니라, **현실적인 가치를 제공하는 혁신적인 서비스**가 될 수 있도록 끝까지 최선을 다하겠습니다! 🚀🔥

✅ **자유로운 소통**: 누구나 아이디어를 제안하고 피드백을 주고받으며 성장하는 환경

✅ **주도적인 문제 해결**: 단순한 실행이 아닌, 문제의 본질을 파악하고 최적의 솔루션을 찾는 과정 중시

✅ **후회 없는 도전**: 완성도 높은 프로젝트를 위해 끊임없이 실험하고 개선하며 최고의 결과를 추구

<table align="center">
    <tr align="center">
        <td><img src="https://github.com/user-attachments/assets/7e481beb-d803-4eb9-946a-010a730031e4" width="120" height="120" alt=""/></td>
        <td><img src="https://github.com/user-attachments/assets/8d0b99b3-6dc7-41aa-a75e-010a489f661b" width="120" height="120" alt=""/></td>
        <td><img src="https://github.com/user-attachments/assets/548b3d5d-f417-492a-a193-6315d6078bed" width="120" height="120" alt=""/></td>
        <td><img src="https://github.com/user-attachments/assets/7b3acbc2-be66-4ce3-bd5e-48bdd9a5fc4d" width="120" height="120" alt=""/></td>
        <td><img src="https://github.com/user-attachments/assets/7d92815a-8169-429f-aac4-a98c5b7071e9" width="120" height="120" alt=""/></td>
        <td><img src="https://github.com/user-attachments/assets/79c93231-e458-44d6-b2f0-a626008ad917" width="120" height="120" alt=""/></td>
    </tr>
    <tr align="center">
        <td><a href="https://github.com/lkl4502" target="_blank">오홍석</a></td>
        <td><a href="https://github.com/lexxsh" target="_blank">이상혁</a></td>
        <td><a href="https://github.com/yejin-s9" target="_blank">이예진</a></td>
        <td><a href="https://github.com/Haneol-Kijm" target="_blank">김한얼</a></td>
        <td><a href="https://github.com/PGSammy" target="_blank">조재만</a></td>
        <td><a href="https://github.com/oweixx" target="_blank">방민혁</a></td>
    </tr>
    <tr align="center">
        <td>T7208</td>
        <td>T7221</td>
        <td>T7225</td>
        <td>T7138</td>
        <td>T7253</td>
        <td>T7158</td>
    </tr>
</table>

<br/>

## Project Timeline

![image](https://github.com/user-attachments/assets/bc81e9a4-5f9a-4dd4-a6fe-d9cf71095586)

<br/>

## Install

Prerequisites

Download Links

- **CUDA Toolkit 12.0:** [CUDA Toolkit 12.0 Downloads](https://developer.nvidia.com/cuda-12-0-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local)
- **cuDNN 9.7.0:** [cuDNN 9.7.0 Downloads](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local)

Install PaddleOCR GPU Version
To install the GPU version of PaddleOCR with CUDA 12.0 support, run the following command:

```
python -m pip install paddlepaddle-gpu==2.6.1.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

Install Required Python Packages
Install the necessary Python packages by running:

```
pip install -r requirements.txt
```

<br/>

## Development Environment

- Tesla V100 32GB & 4EA
- Naver Cloud Platform(NCP)

<br/>

## Refernece

[Notion](https://www.notion.so/SummarAI-17e9d71d841180019e4fec63ed0e5ef7?pvs=4)

[Wrap-UP Report](https://drive.google.com/file/d/1EWkmIMJUIt7Us3AV0ApPt7SP2SLispn0/view?usp=sharing)

[발표 자료](https://docs.google.com/presentation/d/18uIdExpUtLEqidLbUk1QPIW2W_cHzowYZvR8cgwd1Xk/edit?usp=sharing)

[발표영상](https://youtu.be/yDhZjUy1yjg)

<br/>
