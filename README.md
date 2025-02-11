# SummarAI

<div align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/b7f7a88b-0f2d-46e1-8831-cca7cbf18b0b">
        <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/c0a29b43-f7da-405a-9a26-c6319d16c6e1">
        <img alt="IMAGE" src="https://github.com/user-attachments/assets/c0a29b43-f7da-405a-9a26-c6319d16c6e1" alt="thumbnail">
    </picture>
</div>

## Project Introduction

📌 논문 기반으로 한 챗봇 서비스 🫧 🏠

📌 논문 학습을 위한 A to Z 학습 콘텐츠 🌟

📌 기존 서비스와 차별화된 SummarAI 🙌

📌 논문 내용으로 블로그 글 작성 ☕

### 🔍 **Background**

대학원생과 연구자, 기업 등 문서 기반 정보를 다루는 사용자들은 **방대한 논문과 문서에서 필요한 정보를 신속하게 파악해야 하지만, 기존 방식은 시간과 노력이 많이 요구됩니다**. 특히 수동으로 문서를 읽고 요약하는 것은 **비효율적이고 정보 누락 가능성이 존재합니다.**
이를 해결하기 위해 **AI를 활용한 논문 요약 및 문서 이해 챗봇**을 개발하여 **문서 자동 분석, 핵심 요약, 실시간 질의응답 기능**을 제공할 계획입니다 !

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

## Service Architecture

우선 논문 PDF 파일을 업로드하여 텍스트 및 이미지 데이터를 추출합니다. 벡터 데이터베이스에서는 추출된 데이터를 벡터 형태로 저장하고, 사용자 질의와 유사한 벡터를 검색합니다.

<div align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/24ba6392-8544-4bbc-a4c3-da27922361ab">
        <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/5220c846-2858-4521-8d8a-73ae1ec235dc">
        <img alt="IMAGE" src="https://github.com/user-attachments/assets/5220c846-2858-4521-8d8a-73ae1ec235dc" alt="thumbnail">
    </picture>
</div>

챗봇 인터페이스를 통해 사용자는 논문에 대한 질문을 하고, Vector DB에서 검색된 답변을 제공받습니다. 추가로, AI 모델의 답변에 대한 안전성 검사를 통해 부적절한 답변을 필터링합니다.

![유튜브 링크]()
<img width="1147" alt="image" src="https://github.com/user-attachments/assets/d11bdc0c-789d-4577-894c-45e91cf6acd8" />

요약 페이지에서는 논문 요약 정보, 연구 분야, 방법론, 결과 등을 제공하고, 유사 논문을 추천합니다.

![유튜브 링크]()
<img width="982" alt="image" src="https://github.com/user-attachments/assets/1c330dd1-457c-46cf-83cf-741a6509557f" />

오디오 페이지에서는 논문 내용을 요약한 오디오 콘텐츠와 스크립트 텍스트를 제공합니다.

![유튜브 링크]()
<img width="1023" alt="image" src="https://github.com/user-attachments/assets/bccbbd56-4953-4638-8564-cdf0fde5064a" />

<br/>

## Model Architecture

모델 파이프라인은 크게 PDF-to-Text 모듈, 요약 모듈로 나뉩니다.

- PDF-to-Text 모듈은 PDF를 입력으로 받아, 텍스트, 이미지, 테이블로 분리하고, 각각을 OCR 모델을 통해 처리합니다.
- Layout Analysis: DocLayout-YOLO 모델 사용
- Text OCR: Table Transformer
- Image OCR: **DeepSeek-VL**
- Formula OCR: YOLO, PaddleOCR, Mathematical Formula Recognition(MFR)
- 이후 정확성 증가를 위해 번역 모델을 추가로 사용하며, 번역 내용도 추가적인 출력으로 제공합니다.

<div align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/c9c4adb2-1c7e-4625-936e-b27edbd7a54c">
        <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/3394684f-caeb-4d85-896a-c23d8102a9f5">
        <img alt="IMAGE" src="https://github.com/user-attachments/assets/3394684f-caeb-4d85-896a-c23d8102a9f5" alt="thumbnail">
    </picture>
</div>

<br/>

## AI Safety

프롬프트 엔지니어링을 통해 혐오, 차별적 발언을 필터링하고 사용자의 개인 정보를 보호합니다.
![image](https://github.com/user-attachments/assets/9a011f78-14c4-4565-9a2a-203a2ac2d5c7)

직접 구성한 데이터셋을 통해 비교해본 결과, 전반적으로 더 성능이 우수함을 알 수 있었습니다.
![image](https://github.com/user-attachments/assets/e39a780f-e62b-4553-8309-8092e105186d)

<br/>

## Comparison with other services

기존 서비스와의 기능 비교 테이블입니다.
![image](https://github.com/user-attachments/assets/c9775f16-7cdf-4063-8531-df62731b531e)

기존 서비스와의 RAG 성능 리더보드 비교 테이블입니다.
![image](https://github.com/user-attachments/assets/ac772ca2-42d0-40b9-a7a4-b564465ae916)

직접 구축한 논문 RAG 데이터셋 비교 테이블입니다.
![image](https://github.com/user-attachments/assets/04133930-5152-4b58-865f-2d977ee4b084)

<br/>

## Result

![시연 영상.gif]()

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

## Dependency

         psycopg2-binary
         sentence-transformers
         mlflow
         langchain
         langchain_openai
         langchain_anthropic
         langchain_core
         tiktoken
         transformers>=4.37.0
         pillow
         optimum[onnxruntime]
         doclayout_yolo
         PyMuPDF
         frontend
         tools
         python-dotenv
         timm
         paddleocr
         einops
         ninja
         networkx
         scikit-learn
         bert-extractive-summarizer
         fitz

<br/>

## Refernece

[노션]()

[랩업리포트]()

[발표자료]()

[발표영상]()

<br/>
