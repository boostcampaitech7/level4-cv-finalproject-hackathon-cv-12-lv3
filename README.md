# SummarAI


## Team Member

<table align="center">
    <tr align="center">
        <td><img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00003880%2Fuser_image.png&w=1920&q=75" width="120" height="120" alt=""/></td>
        <td><img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00003955%2Fuser_image.png&w=1920&q=75" width="120" height="120" alt=""/></td>
        <td><img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00003894%2Fuser_image.png&w=1920&q=75" width="120" height="120" alt=""/></td>
        <td><img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00003885%2Fuser_image.png&w=1920&q=75" width="120" height="120" alt=""/></td>
        <td><img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00003890%2Fuser_image.png&w=1920&q=75" width="120" height="120" alt=""/></td>
        <td><img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00003872%2Fuser_image.png&w=1920&q=75" width="120" height="120" alt=""/></td>
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

<br>

## Commit Convention
1. `Feature` ✨ **새로운 기능 추가**
2. `Bug` 🐛 **버그 수정**
3. `Docs` 📝 **문서 수정**
4. `Refactor` ♻️ **코드 리펙토링**

커밋할 때 헤더에 위 내용을 작성하고 전반적인 내용을 간단하게 작성합니다.

### 예시

- `git commit -m "[#이슈번호] ✨ feat 간단하게 설명" `
- `git commit -m "[#이슈번호] 🐛 bug 간단하게 설명"`
- `git commit -m "[#이슈번호] 📝 docs 간단하게 설명" `
- `git commit -m "[#이슈번호] ♻️ refactor 간단하게 설명" `

<br/>

## Branch Naming Convention

브랜치를 새롭게 만들 때, 브랜치 이름은 항상 위 `Commit Convention`의 Header와 함께 작성되어야 합니다.

### 예시

- `Feature/~~~`
- `Refactor/~~~`
