import google.generativeai as genai
import torch
import random
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from diffusers import StableDiffusionPipeline


# NLP 모델 로드 (문장 임베딩을 위한 모델)
nlp = SentenceTransformer('all-MiniLM-L6-v2')

themes = [
        "A futuristic city skyline at sunset, glowing neon signs, cyberpunk atmosphere.",
        "A giant library made entirely of glass, where bookshelves form an intricate maze of glowing texts.",
        "A floating dream world where books transform into staircases leading to the sky.",
        "An AI-powered utopia where humanoid robots and humans collaborate in harmony, under a neon sky."
        "A futuristic data temple where AI-generated thoughts are stored as floating holograms.",
        "A futuristic research lab where robotic arms craft new ideas in glowing glass tubes."
        "illustration for a scientists at work in futuristic environment, 60's pulp magazine style, watercolor painting style"
        "A modern digital cover for an AI assistant service. The image features a futuristic AI robot interacting with chat bubbles, sales graphs, and notifications, symbolizing automation and increased revenue"
        "A vast cosmic nebula with floating islands, deep space elements, and magical energy sources.",
        "human brain, many electric light bulbs are attached to it on wires. background is blue"
        "minion with a turned on lightbulb over their head signifying a new idea, white background"
        "A cover for a book on chemistry, rocks and minerals, science and modern design technology"
        "a group of students doing a discussion of academic papers in the realm of AI/Data Science. It should be in a university room before a big monitor with the journal article on and a whiteboard"
        "A neon-lit AI supercomputer with an infinite number of interconnected circuits forming fractal-like designs."
        "A steampunk-inspired library with mechanical book sorters, vintage aesthetics, and warm candlelight.",
        "An abstract sci-fi environment with intricate geometric shapes, floating data streams, and AI holograms.",
        "A surreal, vibrant landscape where knowledge takes a physical form, like rivers of light and floating symbols.",
        "A mind-mapping visualization where thought processes appear as interconnected glowing lines."
        "A floating island where glowing crystal flowers bloom under a gentle golden sunrise."
    ]

def get_best_visual_theme(abstract_text):
    # 논문 요약 및 테마 문장을 임베딩 (벡터 변환)
    abstract_embedding = nlp.encode([abstract_text])
    theme_embeddings = nlp.encode(themes)

    # 유사도 계산 (Cosine Similarity)
    similarities = cosine_similarity(abstract_embedding, theme_embeddings)[0]

    # 유사도가 높은 상위 3개 테마 선택
    top_3_indices = similarities.argsort()[-3:][::-1]
    top_3_themes = [themes[i] for i in top_3_indices]

    # 상위 3개 테마 중 랜덤으로 하나 선택
    selected_theme = random.choice(top_3_themes)
    print(f"🎨 Top 3 Similar Themes: {top_3_themes}")
    print(f"🎨 Selected Theme: {selected_theme}")

    return selected_theme

def generate_prompt(abstract_text):
    best_theme = get_best_visual_theme(abstract_text)

    prompt = (
        f"An artistic and high-quality audiobook cover, visually stunning, ultra-detailed, "
        f"fantasy-inspired, cinematic lighting, highly creative. "
        f"Main theme: {best_theme} "
        f"Concept art, masterpiece, professional illustration, vibrant colors, atmospheric depth."
    )

    return prompt

def generate_audiobook_cover(file_path):
    """
    논문 요약을 기반으로 Stable Diffusion을 사용하여 애니메이션 스타일 오디오북 커버 생성.
    """
    # .txt 파일에서 논문 요약 읽어오기
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            abstract_text = file.read().strip()
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return

     # 프롬프트 생성
    prompt = generate_prompt(abstract_text)
    print(f"🎨 Generated Prompt: {prompt}")

    # 네거티브 프롬프트 (텍스트 삽입 방지 및 품질 개선)
    negative_prompt = (
        "text, watermark, logo, signature, blurry, low quality, distorted, "
        "photorealistic, hyper-realistic, 3D-rendered, stock photo, ultra-realistic, nsfw, nudity, explicit, suggestive"
    )

    # Stable Diffusion 모델 로드
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, safety_checker=None
    )
    pipe.to("cuda")

    # Stable Diffusion 실행
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=50, guidance_scale=7).images[0]

    # 이미지 저장
    image.save("audiobook_cover.png")
    print("✅ Audiobook cover saved as 'audiobook_cover.png'")
    
if __name__ == "__main__":
    file_path = "/data/ephemeral/home/YJ/level4-cv-finalproject-hackathon-cv-12-lv3/utils/summary.txt"  # 읽어올 논문 요약 파일 경로
    generate_audiobook_cover(file_path)
