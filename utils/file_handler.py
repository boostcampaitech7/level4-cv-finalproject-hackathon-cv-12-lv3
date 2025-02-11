import json
import cv2
import numpy as np

from PIL import Image


def save_image(image, path):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image.save(path)


def save_json(data, path):
    # JSON 파일로 저장
    with open(path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
        print(f"{path} 경로에 json 파일 저장!")


def save_text(text, path):
    with open(path, 'w') as f:
        f.write(text)


def save_image(image, path):
    if isinstance(image, np.ndarray):
        cv2.imwrite(path, image)
    elif isinstance(image, Image.Image):
        image.save(path)
    else:
        raise TypeError(
            f"Invalid Input Type : Expected an instance of Image.Image or np.ndarray, but received{type(image)}")


def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"JSON 파일 로드 중 에러 발생: {str(e)}")
        raise
