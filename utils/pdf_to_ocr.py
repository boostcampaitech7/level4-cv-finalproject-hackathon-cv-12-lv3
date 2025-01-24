import fitz
import io
import torch
from PIL import Image
from api import OCRAPIExecutor


def pdf_to_image(pdf_path):
    """
    PDF 파일을 이미지로 변환하는 함수
    :param pdf_path: PDF 파일 경로
    :return: 이미지 리스트(PIL Image 객체)
    """
    pages = fitz.open(pdf_path)
    images = []
    for page in pages:
        pix = page.get_pixmap(dpi=300)
        img_data = pix.tobytes(output='jpg', jpg_quality=200)
        images.append(Image.open(io.BytesIO(img_data)).convert('RGB'))
    return images


def images_to_text(image, ocr_host, ocr_secret_key):
    """
    이미지 파일에서 텍스트를 추출하고 후처리하여 반환하는 함수
    :param image: PIL Image 객체
    :param ocr_host: OCR을 위한 호스트 (임시)
    :param ocr_secret_key: OCR을 위한 시크릿 키 (임시)
    :return: OCR 결과로 추출된 텍스트 (문자열)
    """
    ocr_api = OCRAPIExecutor(ocr_host, ocr_secret_key)
    ocr_result = ocr_api.execute_ocr(image)

    if not isinstance(ocr_result, dict) or 'images' not in ocr_result:
        raise ValueError("Invalid OCR result format")
    text = " ".join([field['inferText']
                    for field in ocr_result['images'][0]['fields']])
    return text


def select_device(device):
    if device is not None:
        return device

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    return device
