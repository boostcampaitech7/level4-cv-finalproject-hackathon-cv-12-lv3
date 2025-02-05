from api import OCRAPIExecutor
from pdf2image import convert_from_path

def images_to_text(image, ocr_host, ocr_secret_key):
    # TODO 구현한 pdf2text 이식
    # TODO Table OCR 이전에 crop한 image patch에 margin을 더해주면 성능 향상이 된다는 정보
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

def pdf_to_image(pdf_path):
    """
    PDF 파일을 이미지로 변환하는 함수
    :param pdf_path: PDF 파일 경로
    :return: 이미지 리스트(PIL Image 객체)
    """
    images = convert_from_path(pdf_path)
    return images