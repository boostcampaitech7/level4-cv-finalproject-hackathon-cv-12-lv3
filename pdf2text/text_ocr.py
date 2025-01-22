from paddleocr import PaddleOCR
import numpy as np
import os
class TextOCR :
    name = 'paddleocr'

    def __init__(self, languages):
        super().__init__()
        lang = languages[0] if len(languages) > 0 else 'en'
        self.ocr_engine = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=True)

    def ocr(self, img: np.ndarray):
        """
        텍스트 박스만 감지.
        """
        result = self.ocr_engine.ocr(img, det=True, rec=True, cls=False)

        return result

    # def recognize_only(self, img: np.ndarray, **kwargs):
    #     """
    #     감지된 텍스트 영역에서 텍스트만 인식.
    #     """
    #     result = self.ocr_engine.ocr(img, det=False, rec = True, cls=False)
    #     if not result or not result[0]:  # None 또는 빈 결과 처리
    #         return {'text': '', 'score': 0.0}

    #     recognized_texts = []
    #     confidences = []
    #     for region in result :
    #         text, confidence = region[0]
    #         recognized_texts.append(text)
    #         confidences.append(confidence)

    #     combined_text = " ".join(recognized_texts)
    #     avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    #     return {'text': combined_text, 'score': avg_confidence}