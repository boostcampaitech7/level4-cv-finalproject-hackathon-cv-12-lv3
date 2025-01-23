from paddleocr import PaddleOCR
import numpy as np
class TextOCR :
    name = 'paddleocr'

    def __init__(self, lang):
        super().__init__()
        self.ocr_engine = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=True)

    def ocr(self, img: np.ndarray):
        """
        텍스트 박스만 감지.
        """
        result = self.ocr_engine.ocr(img, det=True, rec=True, cls=False)

        return result