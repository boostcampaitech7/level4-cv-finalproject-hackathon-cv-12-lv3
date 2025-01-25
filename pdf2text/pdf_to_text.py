from PIL import Image
import cv2
import numpy as np

from pathlib import Path

from .table_ocr import TableOCR
from .layout_analysis import LayoutAnalyzer, ElementType
from .text_pipeline import Text_Extractor
from .pdf2text_utils import select_device, box2list


class Pdf2Text(object):
    def __init__(self,
                 layout_path,
                 lang):
        layout_path = Path.cwd() / Path(layout_path)
        self.device = select_device(None)
        self.layout_analysis = LayoutAnalyzer(layout_path, self.device)
        self.text_ocr = Text_Extractor(lang = lang)
        self.table_ocr = TableOCR(self.text_ocr, self.device)

    def __call__(self, image, **kwargs):
        return self.recognize(image, **kwargs)

    def recognize(self, image, **kwargs):
        if isinstance(image, Image.Image):
            img = image.convert('RGB')

        layout_output, col_meta = self.layout_analysis.parse(img)

        # NOTE fitz에서 제공하는 page_id or page_number를 추가?
        final_outputs = []
        for layout_ele in layout_output:
            ele_type = layout_ele['type']
            if ele_type == ElementType.IGNORED:
                continue

            bbox = box2list(layout_ele['position'])
            crop_img = img.crop(bbox)

            if ele_type in (ElementType.TEXT, ElementType.TITLE, ElementType.PLAIN_TEXT):
                # TODO TEXT OCR 이전의 전처리 코드 작성
                crop_img = np.array(crop_img)
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
                final_outputs.append(self.text_ocr.Recognize_Text(crop_img))

            elif ele_type == ElementType.FORMULA:
                # TODO FORMULA OCR 이전의 전처리 코드 작성
                # 여기로는 독립된 FORMULA만 입력으로 들어온다.
                crop_img = np.array(crop_img)
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
                final_outputs.append(self.text_ocr.Recognize_Formula(crop_img))
                
            elif ele_type == ElementType.TABLE:
                # TODO TABLE OCR 이전의 전처리 코드 작성
                pass
            elif ele_type == ElementType.FIGURE:
                # NOTE 이미지의 경우에는 어떤 방식으로 처리할지 결정되면 진행
                pass
            else:  # 나머지 타입은 처리하지않는 유형이므로 무시
                pass

        return final_outputs
