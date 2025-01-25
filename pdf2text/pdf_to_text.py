from PIL import Image

from pathlib import Path

from .text_ocr import TextOCR
from .table_ocr import TableOCR
from .formula_ocr import FormulaOCR
from .formula_detect import Formula_Detect
from .layout_analysis import LayoutAnalyzer, ElementType

from .pdf2text_utils import select_device, box2list


class Pdf2Text(object):
    def __init__(self,
                 layout_path,
                 lang):
        layout_path = Path.cwd() / Path(layout_path)
        self.device = select_device(None)
        self.layout_analysis = LayoutAnalyzer(layout_path, self.device)

        self.text_ocr = TextOCR(lang)
        self.formula_ocr = FormulaOCR()
        self.formula_detector = Formula_Detect()
        self.table_ocr = TableOCR(self.text_ocr, self.device)

    def __call__(self, image, **kwargs):
        return self.recognize(image, **kwargs)

    def recognize(self, image, **kwargs):
        if isinstance(image, Image.Image):
            img = image.convert('RGB')

        layout_output, col_meta = self.layout_analysis.parse(img)

        # NOTE fitz에서 제공하는 page_id or page_number를 추가?
