import cv2
import torch
import numpy as np

from PIL import Image
from enum import Enum
from utils import select_device
from doclayout_yolo import YOLOv10


class ElementType(Enum):
    ABANDONED = -2  # 텍스트 추출이 필요없는 부분
    IGNORED = -1  # 무시해야할 영역
    UNKNOWN = 0  # 아직 결정되지 않은 영역
    TEXT = 1
    TITLE = 2
    FIGURE = 3
    TABLE = 4
    FORMULA = 5
    PLAIN_TEXT = 11  # 순수 텍스트 영역

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class LayoutAnalyzer:
    ignored_types = {"abandon", "table_footnote"}

    label2type = {
        "title": ElementType.TITLE,
        "figure": ElementType.FIGURE,
        "plain text": ElementType.TEXT,
        "table": ElementType.TABLE,
        "table_caption": ElementType.TEXT,
        "figure_caption": ElementType.TEXT,
        "isolate_formula": ElementType.FORMULA,
        "inline formula": ElementType.FORMULA,
        "formula_caption": ElementType.PLAIN_TEXT,
        "ocr text": ElementType.TEXT,
    }

    # 독립된 구문, 다른 요소들과 결합될 수 없는 것들
    is_isolated = {"table_caption", "figure_caption", "isolate_formula"}

    def __init__(self,
                 model_path,
                 device: str = None):
        # TODO : Layout Analyzer Class
        self.device = select_device(device)
        self.idx2label = {
            0: "title",
            1: "plain text",
            2: "abandon",
            3: "figure",
            4: "figure_caption",
            5: "table",
            6: "table_caption",
            7: "table_footnote",
            8: "isolate_formula",
            9: "formula_caption",
        }
        self.model = YOLOv10(model_path).to(self.device)

    def parse(self, image, reshape_size=1024, confidence=0.2, iou_threshold=0.45):
        # TODO : Layout Analysis 과정
        image = self._prepare_img(image, reshape_size)
        det_res = self.model.predict(
            image,
            imgsz=reshape_size,
            conf=confidence,
            device=self.device)[0]

        scores = det_res.__dict__["boxes"].conf
        boxes = det_res.__dict__["boxes"].xyxy
        _classes = det_res.__dict__["boxes"].cls

    def _prepare_img(self, image, reshape_size, stride=32):
        if isinstance(image, Image.Image):
            img = image.convert("RGB")

        img = np.asarray(image)
        shape = img.shape[:2]
        new_shape = (reshape_size, reshape_size)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
            new_unpad[1]
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img
