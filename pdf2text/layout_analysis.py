import re
import torch

from enum import Enum
from utils import select_device
from transformers import AutoModelForTokenClassification, AutoProcessor
from transformers import AutoModel
from transformers import DonutProcessor, VisionEncoderDecoderModel
import cv2
import numpy as np
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
    def __init__(self,
                 device: str = None):
        # TODO : Layout Analyzer Class
        pass

    def parse(self):
        # TODO : Layout Analysis process
        pass
