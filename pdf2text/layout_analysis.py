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
        self.device = select_device(device)
        self.processor = DonutProcessor.from_pretrained(
            "naver-clova-ix/donut-base-finetuned-cord-v2")
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "naver-clova-ix/donut-base-finetuned-cord-v2").to(self.device)
        pass

    def parse(self, image):
        # TODO : Layout Analysis process
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = self.processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        outputs = self.model.generate(
            pixel_values.to(self.device),
            decoder_input_ids=decoder_input_ids.to(self.device),
            max_length=self.model.decoder.config.max_position_embeddings,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(
            self.processor.tokenizer.pad_token, "")
        # remove first task start token
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
        return self.processor.token2json(sequence)
