import cv2
import numpy as np

from PIL import Image
from pathlib import Path
from fitz import Rect
import torch

from .table_ocr import TableOCR
from .layout_analysis import LayoutAnalyzer, ElementType
from .text_pipeline import Text_Extractor
from .pdf2text_utils import select_device, box2list, add_edge_margin, expand_bbox_with_original, matching_captioning


class Pdf2Text(object):
    def __init__(self,
                 layout_path):
        layout_path = Path.cwd() / Path(layout_path)
        self.device = select_device(None)

        self.layout_analysis = LayoutAnalyzer(layout_path, self.device)
        self.text_ocr = Text_Extractor()

        # 이거는 변수명 좀 수정해야할 듯 합니다.
        self.table_ocr = TableOCR(self.text_ocr.korean_ocr,
                                  self.text_ocr.english_ocr,
                                  self.device)

    def __call__(self, image, **kwargs):
        return self.recognize(image, **kwargs)

    def recognize(self, image, lang, **kwargs):
        if isinstance(image, Image.Image):
            img = image.convert('RGB')

        layout_output, _ = self.layout_analysis.parse(img)
        final_outputs, caption_outputs = [], []
        table_figure_outputs = []
        for idx, layout_ele in enumerate(layout_output):
            ele_type = layout_ele['type']

            if ele_type == ElementType.IGNORED:
                continue

            bbox = box2list(layout_ele['position'])
            crop_img = img.crop(bbox)

            if ele_type in (ElementType.TEXT, ElementType.TITLE, ElementType.PLAIN_TEXT):
                # NOTE 배경 색상을 기준으로 margin 추가 기능
                crop_img = add_edge_margin(crop_img, 20, 20)

                crop_img = np.array(crop_img)
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)

                text_ocr_output = self.text_ocr.Recognize_Text(crop_img, lang)

                # caption일 경우 따로 추출
                if layout_ele['caption'] is not None:
                    caption_outputs.append({
                        'text': text_ocr_output,
                        'bbox': bbox,
                        'type': layout_ele['caption'],
                        'image': crop_img
                    })
                else:
                    final_outputs.append(text_ocr_output)

            elif ele_type == ElementType.FORMULA:
                # 여기로는 독립된 FORMULA만 입력으로 들어온다.
                # NOTE 배경 색상을 기준으로 margin 추가 기능
                crop_img = add_edge_margin(crop_img, 20, 20)

                crop_img = np.array(crop_img)
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
                final_outputs.append(self.text_ocr.Recognize_Formula(crop_img))

            elif ele_type == ElementType.TABLE:
                # NOTE TABLE은 정보가 잘리는 경우가 존재하기 때문에 기존 이미지에서 bbox 재조정
                new_bbox = expand_bbox_with_original(img, bbox, 10, 10)
                crop_img = img.crop(new_bbox)
                table_figure_outputs.append({
                    'obj': self.table_ocr.ocr(crop_img, lang),
                    'bbox': new_bbox,
                    'type': "Table",
                    'image': crop_img
                })
                # final_outputs.append(self.table_ocr.ocr(crop_img))

            elif ele_type == ElementType.FIGURE:
                new_bbox = expand_bbox_with_original(img, bbox, 10, 10)
                table_figure_outputs.append({
                    'obj': new_bbox,
                    'bbox': bbox,
                    'type': "Figure",
                    'image': new_bbox
                })
            else:  # 나머지 타입은 처리하지않는 유형이므로 무시
                pass

        # unmatch_res -> dict
        # 'caption' : 매칭되지 않은 caption , 'obj' : 매칭되지 않은 Figure or Table
        # 하위 속성 'item' : text or obj, 'bbox' : bbox 값, 'type' : 해당 객체의 type

        # match_res
        # 'figure' : 매칭된 Figure와 caption, 'table' : 매칭된 Table과 caption
        # 하위 속성
        # "caption_number": caption_number,
        # 'obj': obj,
        # 'obj_bbox': obj_bbox,
        # 'caption_bbox': caption_bbox,
        # 'caption_text': caption_text
        match_res, unmatch_res = matching_captioning(
            caption_outputs, table_figure_outputs)

        return " ".join(final_outputs), match_res, unmatch_res

    def recognize_only_text(self, image, lang, **kwargs):
        if isinstance(image, Image.Image):
            img = image.convert('RGB')

        layout_output, _ = self.layout_analysis.parse(img)
        final_outputs = []
        for idx, layout_ele in enumerate(layout_output):
            ele_type = layout_ele['type']

            if ele_type == ElementType.IGNORED:
                continue

            bbox = box2list(layout_ele['position'])
            crop_img = img.crop(bbox)

            if ele_type in (ElementType.TEXT, ElementType.TITLE, ElementType.PLAIN_TEXT):
                if layout_ele['caption'] is not None:
                    continue

                crop_img = add_edge_margin(crop_img, 20, 20)

                crop_img = np.array(crop_img)
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)

                text_ocr_output = self.text_ocr.Recognize_Text(crop_img, lang)
                final_outputs.append(text_ocr_output)

            elif ele_type == ElementType.FORMULA:
                # 여기로는 독립된 FORMULA만 입력으로 들어온다.
                crop_img = add_edge_margin(crop_img, 20, 20)

                crop_img = np.array(crop_img)
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
                final_outputs.append(self.text_ocr.Recognize_Formula(crop_img))
            else:
                pass
        return " ".join(final_outputs)

    def recognize_only_table_figure(self, image, lang, **kwargs):
        if isinstance(image, Image.Image):
            img = image.convert('RGB')

        layout_output, _ = self.layout_analysis.parse(img)
        final_outputs, caption_outputs = [], []
        table_figure_outputs = []
        for idx, layout_ele in enumerate(layout_output):
            ele_type = layout_ele['type']

            if ele_type == ElementType.IGNORED:
                continue

            bbox = box2list(layout_ele['position'])
            crop_img = img.crop(bbox)

            if layout_ele['caption'] is not None:
                expand_img = add_edge_margin(crop_img, 20, 20)

                crop_img = np.array(expand_img)
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)

                text_ocr_output = self.text_ocr.Recognize_Text(crop_img, lang)
                caption_outputs.append({
                    'text': text_ocr_output,
                    'bbox': bbox,
                    'type': layout_ele['caption'],
                    'image': expand_img
                })

            elif ele_type == ElementType.TABLE:
                new_bbox = expand_bbox_with_original(img, bbox, 10, 10)
                crop_img = img.crop(new_bbox)
                table_figure_outputs.append({
                    'obj': self.table_ocr.ocr(crop_img, lang),
                    'bbox': new_bbox,
                    'type': "Table",
                    'image': crop_img
                })

            elif ele_type == ElementType.FIGURE:
                new_bbox = expand_bbox_with_original(img, bbox, 10, 10)
                crop_img = img.crop(new_bbox)
                table_figure_outputs.append({
                    'obj': crop_img,
                    'bbox': new_bbox,
                    'type': "Figure",
                    'image': crop_img
                })
            else:  # 나머지 타입은 처리하지않는 유형이므로 무시
                pass

        return matching_captioning(caption_outputs, table_figure_outputs)
