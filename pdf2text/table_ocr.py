import numpy as np
import torch


from PIL import Image
from utils import select_device
# from transformers import AutoImageProcessor, AutoModelForObjectDetection
from transformers import TableTransformerForObjectDetection
from torchvision import transforms
from fitz import Rect


CLASS_NAME2IDX = {
    'table': 0,
    'table column': 1,
    'table row': 2,
    'table column header': 3,
    'table projected row header': 4,
    'table spanning cell': 5,
    'no object': 6,
}

CLASS_IDX2NAME = {v: k for k, v in CLASS_NAME2IDX.items()}

STRUCTURE_THRESHOLD = {
    "table": 0.5,
    "table column": 0.5,
    "table row": 0.5,
    "table column header": 0.5,
    "table projected row header": 0.5,
    "table spanning cell": 0.5,
    "no object": 10,
}


class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize(
            (int(round(scale * width)), int(round(scale * height))))

        return resized_image


STRUCTURE_TRANSFORM = transforms.Compose(
    [
        MaxResize(1000),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class TableOCR:
    def __init__(self,
                 text_ocr_model,  # : PaddleOCR
                 device: str = None):
        self.device = select_device(device)

        # text OCR 모델 준비
        self.text_ocr_model = text_ocr_model

        # 테이블 구조 파악을 위한 모델 준비
        # self.processor = AutoImageProcessor.from_pretrained(
        #     "microsoft/table-transformer-structure-recognition")
        # self.structure_model = AutoModelForObjectDetection.from_pretrained(
        #     "microsoft/table-transformer-structure-recognition").to(self.device)
        self.structure_model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-structure-recognition").to(self.device)
        self.structure_model.eval()

    def ocr(self, image):
        # TODO table ocr
        if isinstance(image, Image.Image):
            img = image.convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert("RGB")
        else:
            raise TypeError(
                f"Invalid Input Type : Expected an instance of Image.Image or np.ndarray, but received{type(image)}")

        # Image.Image -> Tensor
        image_tensor = STRUCTURE_TRANSFORM(img)
        assert isinstance(
            image_tensor, torch.Tensor), f"Expected an instance of torch.Tensor, but received{type(image_tensor)}"

        # Table 구조 파악
        with torch.no_grad():
            outputs = self.structure_model(
                image_tensor.unsqueeze(0).to(self.device))

        parsing_outputs = parse_structure_model_outputs(outputs, img.size)

        table_structure = build_table_from_objects(parsing_outputs)


def parse_structure_model_outputs(outputs, table_size):
    # outputs의 label 추출, (values, indices)
    outputs_info = outputs['logits'].softmax(-1).max(-1)

    # labels, prob 추출
    pred_labels = outputs_info.indices.detach().cpu().squeeze().tolist()
    pred_prob = outputs_info.values.detach().cpu().squeeze().tolist()

    # 예측 박스 추출 [center_x, center_y, w, h]
    pred_bboxes = outputs['pred_boxes'].detach().cpu().squeeze()
    pred_bboxes = [bbox.tolist()
                   for bbox in cxcywh_to_xyxy(pred_bboxes, table_size)]

    def template(label, prob, bbox): return {
        'label': label, 'prob': prob, 'bbox': bbox}

    results = [template(CLASS_IDX2NAME[l], p, bb) for l, p, bb in zip(
        pred_labels, pred_prob, pred_bboxes) if CLASS_IDX2NAME[l] != "no object"]

    return results


def cxcywh_to_xyxy(bboxes, table_size):
    table_w, table_h = table_size

    center_x, center_y, w, h = bboxes.unbind(-1)
    # xmin, ymin, xmax, ymax
    change_bboxes = [center_x - w * 0.5, center_y - h * 0.5,
                     center_x + w * 0.5, center_y + h * 0.5]
    change_bboxes = torch.stack(change_bboxes, dim=1)
    return change_bboxes * torch.Tensor([table_w, table_h, table_w, table_h])


def intersection_ratio_for_b1(b1, b2):
    # b1에 대한 겹치는 영역의 비율 파악
    b1_area = Rect(b1).get_area()
    return Rect(b1).intersect(b2).get_area() / b1_area if b1_area > 0 else 0


def build_table_from_objects(objects):
    # 각 테이블 객체가 테이블 전체를 나타내는 박스를 가짐.
    tables = [ele for ele in objects if ele['label'] == 'table']
    table_structure = []

    for table in tables:
        # table 내부에 포함되는 객체 추출
        objects_in_table = [
            obj for obj in objects if intersection_ratio_for_b1(obj['bbox'], table['bbox']) >= 0.5
        ]

        structure = {}
        # table 내부 객체 간 type에 따른 구분
        columns = [obj for obj in objects_in_table if obj['label']
                   == 'table column']
        rows = [obj for obj in objects_in_table if obj['label'] == 'table row']

        # col header를 의미
        col_headers = [
            obj for obj in objects_in_table if obj['label'] == 'table column header']

        # 병합된 셀
        extended_cells = [
            obj for obj in objects_in_table if obj['label'] == 'table spanning cell']

        # for obj in extended_cells:
        #     obj['projected row header'] = False

        row_headers = [obj for obj in objects_in_table if obj['label']
                       == 'table projected row header']

        # for obj in row_headers:
        #     obj['projected row header'] = True

        extended_cells.extend(row_headers)

        # for obj in rows:
        #     obj['column header'] = False
        #     for header_obj in col_headers:
        #         if iob(obj['bbox'], header_obj['bbox']) >= 0.5:
        #             obj['column header'] = True

        rows = refine_lines('rows', rows, 0.5)
        columns = refine_lines('cols', columns, 0.25)

        print(rows)

        # # Table 크기 조정
        # row_rect = Rect()
        # for obj in rows:
        #     row_rect.include_rect(obj['bbox'])


def refine_lines(mode, lines, threshold):
    # NOTE: 해당 함수는 부가적인 함수로 따로 빼는게 좋으려나?
    if len(lines) == 0:
        return []

    assert mode in ('rows', 'cols')

    lines = sorted(lines, key=lambda x: -x['prob'])

    remove_object = [False] * len(lines)

    # 일정 부분 이상 겹치는 부분 제거
    for obj1 in range(1, len(lines)):
        obj1_rect = Rect(lines[obj1]['bbox'])
        obj1_area = obj1_rect.get_area()

        if obj1_area <= 0 or remove_object[obj1]:
            continue

        for obj2 in range(obj1):
            obj2_rect = Rect(lines[obj2]['bbox'])
            inter = obj1_rect.intersect(obj2_rect).get_area()

            score = inter / obj1_area

            if score >= threshold:
                remove_object[obj1] = True
                break
    lines = [obj for idx, obj in enumerate(lines) if not remove_object[idx]]

    if len(lines) == 1:
        return lines

    # 좌표에 따른 정렬
    axis_range = (1, 3) if mode == 'rows' else (0, 2)

    return sorted(lines, key=lambda x: x['bbox'][axis_range[0]] + x['bbox'][axis_range[1]])
