import torch
import numpy as np

from fitz import Rect
from PIL import Image
from itertools import chain
from torchvision import transforms
from collections import defaultdict
from transformers import TableTransformerForObjectDetection

from . import table_utils
from .text_ocr import TextOCR
from .pdf2text_utils import select_device, check_box_area, list2box, box2list, sort_boxes, merge_line_texts

CLASS_NAME2IDX = {
    'table': 0,  # 표 전체
    'table col': 1,  # 열
    'table row': 2,  # 행
    'table col header': 3,  # 열 헤더
    'table projected row header': 4,  # 행 헤더
    'table extended cell': 5,  # 확장 셀
    'no object': 6,
}

CLASS_IDX2NAME = {v: k for k, v in CLASS_NAME2IDX.items()}

STRUCTURE_THRESHOLD = {
    "table": 0.5,
    "table col": 0.5,
    "table row": 0.5,
    "table col header": 0.5,
    "table projected row header": 0.5,
    "table extended cell": 0.5,
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
                 text_ocr_model: TextOCR,  # : PaddleOCR
                 device: str = None):
        self.device = select_device(device)

        # text OCR 모델 준비
        self.text_ocr_model = text_ocr_model

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

        # model의 결과값에 대한 format 변경
        postprocess_objects = process_and_label_objects(outputs, img.size)

        # 추출한 구성요소로 부터 일관된 Table 구조 빌드
        tables_structure = extract_structure_from_objects(postprocess_objects)

        # 구성된 구조를 바탕으로 각 셀을 정의하고, 위치를 최적화하여 일관된 셀을 생성
        tables_cells = [
            extract_cells_from_structure(table)[0] for table in tables_structure
        ]

        for cells in tables_cells:
            self._ocr_cells(img, cells)

        tables_cells = [table_utils.convert_to_md(
            cells) for cells in tables_cells]

        return tables_cells

    def _ocr_cells(self, image, cells):
        text_box_infos = self.text_ocr_model.detect_only(np.array(image))
        box_infos = []

        for box_info in text_box_infos[0]:  # 4개의 좌표로 구성
            _box_info = np.array(box_info)
            xmin = min(_box_info[:, 0])
            xmax = max(_box_info[:, 0])
            ymin = min(_box_info[:, 1])
            ymax = max(_box_info[:, 1])
            new_box_info = np.array([[xmin, ymin], [xmax, ymin],
                                     [xmax, ymax], [xmin, ymax]])

            if not check_box_area(new_box_info, min_h=8, min_w=2):
                continue
            box_infos.append({'position': new_box_info})

        for cell in cells:
            cell_box = cell['bbox']
            inner_text_boxes = []

            for box_info in box_infos:
                _pos = box_info['position']
                # xmin, ymin, xmax, ymax
                text_box = [_pos[0][0], _pos[0][1], _pos[2][0], _pos[2][1]]
                inner_box = list2box(*cut_bbox(cell_box, text_box))
                if check_box_area(inner_box, min_h=8, min_w=2):
                    inner_text_boxes.append({'position':  inner_box})

            if inner_text_boxes:
                for box_info in inner_text_boxes:
                    box = box2list(box_info['position'])
                    ocr_res = self.text_ocr_model.recognize_only(
                        np.array(image.crop(box)))

                    box_info['text'] = ocr_res[0][0][0]
                    box_info['type'] = 'text'

                outs = sort_boxes(inner_text_boxes, key='position')
                cell['text_bboxes'] = outs
                outs = list(chain(*outs))
                cell['cell text'] = merge_line_texts(
                    outs,
                    auto_line_break=True,
                    line_sep=' ',
                )


def cut_bbox(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return x1, y1, x2, y2


def process_and_label_objects(outputs, table_size):
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


def extract_structure_from_objects(objects):
    # 각 테이블 객체가 테이블 전체를 나타내는 박스를 가짐.
    tables = [ele for ele in objects if ele['label'] == 'table']
    table_structures = []

    for table in tables:
        # table 내부에 포함되는 객체 추출
        objects_in_table = [
            obj for obj in objects if intersection_ratio_for_b1(obj['bbox'], table['bbox']) >= 0.5
        ]

        structure = {}
        # table 내부 객체 간 type에 따른 구분
        cols = [obj for obj in objects_in_table if obj['label'] == 'table col']
        rows = [obj for obj in objects_in_table if obj['label'] == 'table row']

        # col header를 의미
        col_headers = [
            obj for obj in objects_in_table if obj['label'] == 'table col header']

        # 확장된 셀
        extended_cells = [
            obj for obj in objects_in_table if obj['label'] == 'table extended cell']

        # 확장된 셀이라고해서 row header를 의미하는 건 아님
        for obj in extended_cells:
            obj['projected row header'] = False

        row_headers = [obj for obj in objects_in_table
                       if obj['label'] == 'table projected row header']

        # row header 체크
        for obj in row_headers:
            obj['projected row header'] = True

        extended_cells.extend(row_headers)

        for obj in rows:
            obj['col header'] = False
            for header_obj in col_headers:
                # row가 col header와 겹치는지 확인
                if intersection_ratio_for_b1(obj['bbox'], header_obj['bbox']) >= 0.5:
                    obj['col header'] = True

        rows = table_utils.refine_lines('rows', rows, 0.5)
        cols = table_utils.refine_lines('cols', cols, 0.25)

        # Table 크기 조정
        row_rect = Rect()
        for obj in rows:
            row_rect.include_rect(obj['bbox'])

        col_rect = Rect()
        for obj in cols:
            col_rect.include_rect(obj['bbox'])

        # 실질적인 row와 col이 차지하는 좌표를 통해서
        # table 좌표 업데이트
        table['row_col_bbox'] = [
            col_rect[0],  # xmin
            row_rect[1],  # ymin
            col_rect[2],  # xmax
            row_rect[3]  # ymax
        ]
        table['bbox'] = table['row_col_bbox']

        # 열과 행을 정렬하는 후처리를 통해서 일관된 테이블 구조 생성
        rows = table_utils.align_lines('rows', rows, table['bbox'])
        cols = table_utils.align_lines('cols', cols, table['bbox'])

        structure['rows'] = rows
        structure['cols'] = cols
        structure['col headers'] = col_headers
        structure['extended cells'] = extended_cells

        if len(rows) > 0 and len(cols) > 1:
            structure = refine_table_structure(structure)

        table_structures.append(structure)

    return table_structures


def align_headers(headers, rows):
    # 테이블의 최상단 열 헤더를 정의하는 함수
    for row in rows:
        row['col header'] = False

    overlap_row_indices = []
    # header와 row가 겹치는지를 확인
    for header in headers:
        for idx, row in enumerate(rows):
            row_height = row['bbox'][3] - row['bbox'][1]  # ymax - ymin
            min_row_overlap = max(row['bbox'][1], header['bbox'][1])
            max_row_overlap = min(row['bbox'][3], header['bbox'][3])
            overlap_height = max_row_overlap - min_row_overlap
            if overlap_height / row_height >= 0.5:
                overlap_row_indices.append(idx)

    # 겹치는게 없으면 빈 리스트 반환
    if len(overlap_row_indices) == 0:
        return []

    # 첫 번째 헤더와 관련된 row가 0보다 크면, 이전 행도 포함하도록 변경
    if overlap_row_indices[0] > 0:
        overlap_row_indices = list(
            range(overlap_row_indices[0] + 1)) + overlap_row_indices

    header_rect = Rect()
    last_row_idx = -1
    for row_idx in overlap_row_indices:
        if row_idx == last_row_idx + 1:  # 연속되는 Row만 처리
            row = rows[row_idx]
            row['col header'] = True
            header_rect = header_rect.include_rect(row['bbox'])
            last_row_idx = row_idx
        else:  # 연속하지 않으면 헤더가 아님
            break

    header = {'bbox': list(header_rect)}  # bbox 좌표

    return [header]


def refine_table_structure(structure):
    # 테이블 구조 후처리 과정
    rows, cols = structure['rows'], structure['cols']

    col_headers = structure['col headers']

    # col header를 일정 Threshold 이상만 남긴다.
    col_headers = [obj for obj in col_headers
                   if obj['prob'] >= STRUCTURE_THRESHOLD['table col header']]

    col_headers = table_utils.nms(col_headers, 0.05)
    col_headers = align_headers(col_headers, rows)

    # 확장된 셀 처리, True면 row header임
    extended_cells = [cell for cell in structure['extended cells']
                      if not cell['projected row header']]

    # row_headers 추출
    projected_row_headers = [cell for cell in structure['extended cells']
                             if cell['projected row header']]

    extended_cells = [obj for obj in extended_cells
                      if obj['prob'] >= STRUCTURE_THRESHOLD['table extended cell']]

    projected_row_headers = [obj for obj in projected_row_headers
                             if obj['prob'] >= STRUCTURE_THRESHOLD['table projected row header']]

    extended_cells.extend(projected_row_headers)

    # NMS 작업 전 정렬을 통해 NMS 작업의 정확도를 향상시킨다.
    extended_cells = table_utils.align_extended_cells(
        extended_cells, rows, cols)

    # NMS 작업 진행
    extended_cells = table_utils.nms_extended_cells(extended_cells)

    table_utils.header_extended_cell_tree(extended_cells)

    structure['cols'] = cols
    structure['rows'] = rows
    structure['extended cells'] = extended_cells
    structure['col headers'] = col_headers

    return structure


def extract_cells_from_structure(table_structure):
    "테이블 구조를 셀로 변환하고, 헤더 / 데이터 셀로 분류하는 기능"
    cols = table_structure['cols']
    rows = table_structure['rows']
    extended_cells = table_structure['extended cells']
    cells = []
    subcells = []

    for col_idx, col in enumerate(cols):
        for row_idx, row in enumerate(rows):
            # row, col의 교차점을 이용하여 cell 구성
            col_rect = Rect(list(col['bbox']))
            row_rect = Rect(list(row['bbox']))
            cell_rect = row_rect.intersect(col_rect)
            header = 'col header' in row and row['col header']
            cell = {
                'bbox': list(cell_rect),
                'col_indices': [col_idx],
                'row_indices': [row_idx],
                'col header': header,
            }

            # cell이 다른 확장 셀과 겹치는 지 확인, 겹치는 비율이 50% 초과면 서브셀로 분류
            cell['subcell'] = False
            for extended_cell in extended_cells:
                extended_cell_rect = Rect(list(extended_cell['bbox']))
                if (extended_cell_rect.intersect(cell_rect).get_area() / cell_rect.get_area()) > 0.5:
                    cell['subcell'] = True
                    break

            if cell['subcell']:
                subcells.append(cell)
            else:  # 서브셀이 아니면 일반 셀로 저장
                cell['projected row header'] = False
                cells.append(cell)

    # 확장 셀 처리
    for extended_cell in extended_cells:
        extended_cell_rect = Rect(list(extended_cell['bbox']))
        cell_cols = set()
        cell_rows = set()
        cell_rect = None
        header = True

        for subcell in subcells:
            subcell_rect = Rect(list(subcell['bbox']))
            subcell_rect_area = subcell_rect.get_area()

            # 겹치는 부분이 50% 이상이면 포함시킨다.
            if (subcell_rect.intersect(extended_cell_rect).get_area() / subcell_rect_area) > 0.5:
                if cell_rect is None:
                    cell_rect = Rect(list(subcell['bbox']))
                else:
                    cell_rect.include_rect(Rect(list(subcell['bbox'])))

                # 포함한 subcell의 row와 col을 저장
                cell_rows = cell_rows.union(set(subcell['row_indices']))
                cell_cols = cell_cols.union(set(subcell['col_indices']))

                header = header and 'col header' in subcell and subcell['col header']

        # subcell의 row와 col이 저장되어 있다면, 새로운 셀을 만들어 추가
        if len(cell_rows) > 0 and len(cell_cols) > 0:
            cell = {
                'bbox': list(cell_rect),
                'col_indices': list(cell_cols),
                'row_indices': list(cell_rows),
                'col header': header,
                'projected row header': extended_cell['projected row header'],
            }
            cells.append(cell)

    # tokens이 없으므로 빈 배열 반환
    # _, _, cell_match_scores = slot_into_containers(cells, tokens)
    cell_match_scores = []
    try:
        mean_match_score = sum(cell_match_scores) / len(cell_match_scores)
        min_match_score = min(cell_match_scores)
        confidence_score = (mean_match_score + min_match_score) / 2
    except:
        confidence_score = 0

    # 최종 추출전 row, col 확장
    dilated_cols = cols
    dilated_rows = rows
    for cell in cells:
        col_rect = Rect()
        for col_idx in cell['col_indices']:
            col_rect.include_rect(list(dilated_cols[col_idx]['bbox']))

        row_rect = Rect()
        for row_idx in cell['col_indices']:
            row_rect.include_rect(list(dilated_rows[row_idx]['bbox']))

        # 최종 bbox 조정
        cell_rect = col_rect.intersect(row_rect)
        cell['bbox'] = list(cell_rect)

    # span_nums_by_cell, _, _ = slot_into_containers(
    #     cells,
    #     tokens,
    #     overlap_threshold=0.001,
    #     unique_assignment=True,
    #     forced_assignment=False,
    # )
    span_nums_by_cell = [[] for _ in range(len(cells))]

    for cell, cell_span_nums in zip(cells, span_nums_by_cell):
        # cell_spans = [tokens[num] for num in cell_span_nums]
        # cell['cell text'] = extract_text_from_spans(
        #     cell_spans, remove_integer_superscripts=False
        # )
        cell['spans'] = ""

    rows = sorted(rows, key=lambda x: x['bbox'][0] + x['bbox'][2])
    cols = sorted(cols, key=lambda x: x['bbox'][1] + x['bbox'][3])

    min_y_values_by_row = defaultdict(list)
    max_y_values_by_row = defaultdict(list)
    min_x_values_by_col = defaultdict(list)
    max_x_values_by_col = defaultdict(list)

    for cell in cells:
        min_row = min(cell["row_indices"])
        max_row = max(cell["row_indices"])
        min_col = min(cell["col_indices"])
        max_col = max(cell["col_indices"])

        for span in cell['spans']:
            min_x_values_by_col[min_col].append(span['bbox'][0])
            min_y_values_by_row[min_row].append(span['bbox'][1])
            max_x_values_by_col[max_col].append(span['bbox'][2])
            max_y_values_by_row[max_row].append(span['bbox'][3])

    for row_idx, row in enumerate(rows):
        if len(min_x_values_by_col[0]) > 0:
            row['bbox'][0] = min(min_x_values_by_col[0])
        if len(min_y_values_by_row[row_idx]) > 0:
            row['bbox'][1] = min(min_y_values_by_row[row_idx])
        if len(max_x_values_by_col[len(cols) - 1]) > 0:
            row['bbox'][2] = max(max_x_values_by_col[len(cols) - 1])
        if len(max_y_values_by_row[row_idx]) > 0:
            row['bbox'][3] = max(max_y_values_by_row[row_idx])

    for col_idx, col in enumerate(cols):
        if len(min_x_values_by_col[col_idx]) > 0:
            col['bbox'][0] = min(min_x_values_by_col[col_idx])
        if len(min_y_values_by_row[0]) > 0:
            col['bbox'][1] = min(min_y_values_by_row[0])
        if len(max_x_values_by_col[col_idx]) > 0:
            col['bbox'][2] = max(max_x_values_by_col[col_idx])
        if len(max_y_values_by_row[len(rows) - 1]) > 0:
            col['bbox'][3] = max(max_y_values_by_row[len(rows) - 1])

    for cell in cells:
        row_rect = Rect()
        column_rect = Rect()
        for row_idx in cell['row_indices']:
            row_rect.include_rect(list(rows[row_idx]['bbox']))
        for col_idx in cell['col_indices']:
            column_rect.include_rect(list(cols[col_idx]['bbox']))
        cell_rect = row_rect.intersect(column_rect)
        if cell_rect.get_area() > 0:
            cell['bbox'] = list(cell_rect)
            pass

    return cells, confidence_score
