import torch
import numpy as np
import re

from fitz import Rect
from PIL import Image
from itertools import chain
from utils import select_device
from functools import cmp_to_key
from torchvision import transforms
from collections import defaultdict
from transformers import TableTransformerForObjectDetection

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
                 text_ocr_model,  # : PaddleOCR
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


def sort_boxes(boxes, key):
    boxes.sort(key=lambda x: x[key][0, 1])

    for box in boxes:
        box['line_number'] = -1

    def get_anchor():
        anchor = None
        for box in boxes:
            if box['line_number'] == -1:
                anchor = box
                break
        return anchor

    lines = []
    while True:
        anchor = get_anchor()
        if anchor is None:
            break
        anchor['line_number'] = len(lines)
        line_boxes = get_same_line_boxes(anchor, boxes)
        line_boxes = sort_and_filter_line_boxes(line_boxes, key)
        lines.append(line_boxes)

    return lines


def get_same_line_boxes(anchor, total_boxes):
    line_boxes = [anchor]
    for box in total_boxes:
        if box['line_number'] >= 0:
            continue
        if max([y_overlap(box, l_box) for l_box in line_boxes]) > 0.1:
            line_boxes.append(box)
    return line_boxes


def y_overlap(box1, box2, key='position'):
    # Interaction / min(height1, height2)
    if key:
        box1 = [box1[key][0][0], box1[key][0][1],
                box1[key][2][0], box1[key][2][1]]
        box2 = [box2[key][0][0], box2[key][0][1],
                box2[key][2][0], box2[key][2][1]]
    else:
        box1 = [box1[0][0], box1[0][1], box1[2][0], box1[2][1]]
        box2 = [box2[0][0], box2[0][1], box2[2][0], box2[2][1]]

    if box1[3] <= box2[1] or box2[3] <= box1[1]:
        return 0

    y_min = max(box1[1], box2[1])
    y_max = min(box1[3], box2[3])
    return (y_max - y_min) / max(1, min(box1[3] - box1[1], box2[3] - box2[1]))


def sort_and_filter_line_boxes(line_boxes, key):
    if len(line_boxes) <= 1:
        return line_boxes

    allowed_max_overlay_x = 20

    def find_right_box(anchor):
        anchor_width = anchor[key][2, 0] - anchor[key][0, 0]
        allowed_max = min(
            max(allowed_max_overlay_x, anchor_width * 0.5), anchor_width * 0.95
        )
        right_boxes = [
            l_box
            for l_box in line_boxes[1:]
            if l_box['line_number'] < 0
            and l_box[key][0, 0] >= anchor[key][2, 0] - allowed_max
        ]
        if not right_boxes:
            return None
        right_boxes = sorted(
            right_boxes,
            key=cmp_to_key(
                lambda x, y: _compare_box(x, y, anchor, key, left_best=True)
            ),
        )
        return right_boxes[0]

    def find_left_box(anchor):
        anchor_width = anchor[key][2, 0] - anchor[key][0, 0]
        allowed_max = min(
            max(allowed_max_overlay_x, anchor_width * 0.5), anchor_width * 0.95
        )
        left_boxes = [
            l_box
            for l_box in line_boxes[1:]
            if l_box['line_number'] < 0
            and l_box[key][2, 0] <= anchor[key][0, 0] + allowed_max
        ]
        if not left_boxes:
            return None
        left_boxes = sorted(
            left_boxes,
            key=cmp_to_key(
                lambda x, y: _compare_box(x, y, anchor, key, left_best=False)
            ),
        )
        return left_boxes[-1]

    res_boxes = [line_boxes[0]]
    anchor = res_boxes[0]
    line_number = anchor['line_number']

    while True:
        right_box = find_right_box(anchor)
        if right_box is None:
            break
        right_box['line_number'] = line_number
        res_boxes.append(right_box)
        anchor = right_box

    anchor = res_boxes[0]
    while True:
        left_box = find_left_box(anchor)
        if left_box is None:
            break
        left_box['line_number'] = line_number
        res_boxes.insert(0, left_box)
        anchor = left_box

    return res_boxes


def _compare_box(box1, box2, anchor, key, left_best: bool = True):
    over1 = y_overlap(box1, anchor, key)
    over2 = y_overlap(box2, anchor, key)
    if box1[key][2, 0] < box2[key][0, 0] - 3:
        return -1
    elif box2[key][2, 0] < box1[key][0, 0] - 3:
        return 1
    else:
        if max(over1, over2) >= 3 * min(over1, over2):
            return over2 - over1 if left_best else over1 - over2
        return box1[key][0, 0] - box2[key][0, 0]


def box2list(bbox):
    return [int(bbox[0, 0]), int(bbox[0, 1]), int(bbox[2, 0]), int(bbox[2, 1])]


def list2box(xmin, ymin, xmax, ymax):
    return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])


def cut_bbox(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return x1, y1, x2, y2


def check_box_area(box, min_h, min_w):
    return (
        box[0, 0] + min_w <= box[1, 0]
        and box[1, 1] + min_h <= box[2, 1]
        and box[2, 0] >= box[3, 0] + min_w
        and box[3, 1] >= box[0, 1] + min_h
    )


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

        rows = refine_lines('rows', rows, 0.5)
        cols = refine_lines('cols', cols, 0.25)

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
        rows = align_lines('rows', rows, table['bbox'])
        cols = align_lines('cols', cols, table['bbox'])

        structure['rows'] = rows
        structure['cols'] = cols
        structure['col headers'] = col_headers
        structure['extended cells'] = extended_cells

        if len(rows) > 0 and len(cols) > 1:
            structure = refine_table_structure(structure)

        table_structures.append(structure)

    return table_structures


def refine_lines(mode, lines, threshold):
    # NOTE: 해당 함수는 부가적인 함수로 따로 빼는게 좋으려나?
    if len(lines) == 0:
        return []

    assert mode in ('rows', 'cols')

    lines = nms(lines, threshold)

    if len(lines) == 1:
        return lines

    # 좌표에 따른 정렬
    axis_range = (1, 3) if mode == 'rows' else (0, 2)

    return sorted(lines, key=lambda x: x['bbox'][axis_range[0]] + x['bbox'][axis_range[1]])


def nms(objects, threshold):
    # NOTE: 해당 함수는 부가적인 함수로 따로 빼는게 좋으려나?
    if len(objects) == 0:
        return []

    objects = sorted(objects, key=lambda x: -x['prob'])
    remove_object = [False] * len(objects)

    # 일정 부분 이상 겹치는 부분 제거
    for obj1_idx in range(1, len(objects)):
        obj1_rect = Rect(objects[obj1_idx]['bbox'])
        obj1_area = obj1_rect.get_area()

        if obj1_area <= 0 or remove_object[obj1_idx]:
            continue

        for obj2 in range(obj1_idx):
            obj2_rect = Rect(objects[obj2]['bbox'])
            inter = obj1_rect.intersect(obj2_rect).get_area()

            score = inter / obj1_area

            if score >= threshold:
                remove_object[obj1_idx] = True
                break
    return [obj for idx, obj in enumerate(objects) if not remove_object[idx]]


def align_lines(mode, lines, standard):
    assert mode in ('rows', 'cols')

    axis_range = (0, 2) if mode == 'rows' else (1, 3)
    for line in lines:
        line['bbox'][axis_range[0]] = standard[axis_range[0]]
        line['bbox'][axis_range[1]] = standard[axis_range[1]]
    return lines


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


def align_extended_cells(cells, rows, cols):
    # 확장된 셀의 경계를 row, col에 맞게 정렬
    # 확장된 셀이 row와 col과의 경계가 절반이상 겹치지 않으면 제거한다.
    aligned_cells = []

    for cell in cells:
        cell['header'] = False  # 헤더와 관련있는지 판단 여부
        row_rect = None
        col_rect = None
        overlap_row_with_header = set()
        overlap_row_with_data = set()

        for idx, row in enumerate(rows):
            # row와 cell간의 겹치는지 판단 -> 높이로 판단
            row_height = row['bbox'][3] - row['bbox'][1]
            cell_height = cell['bbox'][3] - cell['bbox'][1]

            min_row_overlap = max(row['bbox'][1], cell['bbox'][1])
            max_row_overlap = min(row['bbox'][3], cell['bbox'][3])
            overlap_height = max_row_overlap - min_row_overlap

            if 'span' in cell:
                overlap_fraction = max(overlap_height/row_height,
                                       overlap_height/cell_height)
            else:
                overlap_fraction = overlap_height / row_height

            if overlap_fraction >= 0.5:
                if 'header' in row and row['header']:
                    overlap_row_with_header.add(idx)
                else:
                    overlap_row_with_data.add(idx)

        cell['header'] = False
        # data와 헤더 둘다 겹치면 더 적은 그룹 제거
        if len(overlap_row_with_data) > 0 and len(overlap_row_with_header) > 0:
            if len(overlap_row_with_data) > len(overlap_row_with_header):
                overlap_row_with_header = set()
            else:
                overlap_row_with_data = set()

        # header만 겹치면 cell은 header로 표시
        if len(overlap_row_with_header) > 0:
            cell['header'] = True
        elif 'span' in cell:
            continue

        # 겹치는 행 영역 계산
        overlap_rows = overlap_row_with_data.union(
            overlap_row_with_header)

        for idx in overlap_rows:
            if row_rect is None:
                row_rect = Rect(rows[idx]['bbox'])
            else:
                row_rect = row_rect.include_rect(rows[idx]['bbox'])

        if row_rect is None:
            continue

        overlap_cols = []
        for col_num, col in enumerate(cols):  # 너비로 판단
            col_width = col['bbox'][2] - col['bbox'][0]
            cell_width = cell['bbox'][2] - cell['bbox'][0]

            min_col_overlap = max(col['bbox'][0], cell['bbox'][0])
            max_col_overlap = min(col['bbox'][2], cell['bbox'][2])
            overlap_width = max_col_overlap - min_col_overlap

            if 'span' in cell:
                overlap_fraction = max(overlap_width/col_width,
                                       overlap_width/cell_width)
                if cell['header']:
                    overlap_fraction = overlap_fraction * 2
            else:
                overlap_fraction = overlap_width / col_width

            if overlap_fraction >= 0.5:
                overlap_cols.append(col_num)

                if col_rect is None:
                    col_rect = Rect(col['bbox'])
                else:
                    col_rect = col_rect.include_rect(col['bbox'])

        if col_rect is None:
            continue

        cell_bbox = list(row_rect.intersect(col_rect))
        cell['bbox'] = cell_bbox

        # 하나 이상의 row, col을 포함하면 해당 cell은 정렬된 것으로 판단
        if (len(overlap_rows) > 0 and len(overlap_cols) > 0
                and (len(overlap_rows) > 1 or len(overlap_cols) > 1)):
            cell['row_numbers'] = list(overlap_rows)
            cell['col_numbers'] = overlap_cols
            aligned_cells.append(cell)

            # 현재 처리하려는 pipeline에서는 'span'이라는 속성이 없는 것으로 판단
            if 'span' in cell and cell['header'] and len(cell['col_numbers']) > 1:
                for row_idx in range(0, min(cell['row_numbers'])):
                    new_cell = {'row_numbers': [row_idx], 'col_numbers': cell['col_numbers'],
                                'prob': cell['prob'], 'propagated': True}
                    new_cell_cols = [cols[idx]
                                     for idx in cell['col_numbers']]
                    new_cell_rows = [rows[idx] for idx in cell['row_numbers']]

                    # cell의 bbox 업데이트
                    bbox = [min([col['bbox'][0] for col in new_cell_cols]),
                            min([row['bbox'][1] for row in new_cell_rows]),
                            max([col['bbox'][2]
                                for col in new_cell_cols]),
                            max([row['bbox'][3] for row in new_cell_rows])]

                    new_cell['bbox'] = bbox
                    aligned_cells.append(new_cell)

    return aligned_cells


def nms_extended_cells(cells):
    # 확장된 셀들이 같은 셀에 겹치면 낮은 확률을 가진 확장 셀의 크기를 줄인다.
    cells = sorted(cells, key=lambda x: -x['prob'])
    remove_cells = [False] * len(cells)

    for cell_num1 in range(1, len(cells)):
        cell_obj1 = cells[cell_num1]
        for cell_num2 in range(cell_num1):
            cell_obj2 = cells[cell_num2]
            remove_extended_cell_overlap(cell_obj1, cell_obj2)

        if ((len(cell_obj1['row_numbers']) < 2 and len(cell_obj1['col_numbers']) < 2)
                or len(cell_obj1['row_numbers']) == 0 or len(cell_obj1['col_numbers']) == 0):
            remove_cells[cell_num1] = True

    return [obj for idx, obj in enumerate(cells) if not remove_cells[idx]]


def remove_extended_cell_overlap(cell_obj1, cell_obj2):
    # 확장된 셀간의 중복을 해결한다.
    # 중복된 셀간의 prob를 비교하여 더 낮은 셀에서 row or col을 제거함으로써 해결
    common_rows = set(cell_obj2['row_numbers']).intersection(
        set(cell_obj1['row_numbers']))
    common_cols = set(cell_obj2['col_numbers']).intersection(
        set(cell_obj1['col_numbers']))

    while len(common_rows) > 0 and len(common_cols) > 0:
        if len(cell_obj1['row_numbers']) < len(cell_obj1['col_numbers']):
            min_col = min(cell_obj1['col_numbers'])
            max_col = max(cell_obj1['col_numbers'])

            if max_col in common_cols:
                common_cols.remove(max_col)
                cell_obj1['col_numbers'].remove(max_col)
            elif min_col in common_cols:
                common_cols.remove(min_col)
                cell_obj1['col_numbers'].remove(min_col)
            else:
                cell_obj1['col_numbers'] = []
                common_cols = set()
        else:
            min_row = min(cell_obj1['row_numbers'])
            max_row = max(cell_obj1['row_numbers'])

            if max_row in common_rows:
                common_rows.remove(max_row)
                cell_obj1['row_numbers'].remove(max_row)
            elif min_row in common_rows:
                common_rows.remove(min_row)
                cell_obj1['row_numbers'].remove(min_row)
            else:
                cell_obj1['row_numbers'] = []
                common_rows = set()


def header_extended_cell_tree(cells):
    # Header 속성을 가진 확장셀 추출
    header_cells = [
        cell for cell in cells if 'header' in cell and cell['header']]
    header_cells = sorted(header_cells, key=lambda x: -x['prob'])

    for header_cell in header_cells[:]:
        ancestors_by_row = defaultdict(int)
        min_row = min(header_cell['row_numbers'])

        for header_cell2 in header_cells:
            max_row2 = max(header_cell2['row_numbers'])

            if max_row2 < min_row:
                if (set(header_cell['col_numbers']).issubset(
                        set(header_cell2['col_numbers']))):
                    for row2 in header_cell2['row_numbers']:
                        ancestors_by_row[row2] += 1

        for row in range(min_row):
            if not ancestors_by_row[row] == 1:
                cells.remove(header_cell)
                break


def refine_table_structure(structure):
    # 테이블 구조 후처리 과정
    rows, cols = structure['rows'], structure['cols']

    col_headers = structure['col headers']
    # col header를 일정 Threshold 이상만 남긴다.
    col_headers = [obj for obj in col_headers
                   if obj['prob'] >= STRUCTURE_THRESHOLD['table col header']]

    col_headers = nms(col_headers, 0.05)
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
    extended_cells = align_extended_cells(extended_cells, rows, cols)

    # NMS 작업 진행
    extended_cells = nms_extended_cells(extended_cells)

    header_extended_cell_tree(extended_cells)

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


def merge_line_texts(
    outs,
    auto_line_break: bool = True,
    line_sep='\n',
    embed_sep=(' $', '$ '),
    isolated_sep=('$$\n', '\n$$'),
    spellchecker=None,
) -> str:
    if not outs:
        return ''

    out_texts = []
    line_margin_list = []
    isolated_included = []
    line_height_dict = defaultdict(list)
    line_ymin_ymax_list = []

    for _out in outs:
        line_number = _out.get('line_number', 0)
        while len(out_texts) <= line_number:
            out_texts.append([])
            line_margin_list.append([100000, 0])
            isolated_included.append(False)
            line_ymin_ymax_list.append([100000, 0])

        cur_text = _out['text']
        cur_type = _out.get('type', 'text')
        box = _out['position']

        if cur_type in ('embedding', 'isolated'):
            sep = isolated_sep if _out['type'] == 'isolated' else embed_sep
            cur_text = sep[0] + cur_text + sep[1]

        if cur_type == 'isolated':
            isolated_included[line_number] = True
            cur_text = line_sep + cur_text + line_sep

        out_texts[line_number].append(cur_text)

        line_margin_list[line_number][1] = max(
            line_margin_list[line_number][1], float(box[2, 0])
        )
        line_margin_list[line_number][0] = min(
            line_margin_list[line_number][0], float(box[0, 0])
        )

        if cur_type == 'text':
            line_height_dict[line_number].append(box[2, 1] - box[1, 1])
            line_ymin_ymax_list[line_number][0] = min(
                line_ymin_ymax_list[line_number][0], float(box[0, 1])
            )
            line_ymin_ymax_list[line_number][1] = max(
                line_ymin_ymax_list[line_number][1], float(box[2, 1])
            )

    line_text_list = [smart_join(o) for o in out_texts]

    for _line_number in line_height_dict.keys():
        if line_height_dict[_line_number]:
            line_height_dict[_line_number] = np.mean(
                line_height_dict[_line_number])

    _line_heights = list(line_height_dict.values())
    mean_height = np.mean(_line_heights) if _line_heights else None

    default_res = re.sub(rf'{line_sep}+', line_sep,
                         line_sep.join(line_text_list))

    if not auto_line_break:
        return default_res

    line_lengths = [rx - lx for lx, rx in line_margin_list]
    line_length_thrsh = max(line_lengths) * 0.3

    if line_length_thrsh < 1:
        return default_res

    lines = np.array(
        [
            margin
            for idx, margin in enumerate(line_margin_list)
            if isolated_included[idx] or line_lengths[idx] >= line_length_thrsh
        ]
    )

    if lines.shape[0] < 1:
        return default_res
    min_x, max_x = min(lines[:, 0]), max(lines[:, 1])

    indentation_thrsh = (max_x - min_x) * 0.1
    if mean_height is not None:
        indentation_thrsh = 1.5 * mean_height

    min_x, max_x = cal_block_xmin_xmax(lines, indentation_thrsh)

    res_line_texts = [''] * len(line_text_list)
    line_text_list = [(idx, txt)
                      for idx, txt in enumerate(line_text_list) if txt]

    for idx, (line_number, txt) in enumerate(line_text_list):
        if isolated_included[line_number]:
            res_line_texts[line_number] = line_sep + txt + line_sep
            continue

        tmp = txt
        if line_margin_list[line_number][0] > min_x + indentation_thrsh:
            tmp = line_sep + txt
        if line_margin_list[line_number][1] < max_x - indentation_thrsh:
            tmp = tmp + line_sep
        if idx < len(line_text_list) - 1:
            cur_height = line_ymin_ymax_list[line_number][1] - \
                line_ymin_ymax_list[line_number][0]
            next_line_number = line_text_list[idx + 1][0]
            if (
                cur_height > 0
                and line_ymin_ymax_list[next_line_number][0] < line_ymin_ymax_list[next_line_number][1]
                and line_ymin_ymax_list[next_line_number][0] - line_ymin_ymax_list[line_number][1]
                > cur_height
            ):
                tmp = tmp + line_sep
        res_line_texts[idx] = tmp

    outs = smart_join([c for c in res_line_texts if c], spellchecker)
    return re.sub(rf'{line_sep}+', line_sep, outs)


def smart_join(str_list, spellchecker=None):
    def is_chinese(ch):
        return '\u4e00' <= ch <= '\u9fff'

    def contain_whitespace(s):
        if re.search(r'\s', s):
            return True
        else:
            return False

    def find_first_punctuation_position(text):
        pattern = re.compile(r'[,.!?;:()\[\]{}\'\"\\/-]')
        match = pattern.search(text)
        if match:
            return match.start()
        else:
            return len(text)

    str_list = [s for s in str_list if s]
    if not str_list:
        return ''
    res = str_list[0]
    for i in range(1, len(str_list)):
        if (is_chinese(res[-1]) and is_chinese(str_list[i][0])) or contain_whitespace(
            res[-1] + str_list[i][0]
        ):
            res += str_list[i]
        elif spellchecker is not None and res.endswith('-'):
            fields = res.rsplit(' ', maxsplit=1)
            if len(fields) > 1:
                new_res, prev_word = fields[0], fields[1]
            else:
                new_res, prev_word = '', res

            fields = str_list[i].split(' ', maxsplit=1)
            if len(fields) > 1:
                next_word, new_next = fields[0], fields[1]
            else:
                next_word, new_next = str_list[i], ''

            punct_idx = find_first_punctuation_position(next_word)
            next_word = next_word[:punct_idx]
            new_next = str_list[i][len(next_word):]
            new_word = prev_word[:-1] + next_word
            if (
                next_word
                and spellchecker.unknown([prev_word + next_word])
                and spellchecker.known([new_word])
            ):
                res = new_res + ' ' + new_word + new_next
            else:
                new_word = prev_word + next_word
                res = new_res + ' ' + new_word + new_next
        else:
            res += ' ' + str_list[i]
    return res


def cal_block_xmin_xmax(lines, indentation_thrsh):
    total_min_x, total_max_x = min(lines[:, 0]), max(lines[:, 1])
    if lines.shape[0] < 2:
        return total_min_x, total_max_x

    min_x, max_x = min(lines[1:, 0]), max(lines[1:, 1])
    first_line_is_full = total_max_x > max_x - indentation_thrsh
    if first_line_is_full:
        return min_x, total_max_x

    return total_min_x, total_max_x
