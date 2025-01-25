import numpy as np
import re

from fitz import Rect
from functools import cmp_to_key
from collections import defaultdict


def refine_lines(mode, lines, threshold):
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


def check_box_area(box, min_h, min_w):
    return (
        box[0, 0] + min_w <= box[1, 0]
        and box[1, 1] + min_h <= box[2, 1]
        and box[2, 0] >= box[3, 0] + min_w
        and box[3, 1] >= box[0, 1] + min_h
    )


def box2list(bbox):
    return [int(bbox[0, 0]), int(bbox[0, 1]), int(bbox[2, 0]), int(bbox[2, 1])]


def list2box(xmin, ymin, xmax, ymax):
    return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])


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
    def contain_whitespace(s):
        if re.search(r'\s', s):
            return True
        else:
            return False

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


def is_chinese(ch):
    return '\u4e00' <= ch <= '\u9fff'


def find_first_punctuation_position(text):
    pattern = re.compile(r'[,.!?;:()\[\]{}\'\"\\/-]')
    match = pattern.search(text)
    if match:
        return match.start()
    else:
        return len(text)


def cal_block_xmin_xmax(lines, indentation_thrsh):
    total_min_x, total_max_x = min(lines[:, 0]), max(lines[:, 1])
    if lines.shape[0] < 2:
        return total_min_x, total_max_x

    min_x, max_x = min(lines[1:, 0]), max(lines[1:, 1])
    first_line_is_full = total_max_x > max_x - indentation_thrsh
    if first_line_is_full:
        return min_x, total_max_x

    return total_min_x, total_max_x
