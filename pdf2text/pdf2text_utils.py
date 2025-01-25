import re
import torch
import numpy as np

from collections import defaultdict
from functools import cmp_to_key


def select_device(device):
    if device is not None:
        return device

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    return device


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


def clipbox(box, img_height, img_width):
    new_box = np.zeros_like(box)
    new_box[:, 0] = np.clip(box[:, 0], 0, img_width - 1)
    new_box[:, 1] = np.clip(box[:, 1], 0, img_height - 1)
    return new_box
