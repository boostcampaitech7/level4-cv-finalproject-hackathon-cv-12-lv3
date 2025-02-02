import io
import re
import fitz
import torch
import numpy as np

from pathlib import Path
from fitz import Document, Page
from typing import Optional, Union, Tuple, List, Dict, Any
from PIL import Image
from collections import defaultdict
from functools import cmp_to_key


def divide_pdf_lang(page: Union[Page]) -> str:
    """
    PDF 페이지의 주요 언어를 감지하여 'korean' 또는 'en'을 반환합니다.

    Args:
        page (Union[Page]): 분석할 PDF 페이지 객체.

    Returns:
        str: 'korean' 또는 'en' 중 하나.

    예외:
        TypeError : page 인자가 PyMuPDF의 Page 객체가 아닐 경우
        ValueError : 페이지에서 텍스트를 추출할 수 없는 경우 발생.
    """
    if not isinstance(page, Page):
        raise TypeError("page 인자는 PyMuPDF의 Page 객체여야 합니다.")

    # pdf의 첫 페이지로 판단
    total_text = page.get_text()

    if total_text is None or total_text.strip() == "":
        raise ValueError("페이지에서 텍스트를 추출할 수 없습니다.")

    korean_texts = len(re.findall(r'[\uac00-\ud7a3]', total_text))
    # TODO 한국어가 한 글자라도 있으면 korean으로 반환
    english_texts = len(re.findall(r'[a-zA-Z]', total_text))

    # 길이로 판단
    return "korean" if korean_texts >= english_texts else "en"


def pdf_to_image(pdf_path: Union[Document, str, Path]) -> Tuple[List[Image.Image], str]:
    """
    PDF 파일을 이미지로 변환하고, 첫 번째 페이지의 언어를 감지하는 함수.

    Args:
        pdf_path (Union[Document, str, Path]): 변환할 PDF 파일의 경로 또는 PyMuPDF의 Document 객체.

    Returns:
        Tuple[List[PIL.Image.Image], str]: 변환된 이미지 리스트와 감지된 언어 ('korean' 또는 'en').

    예외:
        FileNotFoundError: PDF 파일이 존재하지 않는 경우 발생.
        ValueError: PDF가 비어 있거나 페이지를 읽을 수 없는 경우 발생.
        TypeError: pdf_path의 타입이 잘못된 경우 발생.
    """

    if isinstance(pdf_path, (str, Path)):
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        pages = fitz.open(str(pdf_path))  # PyMuPDF는 Path 객체를 직접 받지 못함, str로 변환
    elif isinstance(pdf_path, Document):
        pages = pdf_path
    else:
        raise TypeError("pdf_path는 str, Path 또는 PyMuPDF의 Document 객체여야 합니다.")

    if len(pages) == 0:
        raise ValueError("PDF 파일이 비어 있습니다.")

    images = []

    lang = divide_pdf_lang(pages[0])

    for page in pages:
        pix = page.get_pixmap(dpi=300)
        img_data = pix.tobytes(output='jpg', jpg_quality=200)
        images.append(Image.open(io.BytesIO(img_data)).convert('RGB'))
    return images, lang


def select_device(device: Optional[str] = None) -> str:
    """
    주어진 장치를 선택하거나 사용 가능한 경우 CUDA를 기본값으로 설정합니다.

    Args:
        device (Optional[str]): 사용할 장치. 'cuda', 'cpu' 또는 None을 입력할 수 있음.
                                None이면 사용 가능한 최적의 장치를 자동으로 선택함.

    Returns:
        str: 선택된 장치 ('cuda' 또는 'cpu').

    예외:
        ValueError: 지원되지 않는 장치 문자열이 입력된 경우 발생.
    """
    valid_devices = {"cuda", "cpu", None}

    if device not in valid_devices:
        raise ValueError(f"지원되지 않는 장치입니다: {device}. 'cuda' 또는 'cpu'를 사용하세요.")

    if device is not None:
        return device

    return "cuda" if torch.cuda.is_available() else "cpu"


def check_box_area(box: np.ndarray, min_h: int, min_w: int) -> bool:
    """
    주어진 박스가 최소 높이 및 너비 조건을 충족하는지 확인합니다.

    Args:
        box (np.ndarray): 4x2 형태의 좌표 배열. 각 행은 박스의 꼭짓점을 나타냅니다.
        min_h (int): 최소 높이 기준.
        min_w (int): 최소 너비 기준.

    Returns:
        bool: 박스가 조건을 충족하면 True, 그렇지 않으면 False.

    예외:
        ValueError: box의 크기가 올바르지 않을 경우 발생.
        TypeError: 입력 타입이 잘못된 경우 발생.
    """
    if not isinstance(box, np.ndarray):
        raise TypeError("box는 NumPy 배열이어야 합니다.")

    if box.shape != (4, 2):
        raise ValueError(f"box의 크기는 (4, 2)이어야 합니다. 현재 크기: {box.shape}")

    return (
        box[0, 0] + min_w <= box[1, 0]
        and box[1, 1] + min_h <= box[2, 1]
        and box[2, 0] >= box[3, 0] + min_w
        and box[3, 1] >= box[0, 1] + min_h
    )


def box2list(bbox: np.ndarray) -> list:
    """
    4x2 형태의 NumPy 배열(Bounding Box)을 리스트 [xmin, ymin, xmax, ymax] 형식으로 변환합니다.

    Args:
        bbox (np.ndarray): 4x2 형태의 좌표 배열.

    Returns:
        list: [xmin, ymin, xmax, ymax] 형식의 리스트.

    예외:
        TypeError: bbox가 NumPy 배열이 아닐 경우 발생.
        ValueError: bbox의 크기가 올바르지 않을 경우 발생.
    """
    if not isinstance(bbox, np.ndarray):
        raise TypeError("bbox는 NumPy 배열이어야 합니다.")

    if bbox.shape != (4, 2):
        raise ValueError(f"bbox의 크기는 (4, 2)이어야 합니다. 현재 크기: {bbox.shape}")

    return [int(bbox[0, 0]), int(bbox[0, 1]), int(bbox[2, 0]), int(bbox[2, 1])]


def list2box(xmin: Union[int, float], ymin: Union[int, float], xmax: Union[int, float], ymax: Union[int, float]) -> np.ndarray:
    """
    [xmin, ymin, xmax, ymax] 형태의 리스트를 4x2 NumPy 배열(Bounding Box)로 변환합니다.

    Args:
        xmin (Union[int, float]): 박스의 최소 x 좌표.
        ymin (Union[int, float]): 박스의 최소 y 좌표.
        xmax (Union[int, float]): 박스의 최대 x 좌표.
        ymax (Union[int, float]): 박스의 최대 y 좌표.

    Returns:
        np.ndarray: 4x2 형태의 좌표 배열.

    예외:
        TypeError: 입력값이 정수가 아닐 경우 발생.
        ValueError: xmax 또는 ymax가 xmin 또는 ymin보다 작은 경우 발생.
    """
    try:
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
    except ValueError:
        raise TypeError("xmin, ymin, xmax, ymax는 모두 숫자여야 합니다.")

    return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])


def sort_boxes(boxes: List[Dict[str, Any]], key: str) -> List[List[Dict[str, Any]]]:
    """
    주어진 박스 리스트를 정렬하고, 라인별로 그룹화합니다.

    Args:
        boxes (List[Dict[str, Any]]): 정렬할 박스들의 리스트. 각 박스는 딕셔너리로 표현되며, 
                                      key에 해당하는 값이 4x2 NumPy 배열이어야 합니다.
        key (str): 박스 좌표를 찾을 딕셔너리 키.

    Returns:
        List[List[Dict[str, Any]]]: 라인별로 정렬된 박스 그룹 리스트.

    예외:
        TypeError: boxes가 리스트가 아닐 경우 발생.
        ValueError: boxes 내부 요소가 올바른 형식이 아닐 경우 발생.
        KeyError: key가 박스 딕셔너리에 없을 경우 발생.
    """
    if not isinstance(boxes, list):
        raise TypeError("boxes는 리스트여야 합니다.")

    for box in boxes:
        if not isinstance(box, dict):
            raise ValueError("boxes 내부 요소는 딕셔너리여야 합니다.")
        if key not in box:
            raise KeyError(f"'{key}' 키가 박스 데이터에 존재하지 않습니다.")
        if not isinstance(box[key], np.ndarray) or box[key].shape != (4, 2):
            raise ValueError(f"'{key}' 키의 값은 (4,2) 크기의 NumPy 배열이어야 합니다.")

    # y축 기준 정렬
    boxes.sort(key=lambda x: x[key][0, 1])

    # 모든 박스의 line_number 초기화
    for box in boxes:
        box['line_number'] = -1

    def get_anchor():
        """ 아직 할당되지 않은 박스를 반환 """
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


def get_same_line_boxes(anchor: Dict[str, Any], total_boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    기준 박스를 중심으로 동일한 라인에 속하는 박스를 찾습니다.

    Args:
        anchor (Dict[str, Any]): 기준 박스.
        total_boxes (List[Dict[str, Any]]): 전체 박스 리스트.

    Returns:
        List[Dict[str, Any]]: 동일한 라인에 속하는 박스 리스트.
    """
    line_boxes = [anchor]
    for box in total_boxes:
        if box['line_number'] >= 0:
            continue
        if max([y_overlap(box, l_box) for l_box in line_boxes]) > 0.1:
            line_boxes.append(box)
    return line_boxes


def sort_and_filter_line_boxes(line_boxes: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    """
    주어진 라인 내 박스를 정렬하고, 좌우 관계를 고려하여 필터링합니다.

    Args:
        line_boxes (List[Dict[str, Any]]): 정렬할 라인의 박스 리스트.
        key (str): 박스 좌표를 찾을 딕셔너리 키.

    Returns:
        List[Dict[str, Any]]: 정렬된 박스 리스트.
    """
    if len(line_boxes) <= 1:
        return line_boxes

    allowed_max_overlay_x = 20

    def find_right_box(anchor):
        """기준 박스의 오른쪽에 위치한 박스를 찾음"""
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

        return sorted(
            right_boxes,
            key=cmp_to_key(lambda x, y: _compare_box(
                x, y, anchor, key, left_best=True))
        )[0]

    def find_left_box(anchor):
        """기준 박스의 왼쪽에 위치한 박스를 찾음"""
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
        return sorted(
            left_boxes,
            key=cmp_to_key(lambda x, y: _compare_box(
                x, y, anchor, key, left_best=False))
        )[-1]

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


def y_overlap(box1: Union[Dict[str, Union[np.ndarray, List[List[int]]]], List[List[int]]],
              box2: Union[Dict[str, Union[np.ndarray, List[List[int]]]], List[List[int]]],
              key: str = 'position') -> float:
    """
    두 박스의 Y축 방향 겹치는 비율을 계산합니다.

    Args:
        box1 (Union[Dict[str, Union[np.ndarray, List[List[int]]]], List[List[int]]]): 첫 번째 박스 좌표 (딕셔너리 또는 리스트)
        box2 (Union[Dict[str, Union[np.ndarray, List[List[int]]]], List[List[int]]]): 두 번째 박스 좌표 (딕셔너리 또는 리스트)
        key (str): 박스가 딕셔너리일 경우, 좌표가 저장된 키 (기본값: 'position')

    Returns:
        float: Y축 방향 겹치는 비율 (0~1)

    예외:
        TypeError: 입력값이 리스트 또는 딕셔너리가 아닐 경우 발생.
        KeyError: key가 존재하지 않을 경우 발생.
        ValueError: 좌표의 형태가 올바르지 않을 경우 발생.
    """
    # Interaction / min(height1, height2)

    # bbox 타입 검사
    if not isinstance(box1, (dict, list)) or not isinstance(box2, (dict, list)):
        raise TypeError("box1과 box2는 리스트 또는 딕셔너리여야 합니다.")

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


def add_edge_margin(image, horizontal_margin, vertical_margin):
    # 입력된 margin을 양쪽에 추가해주는 기능
    # 일반적으로 Text에 사용됨.
    bg_color = get_section_color(image)

    w, h = image.size

    new_w = w + horizontal_margin * 2
    new_h = h + vertical_margin * 2

    new_image = Image.new("RGB", (new_w, new_h), bg_color)
    new_image.paste(image, (horizontal_margin, vertical_margin))

    return new_image


def get_section_color(image, limit=2):
    from collections import Counter
    w, h = image.size

    pixels = [image.getpixel((x, y)) for x in range(w) for y in range(h)
              if x <= limit or y <= limit or x >= w - limit or y >= h - limit]

    return Counter(pixels).most_common(1)[0][0]


def expand_bbox_with_original(image, bbox, horizontal_margin, vertical_margin):
    w, h = image.size

    xmin, ymin, xmax, ymax = bbox

    xmin, xmax = xmin - horizontal_margin, xmax + horizontal_margin
    ymin, ymax = ymin - vertical_margin, ymax + vertical_margin

    xmin, xmax = np.clip([xmin, xmax], 0, w)
    ymin, ymax = np.clip([ymin, ymax], 0, h)

    return (xmin, ymin, xmax, ymax)


def matching_captioning(captions, objects):
    # 위치기반 caption 매칭 로직
    matched_result = {'figure': [], 'table': []}
    unmatched_result = {'obj': [], 'caption': []}

    idx = 0
    for caption_text, caption_bbox, caption_type in captions:
        caption_number = extract_caption_number(caption_text)
        if caption_number is None:
            unmatched_result['caption'].append({"item": captions[idx][0],
                                                "bbox": captions[idx][1],
                                                'type': captions[idx][2]})
            idx += 1
            continue

        matched_object = None
        min_distance = float('inf')

        for obj, obj_bbox, obj_type in objects:
            if obj_type.lower() != caption_type.split("_")[0].lower():
                continue

            cur_distance = calculate_bbox_distance(caption_bbox, obj_bbox)
            if cur_distance < min_distance:
                min_distance = cur_distance
                matched_object = {
                    "caption_number": caption_number,
                    'obj': obj,
                    'obj_bbox': obj_bbox,
                    'caption_bbox': caption_bbox,
                    'caption_text': caption_text
                }
        if matched_object:
            objects = [obj for obj in objects if obj[0]
                       != matched_object['obj']]
            matched_result[obj_type.lower()].append(matched_object)
        else:
            unmatched_result['caption'].append({"text": captions[idx][0],
                                                "bbox": captions[idx][1],
                                                'type': captions[idx][2]})
        idx += 1

    if len(objects) != 0:
        unmatched_result['obj'] = [{"item": obj[0],
                                    "bbox": obj[1],
                                    "type": obj[2]} for obj in objects]

    return matched_result, unmatched_result


def extract_caption_number(caption_text):
    # caption이 몇 번을 나타내는지 확인
    result = re.search(r"(\d+)", caption_text)
    if result:
        return result.group()
    return None


def calculate_bbox_distance(bbox1, bbox2):
    x = abs(bbox1[0] - bbox2[0])
    y = abs(bbox1[1] - bbox2[1])
    return x + y
