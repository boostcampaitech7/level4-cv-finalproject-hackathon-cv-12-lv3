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
    # english_texts = len(re.findall(r'[a-zA-Z]', total_text))

    # 길이로 판단
    return "korean" if korean_texts else "en"


def pdf_to_image(pdf_path: Union[Document, str, Path, io.BytesIO]) -> Tuple[List[Image.Image], str]:
    """
    PDF 파일을 이미지로 변환하고, 첫 번째 페이지의 언어를 감지하는 함수.

    Args:
        pdf_path (Union[Document, str, Path, io.BytesIO]): 변환할 PDF 파일의 경로 또는 PyMuPDF의 Document 객체.

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
    elif isinstance(pdf_path, io.BytesIO):
        pages = fitz.open(stream=pdf_path, filetype='pdf')
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
    """
    두 개의 박스를 비교하여 정렬 순서를 결정하는 함수.

    Args:
        box1 (dict): 첫 번째 박스.
        box2 (dict): 두 번째 박스.
        anchor (dict): 기준이 되는 앵커 박스.
        key (str): 박스 좌표가 저장된 키.
        left_best (bool): True이면 왼쪽 정렬을 우선, False이면 오른쪽 정렬을 우선.

    Returns:
        int: -1 (box1이 먼저), 1 (box2가 먼저), 또는 두 박스의 X 좌표 차이값.

    예외:
        KeyError: key가 박스에 존재하지 않을 경우 발생.
        ValueError: 박스 좌표가 올바른 형태가 아닐 경우 발생.
    """
    over1 = y_overlap(box1, anchor, key)
    over2 = y_overlap(box2, anchor, key)
    if box1[key][2, 0] < box2[key][0, 0] - 3:
        return -1  # box1이 왼쪽으로 정렬
    elif box2[key][2, 0] < box1[key][0, 0] - 3:
        return 1  # box2가 왼쪽으로 정렬
    else:
        if max(over1, over2) >= 3 * min(over1, over2):
            return over2 - over1 if left_best else over1 - over2
        return box1[key][0, 0] - box2[key][0, 0]


def merge_line_texts(
    outs: List[Dict[str, Union[str, np.ndarray, int]]],
    auto_line_break: bool = True,
    line_sep='\n',
    embed_sep=(' $', '$ '),
    isolated_sep=('$$\n', '\n$$'),
    spellchecker=None,
) -> str:
    """
    OCR 또는 문서 레이아웃 분석 결과를 기반으로 텍스트를 줄 단위로 병합합니다.

    Args:
        outs (List[Dict[str, Union[str, np.ndarray, int]]]): 
            OCR 결과 또는 텍스트 분석 결과 리스트.
            - 'text' (str): 감지된 텍스트.
            - 'position' (np.ndarray): 텍스트의 좌표 (4x2 배열).
            - 'type' (str, optional): 'text', 'embedding', 'isolated' 중 하나.
            - 'line_number' (int, optional): 줄 번호.
        auto_line_break (bool, optional): 
            자동 줄바꿈을 적용할지 여부 (기본값: True).
        line_sep (str, optional): 
            줄 구분자 (기본값: '\n').
        embed_sep (tuple, optional): 
            수식 또는 임베딩 텍스트 감싸는 구분자 (기본값: (' $', '$ ')).
        isolated_sep (tuple, optional): 
            독립된 수식 감싸는 구분자 (기본값: ('$$\n', '\n$$')).
        spellchecker (optional): 
            맞춤법 검사기 함수 (기본값: None).

    Returns:
        str: 병합된 텍스트.
    """

    if not outs:
        return ''

    out_texts = []
    line_margin_list = []
    isolated_included = []
    line_height_dict = defaultdict(list)
    line_ymin_ymax_list = []

    # 줄 단위 텍스트 분류
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

        # 수식 또는 임베딩 텍스트의 구분자 추가
        if cur_type in ('embedding', 'isolated'):
            sep = isolated_sep if _out['type'] == 'isolated' else embed_sep
            cur_text = sep[0] + cur_text + sep[1]

        if cur_type == 'isolated':
            isolated_included[line_number] = True
            cur_text = line_sep + cur_text + line_sep

        out_texts[line_number].append(cur_text)

        # line의 마진 정보 업데이트
        line_margin_list[line_number][1] = max(
            line_margin_list[line_number][1], float(box[2, 0])
        )
        line_margin_list[line_number][0] = min(
            line_margin_list[line_number][0], float(box[0, 0])
        )

        # 일반 텍스트의 높이 정보 저장
        if cur_type == 'text':
            line_height_dict[line_number].append(box[2, 1] - box[1, 1])
            line_ymin_ymax_list[line_number][0] = min(
                line_ymin_ymax_list[line_number][0], float(box[0, 1])
            )
            line_ymin_ymax_list[line_number][1] = max(
                line_ymin_ymax_list[line_number][1], float(box[2, 1])
            )

    # line text 병합
    line_text_list = [smart_join(o) for o in out_texts]

    # 평균 line height 계산
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

    # 줄 길이 기반으로 자동 줄바꿈 결정
    line_lengths = [rx - lx for lx, rx in line_margin_list]
    line_length_thrsh = max(line_lengths) * 0.3

    if line_length_thrsh < 1:
        return default_res

    # 주요 라인 좌표 필터링
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

    # 최종 line text 조정
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


def smart_join(str_list: List[str], spellchecker=None):
    """
    문자열 리스트를 공백 처리 및 맞춤법 검사 로직을 적용하여 하나의 문자열로 병합합니다.

    Args:
        str_list (List[str]): 병합할 문자열 리스트.
        spellchecker (optional): 맞춤법 검사기 객체 (기본값: None).

    Returns:
        str: 병합된 문자열.
    """
    def contain_whitespace(s):
        """ 문자열 공백 문자 포함 확인 """
        if re.search(r'\s', s):
            return True
        else:
            return False

    str_list = [s for s in str_list if s]   # 빈 문자열 제거
    if not str_list:
        return ''

    res = str_list[0]
    for i in range(1, len(str_list)):
        # 마지막 문자와 이후 문자열의 시작 문자가 공백이면 그대로 병합
        if contain_whitespace(res[-1] + str_list[i][0]):
            res += str_list[i]
        # spellchecker를 위한 처리
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
        else:  # 공백 추가 후 병합
            res += ' ' + str_list[i]
    return res


def find_first_punctuation_position(text: str):
    """
    주어진 문자열에서 첫 번째 구두점(문장 부호)의 위치를 찾습니다.

    Args:
        text (str): 검색할 문자열.

    Returns:
        int: 첫 번째 구두점의 인덱스. 구두점이 없으면 문자열 길이를 반환.
    """
    pattern = re.compile(r'[,.!?;:()\[\]{}\'\"\\/-]')  # 패턴 정의
    match = pattern.search(text)  # 매칭되는 패턴 서치
    return match.start() if match else len(text)  # 있으면 첫 번째 구두점 반환


def cal_block_xmin_xmax(lines: np.ndarray, indentation_thrsh: float):
    """
    텍스트 블록의 최소 및 최대 X 좌표를 계산합니다.

    Args:
        lines (np.ndarray): 텍스트 블록의 각 라인의 X 좌표 배열. (N, 2) 형태이며, 
                            각 행은 [line_min_x, line_max_x]로 구성됨.
        indentation_thrsh (float): 들여쓰기 임계값.

    Returns:
        tuple: (min_x, max_x) - 블록의 최소 및 최대 X 좌표.
    """
    total_min_x, total_max_x = min(lines[:, 0]), max(lines[:, 1])

    # line이 하나면 그대로 반환
    if lines.shape[0] < 2:
        return total_min_x, total_max_x

    # 첫 번째 라인을 제외한 최소 및 최대 X값 계산
    min_x, max_x = min(lines[1:, 0]), max(lines[1:, 1])

    # 첫 번째 라인이 전체 블록을 포함하는지 체크
    first_line_is_full = total_max_x > max_x - indentation_thrsh

    if first_line_is_full:
        return min_x, total_max_x

    return total_min_x, total_max_x


def clipbox(box: np.ndarray, img_height: int, img_width: int):
    """
    박스 좌표를 이미지 범위 내로 클리핑합니다.

    Args:
        box (np.ndarray): (N, 2) 형태의 배열로, 각 행이 [x, y] 좌표를 나타냄.
        img_height (int): 이미지의 높이 (Y축 최대값).
        img_width (int): 이미지의 너비 (X축 최대값).

    Returns:
        np.ndarray: 이미지 범위를 벗어나지 않는 조정된 박스 좌표.
    """
    new_box = np.zeros_like(box)
    new_box[:, 0] = np.clip(box[:, 0], 0, img_width - 1)
    new_box[:, 1] = np.clip(box[:, 1], 0, img_height - 1)
    return new_box


def add_edge_margin(image: Union[Image.Image, np.ndarray], horizontal_margin: int, vertical_margin: int):
    """
    입력된 이미지에 지정된 수평 및 수직 마진을 추가하여 새로운 이미지를 생성합니다.

    Args:
        image (PIL.Image or np.ndarray): 마진을 추가할 원본 이미지.
            - `PIL.Image`: PIL 이미지 객체.
            - `np.ndarray`: NumPy 배열 형식의 이미지 (3D 배열, shape: (height, width, channels)).
        horizontal_margin (int): 이미지 양쪽에 추가할 수평 마진의 크기.
        vertical_margin (int): 이미지 상하에 추가할 수직 마진의 크기.

    Returns:
        PIL.Image: 수평 및 수직 마진이 추가된 새로운 이미지.
    """
    # 입력된 margin을 양쪽에 추가해주는 기능
    # 일반적으로 Text에 사용됨.
    if isinstance(image, (Image.Image, np.ndarray)):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
    else:
        raise TypeError(
            f"Invalid Input Type : Expected an instance of Image.Image or np.ndarray, but received {type(image)}")

    bg_color = get_section_color(image)

    w, h = image.size

    new_w = w + horizontal_margin * 2
    new_h = h + vertical_margin * 2

    new_image = Image.new("RGB", (new_w, new_h), bg_color)
    new_image.paste(image, (horizontal_margin, vertical_margin))

    return new_image


def get_section_color(image: Image.Image, limit: int = 2):
    """
    이미지의 외곽(주어진 `limit` 범위 내) 픽셀 중 가장 많이 등장한 색상을 반환합니다.
    주로 배경색 추정을 위해 사용됩니다.

    Args:
        image (PIL.Image): 색상을 추출할 이미지.
            - `PIL.Image`: PIL 이미지 객체.
        limit (int, optional): 이미지 외곽을 구할 때의 범위 한계. 기본값은 2.

    Returns:
        tuple: 가장 많이 등장한 색상의 RGB 값.
    """
    from collections import Counter
    w, h = image.size

    pixels = [image.getpixel((x, y)) for x in range(w) for y in range(h)
              if x <= limit or y <= limit or x >= w - limit or y >= h - limit]

    return Counter(pixels).most_common(1)[0][0]


def expand_bbox_with_original(image: Image.Image, bbox: List[int], horizontal_margin: int, vertical_margin: int):
    """
    주어진 bounding box를 수평 및 수직 마진만큼 확장하고, 확장된 영역이 이미지 크기를 벗어나지 않도록 제한합니다.

    Args:
        image (PIL.Image): bounding box가 적용될 원본 이미지.
        bbox (List[int]): 원본 bounding box (xmin, ymin, xmax, ymax).
            - xmin, ymin: 좌상단 좌표.
            - xmax, ymax: 우하단 좌표.
        horizontal_margin (int): 수평 마진(픽셀 단위)로 bounding box를 확장할 값.
        vertical_margin (int): 수직 마진(픽셀 단위)으로 bounding box를 확장할 값.

    Returns:
        tuple: 확장된 bounding box의 좌표 (xmin, ymin, xmax, ymax).
            - 확장된 좌표는 이미지 크기를 벗어나지 않도록 제한됨.
    """
    w, h = image.size

    xmin, ymin, xmax, ymax = bbox

    xmin, xmax = xmin - horizontal_margin, xmax + horizontal_margin
    ymin, ymax = ymin - vertical_margin, ymax + vertical_margin

    xmin, xmax = np.clip([xmin, xmax], 0, w)
    ymin, ymax = np.clip([ymin, ymax], 0, h)

    return (xmin, ymin, xmax, ymax)


def matching_captioning(captions: List[Dict[str, Union[str, List[int], Image.Image]]],
                        objects: List[Dict[str, Union[str, List[int], Image.Image]]]):
    """
    캡션과 객체의 위치를 기반으로 캡션과 객체를 매칭하는 함수.

    Args:
        captions (List[Dict[str, Union[str, List[int], Image.Image]]]): 캡션과 그에 해당하는 bounding box, 타입을 포함한 리스트.
            각 요소의 key값은 text, bbox, type, image 입니다.
        objects (List[Dict[str, Union[str, List[int], Image.Image]]]): 객체와 그에 해당하는 bounding box, 타입을 포함한 리스트.
            각 항목은 key값은 obj, bbox, type, image 형식입니다.

    Returns:
        tuple: 두 개의 딕셔너리를 반환합니다.
            - matched_result: 매칭된 캡션과 객체를 'figure'와 'table'에 따라 분류하여 반환.
            - unmatched_result: 매칭되지 않은 캡션과 객체를 'caption'과 'obj'에 따라 반환.
    """
    # 위치기반 caption 매칭 로직
    matched_result = {'figure': [], 'table': []}
    unmatched_result = {'obj': [], 'caption': []}

    captions.sort(key=lambda x: (x[1][1], x[1][0]))
    objects.sort(key=lambda x: (x[1][1], x[1][0]))

    for caption in captions:
        caption_number = extract_caption_number(caption['text'])
        if caption_number is None:
            unmatched_result['caption'].append({"item": caption['text'],
                                                "bbox": caption['bbox'],
                                                'type': caption['type'],
                                                'image': caption['image']})
            continue

        matched_object, matched_object_type = None, None
        min_distance = float('inf')

        for obj in objects:
            if obj['type'].lower() != caption['type'].split("_")[0].lower():
                continue

            cur_distance = calculate_bbox_distance(
                caption['bbox'], obj['bbox'])

            if cur_distance < min_distance:
                min_distance = cur_distance
                matched_object_type = obj['type'].lower()
                matched_object = {
                    "caption_number": caption_number,
                    'obj': obj['obj'],
                    'obj_image': obj['image'],
                    'obj_bbox': obj['bbox'],
                    'caption_bbox': caption['bbox'],
                    'caption_text': caption['text'],
                    'caption_image': caption['image']
                }

        if matched_object:
            objects = [obj for obj in objects if obj['obj']
                       != matched_object['obj']]
            matched_result[matched_object_type].append(matched_object)
        else:
            unmatched_result['caption'].append({"item": caption['text'],
                                                "bbox": caption['bbox'],
                                                'type': caption['type'],
                                                'image': caption['image']})

    if len(objects) != 0:
        unmatched_result['obj'] = [{"item": obj['obj'],
                                    "bbox": obj['bbox'],
                                    "type": obj['type'],
                                    "image": obj['image']} for obj in objects]

    return matched_result, unmatched_result


def extract_caption_number(caption_text: str) -> Optional[str]:
    """
    캡션 텍스트에서 숫자를 추출하여 캡션 번호를 반환합니다.

    Args:
        caption_text (str): 캡션 텍스트.

    Returns:
        str or None: 추출된 캡션 번호 또는 숫자가 없으면 None.
    """
    # caption이 몇 번을 나타내는지 확인
    result = re.search(r"(\d+)", caption_text)
    return result.group() if result else None


def calculate_bbox_distance(bbox1: List[int], bbox2: List[int]) -> int:
    """
    두 bounding box 간의 중심점 사이의 유클리드 거리(가까운 정도)를 계산합니다.

    Args:
        bbox1 (List[int]): 첫 번째 bounding box (xmin, ymin, xmax, ymax).
        bbox2 (List[int]): 두 번째 bounding box (xmin, ymin, xmax, ymax).

    Returns:
        int: 두 bounding box 중심점 사이의 거리.
    """
    bbox1_center = [(bbox1[2] + bbox1[0]) / 2, (bbox1[3] + bbox1[1]) / 2]
    bbox2_center = [(bbox2[2] + bbox2[0]) / 2, (bbox2[3] + bbox2[1]) / 2]
    x_diff = abs(bbox1_center[0] - bbox2_center[0])
    y_diff = abs(bbox1_center[1] - bbox2_center[1])
    return int(x_diff + y_diff)
