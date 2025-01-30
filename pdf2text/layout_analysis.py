import cv2
import torch
import numpy as np
import torchvision

from PIL import Image
from enum import Enum
from collections import defaultdict

from doclayout_yolo import YOLOv10
from .pdf2text_utils import select_device, list2box, box2list, clipbox


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
    ignored_types = {"abandon", "table_footnote"}

    label2type = {
        "title": ElementType.TITLE,
        "figure": ElementType.FIGURE,
        "plain text": ElementType.TEXT,
        "table": ElementType.TABLE,
        "table_caption": ElementType.TEXT,
        "figure_caption": ElementType.TEXT,
        "isolate_formula": ElementType.FORMULA,
        "inline formula": ElementType.FORMULA,
        "formula_caption": ElementType.PLAIN_TEXT,
        "ocr text": ElementType.TEXT,
    }

    # 독립된 구문, 다른 요소들과 결합될 수 없는 것들
    is_isolated = {"table_caption", "figure_caption", "isolate_formula"}

    def __init__(self,
                 model_path,
                 device: str = None):
        # TODO : Layout Analyzer Class
        self.device = select_device(device)
        self.idx2label = {
            0: "title",
            1: "plain text",
            2: "abandon",
            3: "figure",
            4: "figure_caption",
            5: "table",
            6: "table_caption",
            7: "table_footnote",
            8: "isolate_formula",
            9: "formula_caption",
        }
        self.model = YOLOv10(model_path).to(self.device)

    def parse(self, image, reshape_size=1024, confidence=0.2, iou_threshold=0.45):
        # TODO : Layout Analysis 과정
        if isinstance(image, Image.Image):
            image = image.convert('RGB')

        image_w, image_h = image.size

        # image = self._prepare_img(image, reshape_size)
        det_res = self.model.predict(
            image,
            imgsz=reshape_size,
            conf=confidence,
            device=self.device)[0]

        probs = det_res.__dict__["boxes"].conf
        boxes = det_res.__dict__["boxes"].xyxy
        _classes = det_res.__dict__["boxes"].cls

        indices = torchvision.ops.nms(
            boxes=torch.Tensor(boxes),
            scores=torch.Tensor(probs),
            iou_threshold=iou_threshold
        )

        boxes, probs, _classes = boxes[indices], probs[indices], _classes[indices]
        _classes = _classes.int().tolist()

        layout_result = []
        for box, prob, _cls in zip(boxes, probs, _classes):
            layout_result.append({
                'type': self.idx2label[_cls],
                'position': list2box(*box.tolist()),
                'prob': float(prob)
            })

        ignored_layout = [
            ele for ele in layout_result if ele['type'] in self.ignored_types]

        for ele in ignored_layout:
            ele['col_number'] = -1

        # 결과 형식 변환
        ignored_layout_out, _ = self._convert_format_outputs(
            image_w, image_h, ignored_layout, False
        )

        if layout_result:  # layout_result가 존재하면
            _layout_result = [
                ele for ele in layout_result if ele['type'] not in self.ignored_types
            ]

            layout_output = fetch_column_info(_layout_result, image_w, image_h)
            layout_output, col_meta = self._convert_format_outputs(
                image_w, image_h, layout_output, False
            )
        else:
            layout_output, col_meta = [], {}

        layout_output.extend(ignored_layout_out)

        return layout_output, col_meta

    def _convert_format_outputs(self, w, h, layout_output, table_as_image):
        # 레이아웃 분석 결과를 특정 형식에 맞춰서 변환하는 함수
        col_numbers = set([item["col_number"] for item in layout_output])
        col_meta = defaultdict(dict)

        for col_idx in col_numbers:
            cur_col_res = [
                item for item in layout_output if item["col_number"] == col_idx]
            mean_score = np.mean([item["prob"] for item in cur_col_res])
            xmin, ymin, xmax, ymax = box2list(cur_col_res[0]["position"])
            for item in cur_col_res[1:]:
                cur_xmin, cur_ymin, cur_xmax, cur_ymax = box2list(
                    item["position"])
                xmin = min(xmin, cur_xmin)
                ymin = min(ymin, cur_ymin)
                xmax = max(xmax, cur_xmax)
                ymax = max(ymax, cur_ymax)
            col_meta[col_idx]["position"] = clipbox(
                list2box(xmin, ymin, xmax, ymax), h, w
            )
            col_meta[col_idx]["prob"] = mean_score

        final_out = []
        for box_info in layout_output:
            image_type = box_info["type"]
            if image_type in ["figure_caption", "table_caption", "formula_caption"]:
                caption = image_type
            else:
                caption = None
            isolated = image_type in self.is_isolated
            if image_type in self.ignored_types:
                image_type = ElementType.IGNORED
            else:
                # NOTE 여기서 caption 구분 가능
                image_type = self.label2type.get(
                    image_type, ElementType.UNKNOWN)
            if table_as_image and image_type == ElementType.TABLE:
                image_type = ElementType.FIGURE
            final_out.append(
                {
                    "type": image_type,
                    'caption': caption,
                    "position": clipbox(box_info["position"], h, w),
                    "prob": box_info["prob"],
                    "col_number": box_info["col_number"],
                    "isolated": isolated,
                }
            )

        return final_out, col_meta

    def _prepare_img(self, image, reshape_size, stride=32):
        if isinstance(image, Image.Image):
            img = image.convert("RGB")

        img = np.asarray(image)
        shape = img.shape[:2]
        new_shape = (reshape_size, reshape_size)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
            new_unpad[1]
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img


def fetch_column_info(layout_res, w, h):
    # 레이아웃 분석 결과의 요소들을 col을 기준으로 분류하는 함수
    # xmin값을 기준으로 layout 분석 결과 정렬

    # TODO 정렬 기준 다시 고민해서 처리하기, 전부 같은 col로 처리되는 중
    layout_res.sort(key=lambda x: x["position"][0][0])

    # 열 너비 계산
    col_width = cal_column_width(layout_res, w, h)
    layout_res = locate_full_column(layout_res, col_width, w)

    # full column일 때는 col_width 반환, sub column이면 col 넓이중 넓은 것으로
    col_width = max([item["position"][1][0] - item["position"][0][0]
                    for item in layout_res if item["category"] == "sub column"], default=col_width)

    col_left = w
    cur_col = 1
    for idx, info in enumerate(layout_res):
        if info["category"] == "full column":
            continue
        xmin, xmax = info["position"][0][0], info["position"][1][0]

        if col_left == w:
            col_left = xmin
        # xmin이 현재 열의 오른쪽 경계값에 포함되고, 요소의 너비가 열의 너비를 초과하지 않으면 열에 포함시킨다.
        if xmin < col_left + col_width * 0.99 and xmax - xmin <= col_width * 1.02:
            info["col_number"] = cur_col
            col_left = min(col_left, xmin)
        else:
            cur_col += 1
            col_left = xmin
            info["col_number"] = cur_col

    # col_number, 왼쪽 위쪽 모서리의 y, x값 기준으로 정렬
    layout_res.sort(key=lambda x: (
        x["col_number"], x["position"][0][1], x["position"][0][0]))
    return layout_res


def cal_column_width(layout_res, w, h):
    # 열의 평균 너비를 계산하는 함수
    widths = [item["position"][1][0] - item["position"][0][0]
              for item in layout_res]  # xmax - xmin
    if len(widths) <= 2:  # 요소가 2개 이하면 너비중 가장 작은 값과 문서 전체 너비 중 최소값 반환
        return min(widths + [w])

    boxes_info = []
    for item in layout_res:
        xmin, ymin = item["position"][0]  # xmin, ymin
        xmax, ymax = item["position"][2]  # xmax, ymax
        width = xmax - xmin
        height = ymax - ymin
        area = width * height
        boxes_info.append({"width": width, "area": area,
                          "y0": ymin, "height": height})

    # 면접 기준 내림차순 정렬
    boxes_info.sort(key=lambda x: x["area"], reverse=True)

    total_weight = 0
    weighted_width_sum = 0

    # 상위 30% 요소 선택, 최소 2개 이상 요소 선택
    top_boxes = boxes_info[: max(2, int(len(boxes_info) * 0.3))]

    # 가중치 평균 열 너비 계산
    for box in top_boxes:
        weight = box["area"]
        # 요소 상단 경계 값이 문서 중간 아래 일경우 가중치 증가
        if box["y0"] > h * 0.5:
            weight *= 1.5

        weighted_width_sum += box["width"] * weight
        total_weight += weight

    estimated_width = (weighted_width_sum /
                       total_weight if total_weight > 0 else w)

    min_width = w * 0.3  # 최소 너비는 전체의 30%
    max_width = w * 0.95  # 최대 너비는 전체의 95%

    return min(max(estimated_width, min_width), max_width)


def locate_full_column(layout_res, col_width, img_width):
    for item in layout_res:
        # xmax - xmin
        cur_width = item["position"][1][0] - item["position"][0][0]
        if cur_width > col_width * 1.5 or cur_width > img_width * 0.7:
            item["category"] = "full column"
            item["col_number"] = 0
        else:
            item["category"] = "sub column"
            item["col_number"] = -1
    return layout_res
