import numpy as np
import torch


from PIL import Image
from utils import select_device
# from transformers import AutoImageProcessor, AutoModelForObjectDetection
from transformers import TableTransformerForObjectDetection
from torchvision import transforms
from pathlib import Path


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
        'lable': label, 'prob': prob, 'bbox': bbox}

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
