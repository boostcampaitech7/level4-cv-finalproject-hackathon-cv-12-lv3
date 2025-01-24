import os
import cv2
import sys
import torch
import numpy as np
from pathlib import Path

# YOLOv7 관련 모듈 불러오기
from pdf2text.yolov7.models.experimental import attempt_load
from pdf2text.yolov7.utils.general import non_max_suppression, scale_coords
from pdf2text.yolov7.utils.plots import plot_one_box
from pdf2text.yolov7.utils.datasets import letterbox
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov7'))

class Formula_Detect :
    def __init__(self, model_path='models/yolo_mfd.pt') :
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.path = os.path.join(script_dir, model_path)
        self.model = attempt_load(self.path, map_location=self.device)

    # image Yolo input에 맞게 전처리 (cv2 -> yolo)
    def prepare_img(self, img) :
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = letterbox(img, new_shape=(1024, 1024), auto=False)[0]
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).float() / 255.0
        img = img.unsqueeze(0).to(self.device)
        return img

    # Detection을 수행하는 함수. (threshold로 임계값 설정 test필요.)
    def detect(self, img, conf_thres=0.3, iou_thres=0.3):
        # Store original image shape
        original_shape = img.shape[:2]  # (height, width)

        # Preprocess the image
        img = self.prepare_img(img)

        with torch.no_grad():
            pred = self.model(img)[0]
        pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)

        results = []
        for det in pred:
            if det is not None and len(det):
                # Rescale boxes to original image dimensions
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], original_shape).round()

                for *bbox, conf, cls in det:
                    bbox = [int(coord) for coord in bbox]
                    label = int(cls)
                    confidence = float(conf)
                    results.append([bbox, label, confidence])
        return results
