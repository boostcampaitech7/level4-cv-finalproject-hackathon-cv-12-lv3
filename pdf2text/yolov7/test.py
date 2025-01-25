import sys
from pathlib import Path
import torch
import cv2

if __name__ == '__main__':
    # yolov7 폴더를 경로에 추가
    yolov7_path = str(Path(__file__).resolve().parent / 'yolov7')
    sys.path.append(yolov7_path)

    from models.experimental import attempt_load
    from utils.general import non_max_suppression, scale_coords
    from utils.plots import plot_one_box
    from utils.datasets import letterbox

    # 모델 및 이미지 경로 설정
    model_path = 'best.pt'   # 모델 가중치 파일
    image_path = '/data/ephemeral/home/mh/level4-cv-finalproject-hackathon-cv-12-lv3/testtest.jpg'  # 예측할 이미지
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 로드
    model = attempt_load(model_path, map_location=device)
    model.eval()

    # 이미지 로드 및 전처리
    img = cv2.imread(image_path)
    img_resized = letterbox(img, new_shape=640)[0]
    img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
    img_resized = torch.from_numpy(img_resized).float() / 255.0
    img_resized = img_resized.unsqueeze(0).to(device)

    # 추론 수행
    with torch.no_grad():
        pred = model(img_resized)[0]

    # NMS 적용
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # 결과 시각화
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img_resized.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, cls in det:
                label = f'{model.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img, label=label, color=(0, 255, 0), line_thickness=2)

    # 결과 저장 및 출력
    cv2.imwrite('output.jpg', img)
    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
