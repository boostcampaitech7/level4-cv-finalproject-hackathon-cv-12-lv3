from .formula_detect import Formula_Detect
from .formula_ocr import FormulaOCR
from .text_ocr import TextOCR
import cv2

# img : text 정보가 담겨있는 layout
# 
class Text_Extractor() :
    def __init__(self, lang='korean') :
        self.lang = lang
        self.mfd = Formula_Detect()
        self.mfr = FormulaOCR()
        self.text_ocr = TextOCR(lang = self.lang)

    def calculate_iou(self, box1, box2):
        """
        IoU 계산 함수
        box: [xmin, ymin, xmax, ymax]
        """
        # 교집합 영역 계산
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        inter_width = max(0, inter_xmax - inter_xmin)
        inter_height = max(0, inter_ymax - inter_ymin)
        intersection = inter_width * inter_height

        # 각 박스의 면적 계산
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection

        return intersection / union if union != 0 else 0

    def Recognize(self, img) :
        
        # 1. 수식 부분 Detection (mfd)
        # input = img
        # output = [bbox, label, confidenc_score] = [[xmin, ymin, xmax, ymax], label, confidence_score]
        formula_det = self.mfd.detect(img)  


        # 2. 수식 부분 Recognize (mfr)
        # input -> Cropimg
        # output -> formula_outs -> [bbox, text]
        formula_outs = []
        for bbox, label, confidence_score in formula_det :
            xmin, ymin, xmax, ymax = bbox
            crop_img = img[ymin:ymax, xmin:xmax]
            # img 확인
            '''
            path = '/data/ephemeral/home/mh/level4-cv-finalproject-hackathon-cv-12-lv3/testoutput'
            cv2.imwrite(f'{path}/mfd{idx}.jpg', crop_img)
            formula_outs.append(crop_img)
            '''
            formula_outs.append([bbox, self.mfr.ocr(crop_img)])

        # 3. Text 부분 Detection 및 Recognize (TextOCR)
        # input -> img
        # output -> text
        print(formula_outs)
        text_outs = []
        for bbox, tup in self.text_ocr.ocr(img)[0] :
            text, score = tup
            # bbox에서 xmin, ymin, xmax, ymax 추출
            x_coords = [point[0] for point in bbox]  # x 좌표 리스트
            y_coords = [point[1] for point in bbox]  # y 좌표 리스트
            xmin = int(min(x_coords))  # x 좌표의 최소값
            ymin = int(min(y_coords))  # y 좌표의 최소값
            xmax = int(max(x_coords))  # x 좌표의 최대값
            ymax = int(max(y_coords))  # y 좌표의 최대값
            text_outs.append([[xmin,ymin,xmax,ymax], text])

        outs = [text for box, text in text_outs]
        print('원본: ' + ' '.join(outs))

        # 4. Text와 수식 겹치는 Box 점검 x로직
        # 4-1. 2중 for문을 통해 겹치는 부분이 있을시 Text의 해당 Content 수식 Content로 교체
        iou_threshold = 0.05
        for text_entry in text_outs:
            text_bbox, text_content = text_entry
            max_iou = 0
            best_formula = None
            
            # 모든 수식과 IoU 계산
            for formula_entry in formula_outs:
                formula_bbox, formula_content = formula_entry
                iou = self.calculate_iou(text_bbox, formula_bbox)
                
                # 최대 IoU 및 해당 수식 추적
                if iou > max_iou:
                    max_iou = iou
                    best_formula = formula_content
            
            # IoU가 임계값 이상이면 텍스트 교체
            if max_iou >= iou_threshold:
                text_entry[1] = best_formula

        # 추가기능 5. Spell Check ?
        outs = []
        return formula_outs, text_outs