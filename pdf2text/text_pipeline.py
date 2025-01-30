from .formula_detect import Formula_Detect
from .formula_ocr import FormulaOCR
from .text_ocr import TextOCR


class Text_Extractor():
    def __init__(self):
        self.mfd = Formula_Detect()
        self.mfr = FormulaOCR()

        self.korean_ocr = TextOCR(lang="korean")
        self.english_ocr = TextOCR(lang="en")

    def masking_image(self, formula, img):
        for bbox, _ in formula:
            xmin, ymin, xmax, ymax = bbox
            ymin = max(0, ymin-5)
            ymax = min(img.shape[1], ymax+5)
            img[ymin-5:ymax+5, xmin:xmax] = 255
        return img

    def Recognize_Formula(self, img):
        # self issue
        # 이거 \ 하나만 나와야 하는데 \\로 나옴.
        # 아마 후처리 해야할듯 나중에 ?
        return f'$${self.mfr.ocr(img)}$$'

    def Recognize_Text(self, img, lang):
        # 1. Formula Detection (mfd)
        # input: img, output: [[xmin, ymin, xmax, ymax], label, confidence_score]
        formula_det = self.mfd.detect(img)

        # 2. Formula Recognize (mfr)
        # input: Cropimg, output: [[xmin, ymin, xmax, ymax], text]
        formula_outs = []
        for bbox, label, confidence_score in formula_det:
            xmin, ymin, xmax, ymax = bbox
            crop_img = img[ymin:ymax, xmin:xmax]
            formula_outs.append([bbox, f'${self.mfr.ocr(crop_img)}$'])

        # 3. Text 부분 Detection 및 Recognize (TextOCR)
        # input: img, output: [[xmin, ymin, xmax, ymax], text]
        text_outs = []
        img0 = img.copy()
        masked_image = self.masking_image(formula_outs, img0)

        if lang == "korean":
            ocr_output = self.korean_ocr.ocr(masked_image)[0]
        elif lang == "en":
            ocr_output = self.english_ocr.ocr(masked_image)[0]
        else:
            raise ValueError(
                f"Unsupported language: {lang}. Supported languages are 'korean' and 'en'.")

        if ocr_output is None:
            return ""
        for bbox, tup in ocr_output:
            text, score = tup
            # bbox에서 xmin, ymin, xmax, ymax 추출
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            xmin = int(min(x_coords))
            ymin = int(min(y_coords))
            xmax = int(max(x_coords))
            ymax = int(max(y_coords))
            text_outs.append([[xmin, ymin, xmax, ymax], text])

        outs = [text for box, text in text_outs]
        # print('원본: ' + ' '.join(outs))

        # 4. Text와 수식 같은 line으로 묶는 작업
        line = []
        prev = text_outs[0][0][3]
        for idx, to in enumerate(text_outs):
            cur_min, cur_max = to[0][1], to[0][3]
            y = (cur_min + cur_max) // 2
            if prev >= cur_min:
                line.append(idx)
            else:
                for i in line:
                    text_outs[i][0][3] = prev
                line = [idx]
                prev = y

        if line:
            for i in line:
                text_outs[i][0][3] = prev

        for idx, fo in enumerate(formula_outs):
            fx1, fy1, fx2, fy2 = fo[0]
            for to in text_outs:
                tx1, ty1, tx2, ty2 = to[0]
                y = (fy2+fy1)//2
                if ty1 <= y <= ty2:
                    formula_outs[idx][0][3] = ty2
                    break

        text_outs.extend(formula_outs)
        text_outs.sort(key=lambda x: (x[0][3], x[0][0]))
        output = [text for bbox, text in text_outs]

        return ' '.join(output)
