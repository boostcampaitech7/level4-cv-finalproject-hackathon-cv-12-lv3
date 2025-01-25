from pdf2text.text_pipeline import Text_Extractor
import cv2
img = cv2.imread('test.jpg')
model = Text_Extractor(lang = 'korean')
result = model.Recognize(img)
outs = [text for box, text in result]
print('변형: ' + ' '.join(outs))