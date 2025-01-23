#! pip install transformers>=4.37.0 pillow optimum[onnxruntime]
from PIL import Image
from transformers import TrOCRProcessor
from optimum.onnxruntime import ORTModelForVision2Seq

class FormulaOCR:
    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained('breezedeus/pix2text-mfr')
        self.model = ORTModelForVision2Seq.from_pretrained('breezedeus/pix2text-mfr', use_cache=False)

    def ocr(self, image):
        #images = [Image.open(fp).convert('RGB') for fp in image_fps]
        images = image
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        #print(f'generated_ids: {generated_ids}, \ngenerated text: {generated_text}')
        return [generated_ids, generated_text]
