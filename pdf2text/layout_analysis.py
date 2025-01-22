from enum import Enum
from utils import select_device
from doclayout_yolo import YOLOv10


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
    def __init__(self,
                 model_path,
                 device: str = None):
        # TODO : Layout Analyzer Class
        self.device = select_device(device)
        self.model = YOLOv10(model_path).to(self.device)

    def parse(self, image):
        # TODO : Layout Analysis process
        det_res = self.model.predict(
            image,
            imgsz=1024,
            conf=0.2,
            device=self.device
        )
        # Annotate and save the result
        annotated_frame = det_res[0].plot(
            pil=True, line_width=5, font_size=20)
        return annotated_frame
