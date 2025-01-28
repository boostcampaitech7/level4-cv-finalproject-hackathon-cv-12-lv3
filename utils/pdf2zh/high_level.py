"""Functions that can be used for the most common use-cases for pdf2zh.six"""
"""pdf2zh.six의 가장 일반적인 사용 사례를 위한 함수들"""
import cv2
import asyncio
import io
import os
import sys
import tempfile
import urllib.request
from asyncio import CancelledError
from pathlib import Path
from typing import Any, BinaryIO, List, Optional, Dict

import numpy as np
import requests
import tqdm
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfexceptions import PDFValueError
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pymupdf import Document, Font

from pdf2zh.converter import TranslateConverter
from pdf2zh.doclayout import OnnxModel
from pdf2zh.pdfinterp import PDFPageInterpreterEx

NOTO_NAME = "noto"

noto_list = [
    "am",  # 암하라어
    "ar",  # 아랍어
    "bn",  # 벵골어
    "bg",  # 불가리아어
    "chr",  # 체로키어
    "el",  # 그리스어
    "gu",  # 구자라트어
    "iw",  # 히브리어
    "hi",  # 힌디어
    "kn",  # 칸나다어
    "ml",  # 말라얄람어
    "mr",  # 마라티어
    "ru",  # 러시아어
    "sr",  # 세르비아어
    "ta",  # 타밀어
    "te",  # 텔루구어
    "th",  # 태국어
    "ur",  # 우르두어
    "uk",  # 우크라이나어
]

def check_files(files: List[str]) -> List[str]:
    files = [
        f for f in files if not f.startswith("http://")
    ]  # 온라인 파일 제외 (http)
    files = [
        f for f in files if not f.startswith("https://")
    ]  # 온라인 파일 제외 (https)
    missing_files = [file for file in files if not os.path.exists(file)]
    return missing_files

def translate_patch(
    inf: BinaryIO,
    pages: Optional[list[int]] = None,
    vfont: str = "",
    vchar: str = "",
    thread: int = 0,
    doc_zh: Document = None,
    lang_in: str = "",
    lang_out: str = "",
    service: str = "",
    noto_name: str = "",
    noto: Font = None,
    callback: object = None,
    cancellation_event: asyncio.Event = None,
    model: OnnxModel = None,
    envs: Dict = None,
    prompt: List = None,
    **kwarg: Any,
) -> None:
    rsrcmgr = PDFResourceManager()
    layout = {}
    device = TranslateConverter(
        rsrcmgr,
        vfont,
        vchar,
        thread,
        layout,
        lang_in,
        lang_out,
        service,
        noto_name,
        noto,
        envs,
        prompt,
    )

    assert device is not None
    obj_patch = {}
    interpreter = PDFPageInterpreterEx(rsrcmgr, device, obj_patch)
    if pages:
        total_pages = len(pages)
    else:
        total_pages = doc_zh.page_count

    parser = PDFParser(inf)
    doc = PDFDocument(parser)
    
    with tqdm.tqdm(total=total_pages) as progress:
        for pageno, page in enumerate(PDFPage.create_pages(doc)):
            if cancellation_event and cancellation_event.is_set():
                raise CancelledError("작업이 취소되었습니다.")
            if pages and (pageno not in pages):
                continue
            progress.update()
            if callback:
                callback(progress)
            page.pageno = pageno
            pix = doc_zh[page.pageno].get_pixmap()
            image = np.fromstring(pix.samples, np.uint8).reshape(
                pix.height, pix.width, 3
            )[:, :, ::-1]            

            page_layout = model.predict(image, imgsz=int(pix.height / 32) * 32)[0]
            # kdtree는 불가능하므로 이미지를 렌더링하여 시간 대신 공간을 사용
            box = np.ones((pix.height, pix.width))
            h, w = box.shape
            vcls = ["abandon", "figure", "table", "isolate_formula", "formula_caption"]
                         
            for i, d in enumerate(page_layout.boxes):
                if page_layout.names[int(d.cls)] not in vcls:
                    x0, y0, x1, y1 = d.xyxy.squeeze()
                    x0, y0, x1, y1 = (
                        np.clip(int(x0 - 1), 0, w - 1),
                        np.clip(int(h - y1 - 1), 0, h - 1),
                        np.clip(int(x1 + 1), 0, w - 1),
                        np.clip(int(h - y0 + 1), 0, h - 1),
                    )
                    box[y0:y1, x0:x1] = i + 2
            for i, d in enumerate(page_layout.boxes):
                if page_layout.names[int(d.cls)] in vcls:
                    x0, y0, x1, y1 = d.xyxy.squeeze()
                    x0, y0, x1, y1 = (
                        np.clip(int(x0 - 1), 0, w - 1),
                        np.clip(int(h - y1 - 1), 0, h - 1),
                        np.clip(int(x1 + 1), 0, w - 1),
                        np.clip(int(h - y0 + 1), 0, h - 1),
                    )
                    box[y0:y1, x0:x1] = 0
            layout[page.pageno] = box
            # 새 xref 생성하여 새로운 명령어 스트림 저장
            page.page_xref = doc_zh.get_new_xref()  # 페이지의 새로운 xref 삽입
            doc_zh.update_object(page.page_xref, "<<>>")
            doc_zh.update_stream(page.page_xref, b"")
            doc_zh[page.pageno].set_contents(page.page_xref)
            interpreter.process_page(page)

    device.close()
    return obj_patch

def translate_stream(
    stream: bytes,
    pages: Optional[list[int]] = None,
    lang_in: str = "",
    lang_out: str = "",
    service: str = "",
    thread: int = 0,
    vfont: str = "",
    vchar: str = "",
    callback: object = None,
    cancellation_event: asyncio.Event = None,
    model: OnnxModel = None,
    envs: Dict = None,
    prompt: List = None,
    **kwarg: Any,
):
    font_list = [("tiro", None)]

    font_path = os.path.join(os.getcwd(), 'utils/pdf2zh/GoNotoKurrent-Regular.ttf')
    noto_name = NOTO_NAME
    noto = Font(noto_name, font_path)
    font_list.append((noto_name, font_path))

    doc_en = Document(stream=stream)
    stream = io.BytesIO()
    doc_en.save(stream)
    doc_zh = Document(stream=stream)
    page_count = doc_zh.page_count
    font_id = {}
    for page in doc_zh:
        for font in font_list:
            font_id[font[0]] = page.insert_font(font[0], font[1])
    xreflen = doc_zh.xref_length()
    for xref in range(1, xreflen):
        for label in ["Resources/", ""]:  # XObj 기반 res일 가능성 있음
            try:  # xref 읽기/쓰기 오류 가능성
                font_res = doc_zh.xref_get_key(xref, f"{label}Font")
                if font_res[0] == "dict":
                    for font in font_list:
                        font_exist = doc_zh.xref_get_key(xref, f"{label}Font/{font[0]}")
                        if font_exist[0] == "null":
                            doc_zh.xref_set_key(
                                xref,
                                f"{label}Font/{font[0]}",
                                f"{font_id[font[0]]} 0 R",
                            )
            except Exception:
                pass

    fp = io.BytesIO()

    doc_zh.save(fp)
    obj_patch: dict = translate_patch(fp, **locals())
    

    for obj_id, ops_new in obj_patch.items():
        doc_zh.update_stream(obj_id, ops_new.encode())

    doc_en.insert_file(doc_zh)
    for id in range(page_count):
        doc_en.move_page(page_count + id, id * 2 + 1)

    doc_zh.subset_fonts(fallback=True)
    doc_en.subset_fonts(fallback=True)
    return (
        doc_zh.write(deflate=True, garbage=3, use_objstms=1),
        doc_en.write(deflate=True, garbage=3, use_objstms=1),
    )

def convert_to_pdfa(input_path, output_path):
    """
    PDF를 PDF/A 형식으로 변환

    Args:
        input_path: 원본 PDF 파일 경로
        output_path: PDF/A 파일 저장 경로
    """
    from pikepdf import Dictionary, Name, Pdf

    pdf = Pdf.open(input_path)

    # PDF/A 적합성 메타데이터 추가
    metadata = {
        "pdfa_part": "2",
        "pdfa_conformance": "B",
        "title": pdf.docinfo.get("/Title", ""),
        "author": pdf.docinfo.get("/Author", ""),
        "creator": "PDF Math Translate",
    }

    with pdf.open_metadata() as meta:
        meta.load_from_docinfo(pdf.docinfo)
        meta["pdfaid:part"] = metadata["pdfa_part"]
        meta["pdfaid:conformance"] = metadata["pdfa_conformance"]

    # OutputIntent 사전 생성
    output_intent = Dictionary(
        {
            "/Type": Name("/OutputIntent"),
            "/S": Name("/GTS_PDFA1"),
            "/OutputConditionIdentifier": "sRGB IEC61966-2.1",
            "/RegistryName": "http://www.color.org",
            "/Info": "sRGB IEC61966-2.1",
        }
    )

    if "/OutputIntents" not in pdf.Root:
        pdf.Root.OutputIntents = [output_intent]
    else:
        pdf.Root.OutputIntents.append(output_intent)

    pdf.save(output_path, linearize=True)
    pdf.close()

def translate(
    files: list[str],
    output: str = "",
    pages: Optional[list[int]] = None,
    lang_in: str = "",
    lang_out: str = "",
    service: str = "",
    thread: int = 0,
    vfont: str = "",
    vchar: str = "",
    callback: object = None,
    compatible: bool = False,
    cancellation_event: asyncio.Event = None,
    model: OnnxModel = None,
    envs: Dict = None,
    prompt: List = None,
    **kwarg: Any,
):
    if not files:
        raise PDFValueError("처리할 파일이 없습니다.")

    missing_files = check_files(files)

    if missing_files:
        print("다음 파일이 존재하지 않습니다:", file=sys.stderr)
        for file in missing_files:
            print(f"  {file}", file=sys.stderr)
        raise PDFValueError("일부 파일이 존재하지 않습니다.")

    result_files = []

    for file in files:
        filename = os.path.splitext(os.path.basename(file))[0]
        doc_raw = open(file, "rb")
        s_raw = doc_raw.read()
        doc_raw.close()

        if file.startswith(tempfile.gettempdir()):
            os.unlink(file)
        s_mono, s_dual = translate_stream(
            s_raw,
            **locals(),
        )
        file_mono = Path(output) / f"{filename}-mono.pdf"
        doc_mono = open(file_mono, "wb")
        doc_mono.write(s_mono)
        doc_mono.close()
        result_files.append(str(file_mono))

    return result_files


def download_remote_fonts(lang: str):
    URL_PREFIX = "https://github.com/timelic/source-han-serif/releases/download/main/"
    LANG_NAME_MAP = {
        **{la: "GoNotoKurrent-Regular.ttf" for la in noto_list},
        **{
            la: f"SourceHanSerif{region}-Regular.ttf"
            for region, langs in {
                "CN": ["zh-cn", "zh-hans", "zh"],
                "TW": ["zh-tw", "zh-hant"],
                "JP": ["ja"],
                "KR": ["ko"],
            }.items()
            for la in langs
        },
    }
    font_name = LANG_NAME_MAP.get(lang, "GoNotoKurrent-Regular.ttf")
    font_path = Path(tempfile.gettempdir(), font_name).as_posix()
    return font_path
