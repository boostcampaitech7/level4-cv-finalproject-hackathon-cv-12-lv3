from typing import Dict, List
from enum import Enum

from pdfminer.pdfinterp import PDFGraphicState, PDFResourceManager
from pdfminer.pdffont import PDFCIDFont
from pdfminer.converter import PDFConverter
from pdfminer.pdffont import PDFUnicodeNotDefined
from pdfminer.utils import apply_matrix_pt, mult_matrix
from pdfminer.layout import (
    LTChar,
    LTFigure,
    LTLine,
    LTPage,
)
import logging
import re
import concurrent.futures
import numpy as np
import unicodedata
from tenacity import retry, wait_fixed
from pdf2zh.translator import (
    BaseTranslator, GoogleTranslator, BingTranslator

)
from pymupdf import Font

log = logging.getLogger(__name__)

from pdfminer.layout import LTPage, LTChar, LTFigure
import os



class PDFConverterEx(PDFConverter):
    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
    ) -> None:
        PDFConverter.__init__(self, rsrcmgr, None, "utf-8", 1, None)

    def begin_page(self, page, ctm) -> None:
        # cropbox를 대체하여 재정의
        (x0, y0, x1, y1) = page.cropbox
        (x0, y0) = apply_matrix_pt(ctm, (x0, y0))
        (x1, y1) = apply_matrix_pt(ctm, (x1, y1))
        mediabox = (0, 0, abs(x0 - x1), abs(y0 - y1))
        self.cur_item = LTPage(page.pageno, mediabox)

    def end_page(self, page):
        # 명령 스트림 반환을 재정의
        return self.receive_layout(self.cur_item)

    def begin_figure(self, name, bbox, matrix) -> None:
        # pageid 설정을 재정의
        self._stack.append(self.cur_item)
        self.cur_item = LTFigure(name, bbox, mult_matrix(matrix, self.ctm))
        self.cur_item.pageid = self._stack[-1].pageid

    def end_figure(self, _: str) -> None:
        # 명령 스트림 반환을 재정의
        fig = self.cur_item
        assert isinstance(self.cur_item, LTFigure), str(type(self.cur_item))
        self.cur_item = self._stack.pop()
        self.cur_item.add(fig)
        return self.receive_layout(fig)

    def render_char(
        self,
        matrix,
        font,
        fontsize: float,
        scaling: float,
        rise: float,
        cid: int,
        ncs,
        graphicstate: PDFGraphicState,
    ) -> float:
        # cid와 font 설정을 재정의
        try:
            text = font.to_unichr(cid)
            assert isinstance(text, str), str(type(text))
        except PDFUnicodeNotDefined:
            text = self.handle_undefined_char(font, cid)
        textwidth = font.char_width(cid)
        textdisp = font.char_disp(cid)
        item = LTChar(
            matrix,
            font,
            fontsize,
            scaling,
            rise,
            text,
            textwidth,
            textdisp,
            ncs,
            graphicstate,
        )
        self.cur_item.add(item)
        item.cid = cid  # hack: 원래 문자 인코딩 삽입
        item.font = font  # hack: 원래 문자 폰트 삽입
        return item.adv


class Paragraph:
    def __init__(self, y, x, x0, x1, y0, y1, size, brk):
        self.y: float = y  # 초기 세로 좌표
        self.x: float = x  # 초기 가로 좌표
        self.x0: float = x0  # 왼쪽 경계
        self.x1: float = x1  # 오른쪽 경계
        self.y0: float = y0  # 상단 경계
        self.y1: float = y1  # 하단 경계
        self.size: float = size  # 글자 크기
        self.brk: bool = brk  # 줄 바꿈 여부


# fmt: off
class TranslateConverter(PDFConverterEx):
    def __init__(
        self,
        rsrcmgr,
        vfont: str = None,
        vchar: str = None,
        thread: int = 0,
        layout={},
        lang_in: str = "",
        lang_out: str = "",
        service: str = "",
        noto_name: str = "",
        noto: Font = None,
        envs: Dict = None,
        prompt: List = None,
    ) -> None:
        super().__init__(rsrcmgr)
        self.vfont = vfont
        self.vchar = vchar
        self.thread = thread
        self.layout = layout
        self.noto_name = noto_name
        self.noto = noto
        self.translator: BaseTranslator = None
        param = service.split(":", 1)
        service_name = param[0]
        service_model = param[1] if len(param) > 1 else None
        if not envs:
            envs = {}
        if not prompt:
            prompt = []
        for translator in [GoogleTranslator, BingTranslator]:
            if service_name == translator.name:
                self.translator = translator(lang_in, lang_out, service_model, envs=envs, prompt=prompt)
        if not self.translator:
            raise ValueError("지원되지 않는 번역 서비스입니다.")

    def receive_layout(self, ltpage: LTPage):
        # 단락
        sstk: list[str] = []            # 단락 텍스트 스택
        pstk: list[Paragraph] = []      # 단락 속성 스택
        vbkt: int = 0                   # 단락 수식 괄호 카운트
        # 수식 그룹 스택 및 기타 변수
        vstk: list[LTChar] = []         # 현재 수식 문자
        vlstk: list[LTLine] = []        # 수식 선 그룹
        vfix: float = 0                 # 수식 세로 오프셋
        # 수식 그룹 스택
        var: list[list[LTChar]] = []    # 최종 수식 그룹
        varl: list[list[LTLine]] = []   # 수식 선 그룹 스택
        varf: list[float] = []          # 수식 세로 오프셋 스택
        vlen: list[float] = []          # 수식 길이
        # 전역
        lstk: list[LTLine] = []         # 전역 선 스택
        xt: LTChar = None               # 이전 문자
        xt_cls: int = -1                # 이전 문자의 단락 소속, 첫 번째 문자 단락을 트리거할 수 있도록 설정
        vmax: float = ltpage.width / 4  # 행 내 수식 최대 너비
        ops: str = ""                   # 렌더링 결과

        def vflag(font: str, char: str):    # 수식(및 첨자) 폰트 매칭
            if isinstance(font, bytes):     # decode가 불가능할 수 있으므로 str로 변환
                try:
                    font = font.decode('utf-8')  # UTF-8로 디코딩 시도
                except UnicodeDecodeError:
                    font = ""
            font = font.split("+")[-1]      # 폰트 이름 자르기
            if re.match(r"\(cid:", char):
                return True
            # 폰트 이름 규칙 기반 판단
            if self.vfont:
                if re.match(self.vfont, font):
                    return True
            else:
                if re.match(                                            # latex 폰트
                    r"(CM[^R]|MS.M|XY|MT|BL|RM|EU|LA|RS|LINE|LCIRCLE|TeX-|rsfs|txsy|wasy|stmary|.*Mono|.*Code|.*Ital|.*Sym|.*Math)",
                    font,
                ):
                    return True
            # 문자 집합 규칙 기반 판단
            if self.vchar:
                if re.match(self.vchar, char):
                    return True
            else:
                if (
                    char
                    and char != " "                                     # 공백이 아님
                    and (
                        unicodedata.category(char[0])
                        in ["Lm", "Mn", "Sk", "Sm", "Zl", "Zp", "Zs"]   # 문자 수정자, 수학 기호, 구분 기호
                        or ord(char[0]) in range(0x370, 0x400)          # 그리스 문자
                    )
                ):
                    return True
            return False

############################################################
        # A. 원본 문서 분석
        for child in ltpage:
            if isinstance(child, LTChar):
                cur_v = False
                layout = self.layout[ltpage.pageid]
                # ltpage.height는 fig 내부의 높이일 수 있으므로 여기서는 layout.shape를 사용
                h, w = layout.shape
                # layout에서 현재 문자의 카테고리를 읽어옴
                cx, cy = np.clip(int(child.x0), 0, w - 1), np.clip(int(child.y0), 0, h - 1)
                cls = layout[cy, cx]
                # 문서에서 bullet 위치 고정
                if child.get_text() == "•":
                    cls = 0
                # 수식 여부 판단
                if (
                    cls == 0                                                                                # 1. 보존 영역 카테고리
                    or (cls == xt_cls and len(sstk[-1].strip()) > 1 and child.size < pstk[-1].size * 0.79)  # 2. 첨자 폰트, 0.76과 0.799 사이 값 사용, 첫 글자 확대 고려
                    or vflag(child.fontname, child.get_text())                                              # 3. 수식 폰트
                    or (child.matrix[0] == 0 and child.matrix[3] == 0)                                      # 4. 세로 폰트
                ):
                    cur_v = True

                # 괄호 그룹이 수식인지 판단
                if not cur_v:
                    if vstk and child.get_text() == "(":
                        cur_v = True
                        vbkt += 1
                    if vbkt and child.get_text() == ")":
                        cur_v = True
                        vbkt -= 1
                if (
                    not cur_v                                               # 1. 현재 문자가 수식에 포함되지 않음
                    or cls != xt_cls                                        # 2. 현재 문자와 이전 문자가 동일 단락이 아님
                    or (sstk[-1] != "" and abs(child.x0 - xt.x0) > vmax)    # 3. 단락 내 줄 바꿈
                ):
                    if vstk:
                        if (
                            not cur_v                                       # 1. 현재 문자가 수식에 포함되지 않음
                            and cls == xt_cls                               # 2. 현재 문자와 이전 문자가 동일 단락
                            and child.x0 > max([vch.x0 for vch in vstk])    # 3. 현재 문자가 수식 우측에 위치
                        ):
                            vfix = vstk[0].y0 - child.y0
                        if sstk[-1] == "":
                            xt_cls = -1 # 순수 수식 단락 연결 금지 (sstk[-1]="{v*}"), 이후 문자와의 연결 고려 필요
                        sstk[-1] += f"{{v{len(var)}}}"
                        var.append(vstk)
                        varl.append(vlstk)
                        varf.append(vfix)
                        vstk = []
                        vlstk = []
                        vfix = 0
                if not vstk:
                    if cls == xt_cls:               # 현재 문자와 이전 문자가 동일 단락
                        if child.x0 > xt.x1 + 1:    # 행 내 공백 추가
                            sstk[-1] += " "
                        elif child.x1 < xt.x0:      # 줄 바꿈 공백 추가 및 원본 단락에 줄 바꿈 표시
                            sstk[-1] += " "
                            pstk[-1].brk = True
                    else:                           # 현재 문자로 새로운 단락 생성
                        sstk.append("")
                        pstk.append(Paragraph(child.y0, child.x0, child.x0, child.x0, child.y0, child.y1, child.size, False))
                if not cur_v:                                               # 텍스트 스택에 추가
                    if (
                        child.size > pstk[-1].size                          # 1. 현재 문자가 단락 글자보다 큼
                        or len(sstk[-1].strip()) == 1                       # 2. 현재 문자가 단락 두 번째 글자
                    ) and child.get_text() != " ":                          # 3. 현재 문자가 공백이 아님
                        pstk[-1].y -= child.size - pstk[-1].size            # 단락 초기 세로 좌표 수정
                        pstk[-1].size = child.size
                    sstk[-1] += child.get_text()
                else:                                                       # 수식 스택에 추가
                    if (
                        not vstk                                            # 1. 현재 문자가 수식의 첫 번째 문자
                        and cls == xt_cls                                   # 2. 현재 문자와 이전 문자가 동일 단락
                        and child.x0 > xt.x0                                # 3. 이전 문자가 수식 좌측에 위치
                    ):
                        vfix = child.y0 - xt.y0
                    vstk.append(child)
                # 단락 경계 업데이트 (줄 바꿈 후 수식 시작 가능성을 고려)
                pstk[-1].x0 = min(pstk[-1].x0, child.x0)
                pstk[-1].x1 = max(pstk[-1].x1, child.x1)
                pstk[-1].y0 = min(pstk[-1].y0, child.y0)
                pstk[-1].y1 = max(pstk[-1].y1, child.y1)
                # 이전 문자 업데이트
                xt = child
                xt_cls = cls
            elif isinstance(child, LTFigure):   # 도표
                pass
            elif isinstance(child, LTLine):     # 선
                layout = self.layout[ltpage.pageid]
                # ltpage.height는 fig 내부의 높이일 수 있으므로 여기서는 layout.shape를 사용
                h, w = layout.shape
                # layout에서 현재 선의 카테고리를 읽어옴
                cx, cy = np.clip(int(child.x0), 0, w - 1), np.clip(int(child.y0), 0, h - 1)
                cls = layout[cy, cx]
                if vstk and cls == xt_cls:      # 수식 선
                    vlstk.append(child)
                else:                           # 전역 선
                    lstk.append(child)
            else:
                pass

        # 처리 끝부분
        if vstk:    # 수식 스택 처리
            sstk[-1] += f"{{v{len(var)}}}"
            var.append(vstk)
            varl.append(vlstk)
            varf.append(vfix)
        log.debug("\n==========[VSTACK]==========\n")

        for id, v in enumerate(var):  # 수식 너비 계산
            l = max([vch.x1 for vch in v]) - v[0].x0
            log.debug(f'< {l:.1f} {v[0].x0:.1f} {v[0].y0:.1f} {v[0].cid} {v[0].fontname} {len(varl[id])} > v{id} = {"".join([ch.get_text() for ch in v])}')
            vlen.append(l)

        ############################################################
        # B. 단락 번역
        log.debug("\n==========[SSTACK]==========\n")

        @retry(wait=wait_fixed(1))
        def worker(s: str):  # 다중 스레드 번역
            if not s.strip() or re.match(r"^\{v\d+\}$", s):  # 공백 및 수식은 번역하지 않음
                return s
            try:
                new = self.translator.translate(s)
                return new
            except BaseException as e:
                if log.isEnabledFor(logging.DEBUG):
                    log.exception(e)
                else:
                    log.exception(e, exc_info=False)
                raise e
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.thread
        ) as executor:
            news = list(executor.map(worker, sstk))

 ############################################################
        # C. 새 문서 레이아웃
        def raw_string(fcur: str, cstk: str):  # 문자열 인코딩
            if fcur == self.noto_name:
                return "".join(["%04x" % self.noto.has_glyph(ord(c)) for c in cstk])
            elif isinstance(self.fontmap[fcur], PDFCIDFont):  # 인코딩 길이 판단
                return "".join(["%04x" % ord(c) for c in cstk])
            else:
                return "".join(["%02x" % ord(c) for c in cstk])

        # 목표 언어에 따라 기본 행간 가져오기
        LANG_LINEHEIGHT_MAP = {
            "zh-cn": 1.4, "zh-tw": 1.4, "zh-hans": 1.4, "zh-hant": 1.4, "zh": 1.4,
            "ja": 1.1, "ko": 1.2, "en": 1.2, "ar": 1.0, "ru": 0.8, "uk": 0.8, "ta": 0.8
        }
        default_line_height = LANG_LINEHEIGHT_MAP.get(self.translator.lang_out.lower(), 1.1) # 소수 언어 기본값 1.1
        _x, _y = 0, 0
        ops_list = []

        def gen_op_txt(font, size, x, y, rtxt):
            return f"/{font} {size:f} Tf 1 0 0 1 {x:f} {y:f} Tm [<{rtxt}>] TJ "


        def gen_op_line(x, y, xlen, ylen, linewidth):
            return f"ET q 1 0 0 1 {x:f} {y:f} cm [] 0 d 0 J {linewidth:f} w 0 0 m {xlen:f} {ylen:f} l S Q BT "

        for id, new in enumerate(news):
            x: float = pstk[id].x                       # 단락 초기 가로 좌표
            y: float = pstk[id].y                       # 단락 초기 세로 좌표
            x0: float = pstk[id].x0                     # 단락 왼쪽 경계
            x1: float = pstk[id].x1                     # 단락 오른쪽 경계
            height: float = pstk[id].y1 - pstk[id].y0   # 단락 높이
            size: float = pstk[id].size                 # 단락 글자 크기
            brk: bool = pstk[id].brk                    # 단락 줄 바꿈 표시
            cstk: str = ""                              # 현재 텍스트 스택
            fcur: str = None                            # 현재 폰트 ID
            lidx = 0                                    # 줄 바꿈 횟수 기록
            tx = x
            fcur_ = fcur
            ptr = 0
            log.debug(f"< {y} {x} {x0} {x1} {size} {brk} > {sstk[id]} | {new}")

            ops_vals: list[dict] = []

            while ptr < len(new):
                vy_regex = re.match(
                    r"\{\s*v([\d\s]+)\}", new[ptr:], re.IGNORECASE
                )  # {vn} 수식 표시 매칭
                mod = 0  # 텍스트 수정자
                if vy_regex:  # 수식 로드
                    ptr += len(vy_regex.group(0))
                    try:
                        vid = int(vy_regex.group(1).replace(" ", ""))
                        adv = vlen[vid]
                    except Exception:
                        continue  # 번역기가 잘못된 수식 표시를 자동으로 추가할 수 있음
                    if var[vid][-1].get_text() and unicodedata.category(var[vid][-1].get_text()[0]) in ["Lm", "Mn", "Sk"]:  # 텍스트 수정자
                        mod = var[vid][-1].width
                else:  # 텍스트 로드
                    ch = new[ptr]
                    fcur_ = None
                    try:
                        if fcur_ is None and self.fontmap["tiro"].to_unichr(ord(ch)) == ch:
                            fcur_ = "tiro"  # 기본 라틴 폰트
                    except Exception:
                        pass
                    if fcur_ is None:
                        fcur_ = self.noto_name  # 기본 비라틴 폰트
                    if fcur_ == self.noto_name: # FIXME: CONST로 변경
                        adv = self.noto.char_lengths(ch, size)[0]
                    else:
                        adv = self.fontmap[fcur_].char_width(ord(ch)) * size
                    ptr += 1
                if (
                    fcur_ != fcur                   # 1. 폰트 업데이트
                    or vy_regex                     # 2. 수식 삽입
                    or x + adv > x1 + 0.1 * size    # 3. 오른쪽 경계 도달 (부동 소수점 오차 고려)
                ):
                    if cstk:
                        ops_vals.append({
                            "type": OpType.TEXT,
                            "font": fcur,
                            "size": size,
                            "x": tx,
                            "dy": 0,
                            "rtxt": raw_string(fcur, cstk),
                            "lidx": lidx
                        })
                        cstk = ""
                if brk and x + adv > x1 + 0.1 * size:  # 오른쪽 경계 도달 및 원본 단락 줄 바꿈 존재
                    x = x0
                    lidx += 1
                if vy_regex:  # 수식 삽입
                    fix = 0
                    if fcur is not None:  # 단락 내 수식 세로 오프셋 수정
                        fix = varf[vid]
                    for vch in var[vid]:  # 수식 문자 레이아웃
                        vc = chr(vch.cid)
                        ops_vals.append({
                            "type": OpType.TEXT,
                            "font": self.fontid[vch.font],
                            "size": vch.size,
                            "x": x + vch.x0 - var[vid][0].x0,
                            "dy": fix + vch.y0 - var[vid][0].y0,
                            "rtxt": raw_string(self.fontid[vch.font], vc),
                            "lidx": lidx
                        })
                        if log.isEnabledFor(logging.DEBUG):
                            lstk.append(LTLine(0.1, (_x, _y), (x + vch.x0 - var[vid][0].x0, fix + y + vch.y0 - var[vid][0].y0)))
                            _x, _y = x + vch.x0 - var[vid][0].x0, fix + y + vch.y0 - var[vid][0].y0
                    for l in varl[vid]:  # 수식 선 레이아웃
                        if l.linewidth < 5:  # hack 일부 문서는 굵은 선을 배경으로 사용
                            ops_vals.append({
                                "type": OpType.LINE,
                                "x": l.pts[0][0] + x - var[vid][0].x0,
                                "dy": l.pts[0][1] + fix - var[vid][0].y0,
                                "linewidth": l.linewidth,
                                "xlen": l.pts[1][0] - l.pts[0][0],
                                "ylen": l.pts[1][1] - l.pts[0][1],
                                "lidx": lidx
                            })
                else:  # 텍스트 버퍼 삽입
                    if not cstk:  # 단일 행 시작
                        tx = x
                        if x == x0 and ch == " ":  # 단락 줄 바꿈 공백 제거
                            adv = 0
                        else:
                            cstk += ch
                    else:
                        cstk += ch
                adv -= mod # 텍스트 수정자
                fcur = fcur_
                x += adv
                if log.isEnabledFor(logging.DEBUG):
                    lstk.append(LTLine(0.1, (_x, _y), (x, y)))
                    _x, _y = x, y
            # 처리 끝부분
            if cstk:
                ops_vals.append({
                    "type": OpType.TEXT,
                    "font": fcur,
                    "size": size,
                    "x": tx,
                    "dy": 0,
                    "rtxt": raw_string(fcur, cstk),
                    "lidx": lidx
                })

            line_height = default_line_height

            while (lidx + 1) * size * line_height > height and line_height >= 1:
                line_height -= 0.05

            for vals in ops_vals:
                if vals["type"] == OpType.TEXT:
                    ops_list.append(gen_op_txt(vals["font"], vals["size"], vals["x"], vals["dy"] + y - vals["lidx"] * size * line_height, vals["rtxt"]))
                elif vals["type"] == OpType.LINE:
                    ops_list.append(gen_op_line(vals["x"], vals["dy"] + y - vals["lidx"] * size * line_height, vals["xlen"], vals["ylen"], vals["linewidth"]))

        for l in lstk:  # 전역 선 레이아웃
            if l.linewidth < 5:  # hack 일부 문서는 굵은 선을 배경으로 사용
                ops_list.append(gen_op_line(l.pts[0][0], l.pts[0][1], l.pts[1][0] - l.pts[0][0], l.pts[1][1] - l.pts[0][1], l.linewidth))

        ops = f"BT {''.join(ops_list)}ET "
        return ops


class OpType(Enum):
    TEXT = "text"
    LINE = "line"
