import numpy as np
import re

from fitz import Rect
from collections import defaultdict


def refine_lines(mode, lines, threshold):
    if len(lines) == 0:
        return []

    assert mode in ('rows', 'cols')

    lines = nms(lines, threshold)

    if len(lines) == 1:
        return lines

    # 좌표에 따른 정렬
    axis_range = (1, 3) if mode == 'rows' else (0, 2)

    return sorted(lines, key=lambda x: x['bbox'][axis_range[0]] + x['bbox'][axis_range[1]])


def nms(objects, threshold):
    if len(objects) == 0:
        return []

    objects = sorted(objects, key=lambda x: -x['prob'])
    remove_object = [False] * len(objects)

    # 일정 부분 이상 겹치는 부분 제거
    for obj1_idx in range(1, len(objects)):
        obj1_rect = Rect(objects[obj1_idx]['bbox'])
        obj1_area = obj1_rect.get_area()

        if obj1_area <= 0 or remove_object[obj1_idx]:
            continue

        for obj2 in range(obj1_idx):
            obj2_rect = Rect(objects[obj2]['bbox'])
            inter = obj1_rect.intersect(obj2_rect).get_area()

            score = inter / obj1_area

            if score >= threshold:
                remove_object[obj1_idx] = True
                break
    return [obj for idx, obj in enumerate(objects) if not remove_object[idx]]


def align_lines(mode, lines, standard):
    assert mode in ('rows', 'cols')

    axis_range = (0, 2) if mode == 'rows' else (1, 3)
    for line in lines:
        line['bbox'][axis_range[0]] = standard[axis_range[0]]
        line['bbox'][axis_range[1]] = standard[axis_range[1]]
    return lines


def align_extended_cells(cells, rows, cols):
    # 확장된 셀의 경계를 row, col에 맞게 정렬
    # 확장된 셀이 row와 col과의 경계가 절반이상 겹치지 않으면 제거한다.
    aligned_cells = []

    for cell in cells:
        cell['header'] = False  # 헤더와 관련있는지 판단 여부
        row_rect = None
        col_rect = None
        overlap_row_with_header = set()
        overlap_row_with_data = set()

        for idx, row in enumerate(rows):
            # row와 cell간의 겹치는지 판단 -> 높이로 판단
            row_height = row['bbox'][3] - row['bbox'][1]
            cell_height = cell['bbox'][3] - cell['bbox'][1]

            min_row_overlap = max(row['bbox'][1], cell['bbox'][1])
            max_row_overlap = min(row['bbox'][3], cell['bbox'][3])
            overlap_height = max_row_overlap - min_row_overlap

            if 'span' in cell:
                overlap_fraction = max(overlap_height/row_height,
                                       overlap_height/cell_height)
            else:
                overlap_fraction = overlap_height / row_height

            if overlap_fraction >= 0.5:
                if 'header' in row and row['header']:
                    overlap_row_with_header.add(idx)
                else:
                    overlap_row_with_data.add(idx)

        cell['header'] = False
        # data와 헤더 둘다 겹치면 더 적은 그룹 제거
        if len(overlap_row_with_data) > 0 and len(overlap_row_with_header) > 0:
            if len(overlap_row_with_data) > len(overlap_row_with_header):
                overlap_row_with_header = set()
            else:
                overlap_row_with_data = set()

        # header만 겹치면 cell은 header로 표시
        if len(overlap_row_with_header) > 0:
            cell['header'] = True
        elif 'span' in cell:
            continue

        # 겹치는 행 영역 계산
        overlap_rows = overlap_row_with_data.union(
            overlap_row_with_header)

        for idx in overlap_rows:
            if row_rect is None:
                row_rect = Rect(rows[idx]['bbox'])
            else:
                row_rect = row_rect.include_rect(rows[idx]['bbox'])

        if row_rect is None:
            continue

        overlap_cols = []
        for col_num, col in enumerate(cols):  # 너비로 판단
            col_width = col['bbox'][2] - col['bbox'][0]
            cell_width = cell['bbox'][2] - cell['bbox'][0]

            min_col_overlap = max(col['bbox'][0], cell['bbox'][0])
            max_col_overlap = min(col['bbox'][2], cell['bbox'][2])
            overlap_width = max_col_overlap - min_col_overlap

            if 'span' in cell:
                overlap_fraction = max(overlap_width/col_width,
                                       overlap_width/cell_width)
                if cell['header']:
                    overlap_fraction = overlap_fraction * 2
            else:
                overlap_fraction = overlap_width / col_width

            if overlap_fraction >= 0.5:
                overlap_cols.append(col_num)

                if col_rect is None:
                    col_rect = Rect(col['bbox'])
                else:
                    col_rect = col_rect.include_rect(col['bbox'])

        if col_rect is None:
            continue

        cell_bbox = list(row_rect.intersect(col_rect))
        cell['bbox'] = cell_bbox

        # 하나 이상의 row, col을 포함하면 해당 cell은 정렬된 것으로 판단
        if (len(overlap_rows) > 0 and len(overlap_cols) > 0
                and (len(overlap_rows) > 1 or len(overlap_cols) > 1)):
            cell['row_numbers'] = list(overlap_rows)
            cell['col_numbers'] = overlap_cols
            aligned_cells.append(cell)

            # 현재 처리하려는 pipeline에서는 'span'이라는 속성이 없는 것으로 판단
            if 'span' in cell and cell['header'] and len(cell['col_numbers']) > 1:
                for row_idx in range(0, min(cell['row_numbers'])):
                    new_cell = {'row_numbers': [row_idx], 'col_numbers': cell['col_numbers'],
                                'prob': cell['prob'], 'propagated': True}
                    new_cell_cols = [cols[idx]
                                     for idx in cell['col_numbers']]
                    new_cell_rows = [rows[idx] for idx in cell['row_numbers']]

                    # cell의 bbox 업데이트
                    bbox = [min([col['bbox'][0] for col in new_cell_cols]),
                            min([row['bbox'][1] for row in new_cell_rows]),
                            max([col['bbox'][2]
                                for col in new_cell_cols]),
                            max([row['bbox'][3] for row in new_cell_rows])]

                    new_cell['bbox'] = bbox
                    aligned_cells.append(new_cell)

    return aligned_cells


def nms_extended_cells(cells):
    # 확장된 셀들이 같은 셀에 겹치면 낮은 확률을 가진 확장 셀의 크기를 줄인다.
    cells = sorted(cells, key=lambda x: -x['prob'])
    remove_cells = [False] * len(cells)

    for cell_num1 in range(1, len(cells)):
        cell_obj1 = cells[cell_num1]
        for cell_num2 in range(cell_num1):
            cell_obj2 = cells[cell_num2]
            remove_extended_cell_overlap(cell_obj1, cell_obj2)

        if ((len(cell_obj1['row_numbers']) < 2 and len(cell_obj1['col_numbers']) < 2)
                or len(cell_obj1['row_numbers']) == 0 or len(cell_obj1['col_numbers']) == 0):
            remove_cells[cell_num1] = True

    return [obj for idx, obj in enumerate(cells) if not remove_cells[idx]]


def remove_extended_cell_overlap(cell_obj1, cell_obj2):
    # 확장된 셀간의 중복을 해결한다.
    # 중복된 셀간의 prob를 비교하여 더 낮은 셀에서 row or col을 제거함으로써 해결
    common_rows = set(cell_obj2['row_numbers']).intersection(
        set(cell_obj1['row_numbers']))
    common_cols = set(cell_obj2['col_numbers']).intersection(
        set(cell_obj1['col_numbers']))

    while len(common_rows) > 0 and len(common_cols) > 0:
        if len(cell_obj1['row_numbers']) < len(cell_obj1['col_numbers']):
            min_col = min(cell_obj1['col_numbers'])
            max_col = max(cell_obj1['col_numbers'])

            if max_col in common_cols:
                common_cols.remove(max_col)
                cell_obj1['col_numbers'].remove(max_col)
            elif min_col in common_cols:
                common_cols.remove(min_col)
                cell_obj1['col_numbers'].remove(min_col)
            else:
                cell_obj1['col_numbers'] = []
                common_cols = set()
        else:
            min_row = min(cell_obj1['row_numbers'])
            max_row = max(cell_obj1['row_numbers'])

            if max_row in common_rows:
                common_rows.remove(max_row)
                cell_obj1['row_numbers'].remove(max_row)
            elif min_row in common_rows:
                common_rows.remove(min_row)
                cell_obj1['row_numbers'].remove(min_row)
            else:
                cell_obj1['row_numbers'] = []
                common_rows = set()


def header_extended_cell_tree(cells):
    # Header 속성을 가진 확장셀 추출
    header_cells = [
        cell for cell in cells if 'header' in cell and cell['header']]
    header_cells = sorted(header_cells, key=lambda x: -x['prob'])

    for header_cell in header_cells[:]:
        ancestors_by_row = defaultdict(int)
        min_row = min(header_cell['row_numbers'])

        for header_cell2 in header_cells:
            max_row2 = max(header_cell2['row_numbers'])

            if max_row2 < min_row:
                if (set(header_cell['col_numbers']).issubset(
                        set(header_cell2['col_numbers']))):
                    for row2 in header_cell2['row_numbers']:
                        ancestors_by_row[row2] += 1

        for row in range(min_row):
            if not ancestors_by_row[row] == 1:
                cells.remove(header_cell)
                break


def convert_to_md(cells):
    table = convert_to_element_tree(cells)
    if table.tag != 'table':
        return "Unable to construct the table due to invalid XML input. Not Table"

    md_table = []
    # .	현재 노드를 선택 (상대 경로)
    # 모든 하위 요소를 탐색 //
    # .//th -> 최상위 태그에서 이루어지므로, 모든 <th> 태그를 탐색
    headers = [th.text for th in table.findall(".//th")]

    if headers:
        # markdown 문법의 테이블 구성 양식을 추가
        # | head1 | head2 | head3 |
        md_table.append("| " + " | ".join(headers) + " |")
        # | head1 | head2 | head3 |
        # | --- | --- | --- |
        md_table.append("| " + " | ".join(["---"] * len(headers)) + " |")

    rows = table.findall(".//tr")

    if rows:
        for row in rows:
            md_format_output = [td.text.replace(
                "\n", ' ') for td in row.findall("td")]
            if not md_format_output:
                continue
            md_table.append("| " + " | ".join(md_format_output) + " |")
    else:
        return "Unable to construct the table due to invalid XML input. Not Found Rows"
    return "\n" + "\n".join(md_table) + "\n"


def convert_to_element_tree(cells):
    import xml.etree.ElementTree as ET

    # 열 순서 보장
    cells = sorted(cells, key=lambda cell: min([cell['col_indices']]))
    # 행 순서 보장
    cells = sorted(cells, key=lambda cell: min([cell['row_indices']]))

    # 최상위 <table> 태그 생성
    table = ET.Element("table")
    before_row = -1

    for cell in cells:
        present_row = min(cell['row_indices'])

        properties = {}
        # col_indices가 1보다 크면 병합 열 수를 설정
        col_span = len(cell['col_indices'])
        if col_span > 1:
            properties['col_span'] = str(col_span)

        # row_indices가 1보다 크면 병합 행 수를 설정
        row_span = len(cell['row_indices'])
        if row_span > 1:
            properties['row_span'] = str(row_span)

        # 현재 row가 before_row보다 크면 새로운 행 생성
        if present_row > before_row:
            before_row = present_row
            # cell이 col header 속성을 가지면 <th> 태그 생성
            if cell['col header']:
                cell_tag = "th"
                row = ET.SubElement(table, "thead")
            else:  # 아니면 <td> 태그 생성, <tr> 태그 내부에 생성됨.
                cell_tag = "td"
                row = ET.SubElement(table, "tr")
        new_cell = ET.SubElement(row, cell_tag, attrib=properties)
        new_cell.text = cell.get('cell text', "")
    # <table>
    #     <thead>
    #         <th>Header 1</th>
    #         <th>Header 2</th>
    #     </thead>
    #     <tr>
    #         <td>Cell 1</td>
    #         <td>Cell 2</td>
    #     </tr>
    # </table>
    return table
