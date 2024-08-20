from typing import Any, List, Tuple
import math
import numpy as np


def sort_regions(regions: List[dict],
                 lines: List[Tuple[int, int, int, int]] = None,
                 target_cls: List[int] = None):
    if not target_cls:
        target_cls = [2, 9]
    if lines is None:
        lines = []
    target_regions: List[Tuple[dict, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    stamp_regions: List[Tuple[dict, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    other_regions: List[Tuple[dict, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    for rgn in regions:
        rgn_cls = rgn.get('cls')
        if rgn_cls in target_cls:
            target_regions.append((rgn, __point3_to_cdxdy(rgn.get('region_3point'))))
        elif rgn_cls == 6:
            stamp_regions.append((rgn, __point3_to_cdxdy(rgn.get('region_3point'))))
        else:
            other_regions.append((rgn, __point3_to_cdxdy(rgn.get('region_3point'))))
    if not target_regions:
        return regions
    stamp_rects = [__cdxdy_to_bbox(center, dx, dy) for _, (center, dx, dy) in stamp_regions]
    target_height_list = [np.linalg.norm(dy) * 2 for _, (_, _, dy) in target_regions]
    avg_height = sum(target_height_list) / len(target_height_list)
    grect = __cdxdys_bbox([cdxdy for _, cdxdy in target_regions] + [cdxdy for _, cdxdy in stamp_regions] + [cdxdy for _, cdxdy in other_regions])

    long_lines = []
    for line in lines:
        pos0 = np.array((line[0], line[1]))
        pos1 = np.array((line[2], line[3]))
        length = np.linalg.norm(pos1 - pos0)
        if length > avg_height * 2.0 and not __pos_in_rects(pos0, stamp_rects) and not __pos_in_rects(pos1, stamp_rects):
            long_lines.append((pos0, pos1))
    for rgn, (center, dx, dy) in target_regions:
        width = np.linalg.norm(dx) * 2
        if width > avg_height * 3.0:
            pos0, pos1 = center - dx - dy, center + dx - dy
            if not __pos_in_rects(pos0, stamp_rects) and not __pos_in_rects(pos1, stamp_rects):
                long_lines.append((pos0, pos1))
            pos0, pos1 = center - dx + dy, center + dx + dy
            if not __pos_in_rects(pos0, stamp_rects) and not __pos_in_rects(pos1, stamp_rects):
                long_lines.append((pos0, pos1))

    avg_dx = (sum([dx for _, (_, dx, _) in target_regions]))
    avg_dx = avg_dx / np.linalg.norm(avg_dx)
    avg_dy = (sum([dy for _, (_, _, dy) in target_regions]))
    avg_dy = avg_dy / np.linalg.norm(avg_dy)
    tessellation_size = (31, 31)
    w = int(grect[2] * 1.05)
    h = int(grect[3] * 1.05)
    dir_field = np.zeros((tessellation_size[0], tessellation_size[1], 4), dtype=float)
    xy_mat, ux, uy = _tessellation_directions(dir_field, (w, h), long_lines, (avg_dx, avg_dy))

    mapper = CoordinateInterpretor(xy_mat, ux, uy)
    for rgn, (center, dx, dy) in target_regions:
        center_x, center_y = mapper.map(center)
        half_width = np.linalg.norm(dx)
        half_height = np.linalg.norm(dy)
        rgn['__ltrb'] = (center_x - half_width, center_y - half_height, center_x + half_width, center_y + half_height)

    grouped_rgns: List[List[dict]] = _group_rows(target_regions, lines)
    sorted_rgns: List[dict] = []
    for group in grouped_rgns:
        sorted_rgns.extend(group)
    sorted_rgns.extend([rgn for rgn, _ in stamp_regions])
    sorted_rgns.extend([rgn for rgn, _ in other_regions])
    for rgn in sorted_rgns:
        rgn.pop('__ltrb', None)
    return sorted_rgns


def _group_rows(rgn_cdxdy_list: List[Tuple[dict, Tuple[np.ndarray, np.ndarray, np.ndarray]]], solid_lines: List[Tuple[int, int, int, int]]):
    n_rgns = len(rgn_cdxdy_list)
    if not n_rgns:
        return rgn_cdxdy_list
    dy = sum([dy for _, (_, _, dy) in rgn_cdxdy_list])
    avg_height = np.linalg.norm(dy) / (n_rgns + 0.000000000001)
    rgn_cdxdy_list.sort(key=lambda rgn_cdxdy: rgn_cdxdy[0]['__ltrb'][1])

    lines = []
    current_line = []
    current_line_height = None
    current_y = 0
    for rgn, (_, _, dy) in rgn_cdxdy_list:
        half_height = np.linalg.norm(dy)
        if not current_line:
            current_line.append(rgn)
            current_y = rgn['__ltrb'][1]
            current_line_height = half_height
        elif _height_diff_ratio(current_line_height, half_height) > 0.25:
            current_line.sort(key=lambda rgn: rgn['__ltrb'][0])
            lines.append(current_line)
            current_line = [rgn]
            current_y = rgn['__ltrb'][1]
            current_line_height = half_height
        elif abs(current_y - rgn['__ltrb'][1]) < avg_height * 0.45:
            current_line.append(rgn)
            current_y = rgn['__ltrb'][1]
        else:
            current_line.sort(key=lambda rgn: rgn['__ltrb'][0])
            lines.append(current_line)
            current_line = [rgn]
            current_y = rgn['__ltrb'][1]
            current_line_height = half_height
    if current_line:
        current_line.sort(key=lambda rgn: rgn['__ltrb'][0])
        lines.append(current_line)
    return lines


def _height_diff_ratio(height0, height1):
    diff = abs(height0 - height1)
    larger = max([0, height0, height1]) + 0.001
    return diff / larger


def __point3_to_cdxdy(point3: Tuple[int, int, int, int, int, int]):
    center = np.array((point3[0] + point3[4], point3[1] + point3[5])) * 0.5
    dx = np.array((point3[2] - point3[0], point3[3] - point3[1])) * 0.5
    dy = np.array((point3[4] - point3[2], point3[5] - point3[3])) * 0.5
    return (center, dx, dy)


def __cdxdy_to_bbox(center, dx, dy):
    dwidth = abs(dx[0]) + abs(dy[0])
    dheight = abs(dx[1]) + abs(dy[1])
    return (center[0] - dwidth, center[1] - dheight, center[0] + dwidth, center[1] + dheight)


def __cdxdys_bbox(cdxdy_list: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Tuple[int, int, int, int]:
    x0, y0 = cdxdy_list[0][0][0], cdxdy_list[0][0][1]
    x1, y1 = x0, y0
    for center, dx, dy in cdxdy_list:
        x0 = min(x0, center[0] - abs(dx[0]) - abs(dy[0]))
        x1 = max(x1, center[0] + abs(dx[0]) + abs(dy[0]))
        y0 = min(y0, center[1] - abs(dx[1]) - abs(dy[1]))
        y1 = max(y1, center[1] + abs(dx[1]) + abs(dy[1]))
    return x0, y0, x1 + 1, y1 + 1


def _tessellation_directions(dir_field: np.ndarray,
                             image_size: Tuple[int, int],
                             long_lines: List[Tuple[np.ndarray, np.ndarray]],
                             avg_dir: Tuple[np.ndarray, np.ndarray]):
    w, h = image_size
    dx, dy = avg_dir
    ux, uy = w / (dir_field.shape[1] - 1.0), h / (dir_field.shape[0] - 1.0)
    h_mat = _make_matrix_2d((dir_field.shape[0], dir_field.shape[1]), lambda: [])
    v_mat = _make_matrix_2d((dir_field.shape[0], dir_field.shape[1]), lambda: [])
    h_line_count, v_line_count = 0, 0
    for pos0, pos1 in long_lines:
        vline = pos1 - pos0
        dl = vline / np.linalg.norm(vline)
        if abs(np.dot(dx, dl)) > 0.9:
            _add_to_h_mat(h_mat, pos0, pos1, dl, ux, uy, dir_field.shape[:2])
            h_line_count += 1
        elif abs(np.dot(dy, dl)) > 0.9:
            _add_to_v_mat(v_mat, pos0, pos1, dl, ux, uy, dir_field.shape[:2])
            v_line_count += 1
    if v_line_count == 0:
        for r in range(dir_field.shape[0]):
            for c in range(dir_field.shape[1]):
                v_mat[r][c].append((0, dy))
    if h_line_count == 0:
        for r in range(dir_field.shape[0]):
            for c in range(dir_field.shape[1]):
                h_mat[r][c].append((0, dx))
    _interpolate_h_mat(h_mat, dir_field.shape[:2])
    _interpolate_v_mat(v_mat, dir_field.shape[:2])
    xy_mat = np.zeros((dir_field.shape[0], dir_field.shape[1], 2), dtype=float)
    _fill_xy_gradian(xy_mat, h_mat, v_mat, dir_field.shape[:2], ux, uy)
    return xy_mat, ux, uy


def _make_matrix_2d(sizes: Tuple[int, int], init_func):
    rows: List[List[Any]] = []
    row_count, col_count = sizes
    for _ in range(row_count):
        row = []
        for _ in range(col_count):
            row.append(init_func())
        rows.append(row)
    return rows


class CoordinateInterpretor:
    def __init__(self, xy_mat, ux, uy) -> None:
        self.xy_mat = xy_mat
        self.ux = ux
        self.uy = uy

    def map(self, point2: np.ndarray):
        c0 = math.floor(point2[0] / self.ux)
        c1 = c0 + 1
        r0 = math.floor(point2[1] / self.uy)
        r1 = r0 + 1
        c0 = max(0, min(self.xy_mat.shape[1] - 1, c0))
        c1 = max(0, min(self.xy_mat.shape[1] - 1, c1))
        r0 = max(0, min(self.xy_mat.shape[0] - 1, r0))
        r1 = max(0, min(self.xy_mat.shape[0] - 1, r1))
        kx = point2[0] / self.ux - c0
        ky = point2[1] / self.uy - r0
        p00 = self.xy_mat[r0, c0]
        p01 = self.xy_mat[r0, c1]
        p10 = self.xy_mat[r1, c0]
        p11 = self.xy_mat[r1, c1]
        point = p00 * (1 - kx) * (1 - ky) + p01 * kx * (1 - ky) + p10 * (1 - kx) * ky + p11 * kx * ky
        return (point[0], point[1])


def _fill_xy_gradian(xy_mat, h_mat, v_mat, shape, ux, uy):
    # Y
    for r in range(shape[0]):
        xy_mat[r, 0, 1] = uy * r
    for c in range(1, shape[1]):
        for r in range(shape[0]):
            dp01 = h_mat[r][c - 1][0][1]
            dp1 = h_mat[r][c][0][1]
            if dp01[1] + dp1[1] > 0:
                dp00 = h_mat[r - 1][c - 1][0][1] if r > 0 else dp01
                k = (dp01[1] + dp1[1]) / (2 * uy / ux + dp01[1] - dp00[1])
                y = xy_mat[r, c - 1, 1] - k * uy
            elif dp01[1] + dp1[1] < 0:
                dp00 = h_mat[r + 1][c - 1][0][1] if r < shape[0] - 1 else dp01
                k = (dp01[1] + dp1[1]) / (-2 * uy / ux + dp01[1] - dp00[1])
                y = xy_mat[r, c - 1, 1] + k * uy
            else:
                y = xy_mat[r, c - 1, 1]
            xy_mat[r, c, 1] = y

    # X
    for c in range(shape[1]):
        xy_mat[0, c, 0] = ux * c
    for r in range(1, shape[0]):
        for c in range(shape[1]):
            dp01 = v_mat[r - 1][c][0][1]
            dp1 = v_mat[r][c][0][1]
            if dp01[0] + dp1[0] > 0:
                dp00 = v_mat[r - 1][c - 1][0][1] if c > 0 else dp01
                k = (dp01[0] + dp1[0]) / (2 * ux / uy + dp01[0] - dp00[0])
                x = xy_mat[r - 1, c, 0] - k * ux
            elif dp01[0] + dp1[0] < 0:
                dp00 = v_mat[r - 1][c + 1][0][1] if c < shape[1] - 1 else dp01
                k = (dp01[0] + dp1[0]) / (-2 * ux / uy + dp01[0] - dp00[0])
                x = xy_mat[r - 1, c, 0] + k * ux
            else:
                x = xy_mat[r - 1, c, 0]
            xy_mat[r, c, 0] = x


def _add_to_h_mat(h_mat: List[List[List[Tuple[float, np.ndarray]]]], pos0, pos1, dl, ux, uy, mat_shape):
    if pos0[0] > pos1[0]:
        pos0, pos1 = pos1, pos0
        dl = -dl
    x0, x1 = pos0[0], pos1[0]
    c0 = int(x0 / ux)
    c1 = int(x1 / ux) + 1
    for c in range(c0, c1 + 1):
        y = (c * ux - pos0[0]) * dl[1] / dl[0] + pos0[1]
        r0 = math.floor(y / uy)
        r1 = r0 + 1
        if 0 <= c < mat_shape[1]:
            if 0 <= r0 < mat_shape[0]:
                distance = abs(y - r0 * uy) / uy
                h_mat[r0][c].append((distance, dl))
            if 0 <= r1 < mat_shape[0]:
                distance = abs(y - r1 * uy) / uy
                h_mat[r1][c].append((distance, dl))


def _add_to_v_mat(v_mat: List[List[List[Tuple[float, np.ndarray]]]], pos0, pos1, dl, ux, uy, mat_shape):
    if pos0[1] > pos1[1]:
        pos0, pos1 = pos1, pos0
        dl = -dl
    y0, y1 = pos0[1], pos1[1]
    r0 = int(y0 / uy)
    r1 = int(y1 / uy) + 1
    for r in range(r0, r1 + 1):
        x = (r * uy - pos0[1]) * dl[0] / dl[1] + pos0[0]
        c0 = math.floor(x / ux)
        c1 = c0 + 1
        if 0 <= r < mat_shape[0]:
            if 0 <= c0 < mat_shape[1]:
                distance = abs(x - c0 * ux) / ux
                v_mat[r][c0].append((distance, dl))
            if 0 <= c1 < mat_shape[1]:
                distance = abs(x - c1 * ux) / ux
                v_mat[r][c1].append((distance, dl))


def _interpolate_h_mat(h_mat: List[List[List[Tuple[float, np.ndarray]]]], mat_shape: Tuple[int, int]):
    for r in range(mat_shape[0]):
        for c in range(mat_shape[1]):
            dls = h_mat[r][c]
            if len(dls) > 1:
                v = np.array((0, 0))
                for dist, dl in dls:
                    weight = min(1, max(0, 1 - dist))
                    if weight == 0:
                        weight = 0.001
                    v = v + dl * weight
                v = v / np.linalg.norm(v)
                h_mat[r][c] = [(1, v)]

    valid_lines = []
    for c in range(mat_shape[1]):
        valid_points = []
        for r in range(mat_shape[0]):
            if h_mat[r][c]:
                valid_points.append(r)
        if valid_points:
            r0, r1 = valid_points[0], valid_points[-1]
            for r in range(r0, -1, -1):
                h_mat[r][c] = h_mat[r0][c]
            for r in range(r1, mat_shape[0]):
                h_mat[r][c] = h_mat[r1][c]
            for i, r1 in enumerate(valid_points):
                if i:
                    r0 = valid_points[i - 1]
                    for r in range(r0 + 1, r1):
                        v = h_mat[r0][c][0][1] * (r1 - r) + h_mat[r1][c][0][1] * (r - r0)
                        v = v / np.linalg.norm(v)
                        h_mat[r][c].append((0, v))
            valid_lines.append(c)
    if valid_lines:
        c0, c1 = valid_lines[0], valid_lines[-1]
        for c in range(c0, -1, -1):
            for r in range(mat_shape[0]):
                h_mat[r][c] = h_mat[r][c0]
        for c in range(c1, mat_shape[1]):
            for r in range(mat_shape[0]):
                h_mat[r][c] = h_mat[r][c1]
        for i, c1 in enumerate(valid_lines):
            if i:
                c0 = valid_lines[i - 1]
                for r in range(mat_shape[0]):
                    for c in range(c0 + 1, c1):
                        v = h_mat[r][c0][0][1] * (c1 - c) + h_mat[r][c1][0][1] * (c - c0)
                        v = v / np.linalg.norm(v)
                        h_mat[r][c].append((0, v))


def _interpolate_v_mat(v_mat: List[List[List[Tuple[float, np.ndarray]]]], mat_shape: Tuple[int, int]):
    for c in range(mat_shape[1]):
        for r in range(mat_shape[0]):
            dls = v_mat[r][c]
            if len(dls) > 1:
                v = np.array((0, 0))
                for dist, dl in dls:
                    weight = min(1, max(0, 1 - dist))
                    if weight == 0:
                        weight = 0.001
                    v = v + dl * weight
                v = v / np.linalg.norm(v)
                v_mat[r][c] = [(1, v)]

    valid_lines = []
    for r in range(mat_shape[0]):
        valid_points = []
        for c in range(mat_shape[1]):
            if v_mat[r][c]:
                valid_points.append(c)
        if valid_points:
            c0, c1 = valid_points[0], valid_points[-1]
            for c in range(c0, -1, -1):
                v_mat[r][c] = v_mat[r][c0]
            for c in range(c1, mat_shape[1]):
                v_mat[r][c] = v_mat[r][c1]
            for i, c1 in enumerate(valid_points):
                if i:
                    c0 = valid_points[i - 1]
                    for c in range(c0 + 1, c1):
                        v = v_mat[r][c0][0][1] * (c1 - c) + v_mat[r][c1][0][1] * (c - c0)
                        v = v / np.linalg.norm(v)
                        v_mat[r][c].append((0, v))
            valid_lines.append(r)
    if valid_lines:
        r0, r1 = valid_lines[0], valid_lines[-1]
        for r in range(r0, -1, -1):
            for c in range(mat_shape[1]):
                v_mat[r][c] = v_mat[r0][c]
        for r in range(r1, mat_shape[0]):
            for c in range(mat_shape[1]):
                v_mat[r][c] = v_mat[r1][c]
        for i, r1 in enumerate(valid_lines):
            if i:
                r0 = valid_lines[i - 1]
                for c in range(mat_shape[1]):
                    for r in range(r0 + 1, r1):
                        v = v_mat[r0][c][0][1] * (r1 - r) + v_mat[r1][c][0][1] * (r - r0)
                        v = v / np.linalg.norm(v)
                        v_mat[r][c].append((0, v))


def __pos_in_rects(pos: Tuple[float, float], stamp_ltrbs: List[Tuple[float, float, float, float]]):
    x, y = pos
    for x0, y0, x1, y1 in stamp_ltrbs:
        if x0 < x < x1 and y0 < y < y1:
            return True
    return False