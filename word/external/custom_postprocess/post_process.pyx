import numpy as np
from shapely.geometry import Polygon
from math import sqrt, sin, atan2, inf, pi
from typing import Dict
from copy import deepcopy
from scipy import odr


cdef inline float distance(list p1, list p2):
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


cpdef inline bint check_is_same_point(list p0, list p1, float max_offset=4.0):
    return max(abs(p0[0] - p1[0]), abs(p0[1] - p1[1])) <= max_offset


def average_points(points):
    x, y = 0, 0
    for p in points:
        x += p[0]
        y += p [1]
    x /= len(points)
    y /= len(points)
    return [x, y]


def calc_angle(p1, p2, axis):
    another_axis = 1 - axis
    radian = atan2(p2[another_axis] - p1[another_axis], p2[axis] - p1[axis])
    angle = radian / pi * 180
    return angle


def bbox2rect(bbox):
    min_x = min(p[0] for p in bbox)
    max_x = max(p[0] for p in bbox)
    min_y = min(p[1] for p in bbox)
    max_y = max(p[1] for p in bbox)
    return [min_x, min_y, max_x, max_y]


def get_center_point(bbox):
    x = sum([p[0] for p in bbox]) / 4
    y = sum([p[1] for p in bbox]) / 4
    return x, y


def linregress(line):
    # improve: vertical line need swap x and y, because c[1] is very large
    x = []
    y = []
    unique_x = True
    unique_y = True
    for seg in line:
        for p in seg[:2]:
            x.append(round(p[0]))
            y.append(round(p[1]))

        if x[-2] != x[0] or x[-1] != x[0]:
            unique_x = False
        if y[-2] != y[0] or y[-1] != y[0]:
            unique_y = False

    if unique_x:
        return [inf, x[0]]
    elif unique_y:
        return [0, y[0]]
    else:
        model = odr.Model(lambda beta, x: beta[0]*x + beta[1])
        data = odr.Data(x, y)
        regressor = odr.ODR(data, model, beta0=[0., 0.])
        output = regressor.run()
        return output.beta


def get_line_cross_point(line1, line2):
    def calc_abc_from_line_2d(x0, y0, x1, y1):
        a = y0 - y1
        b = x1 - x0
        c = x0 * y1 - x1 * y0
        return a, b, c

    a0, b0, c0 = calc_abc_from_line_2d(line1[0][0], line1[0][1], line1[1][0], line1[1][1])
    a1, b1, c1 = calc_abc_from_line_2d(line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    D = a0 * b1 - a1 * b0
    if D == 0:
        return None
    x = (b0 * c1 - b1 * c0) / D
    y = (a1 * c0 - a0 * c1) / D
    return [x, y]


def splice_cells(cells, r=0.1, min_offset=15.0, dist_thresh=30):
    if len(cells) == 0:
        return cells

    def find_root(node):
        f = fa[node]
        fa[node] = f if f == node else find_root(f)
        return fa[node]

    def merge(node1, node2):
        root1 = find_root(node1)
        root2 = find_root(node2)
        if root1 != root2:
            fa[root1] = root2
            group[root2].append(root1)
            group[root1].remove(root1)

    def search(node, root):
        children = [node]
        for child in group[node]:
            if child != node:
                children.extend(search(child, root))
            fa[child] = root
        group[node] = children if node == root else []
        return children

    def valid_merge(node1, node2):
        root1 = find_root(node1)
        root2 = find_root(node2)
        if root1 == root2:
            return True

        nodes = search(root1, root1) + search(root2, root2)
        nodes.sort()
        for i in range(1, len(nodes)):
            if nodes[i]//4 == nodes[i-1]//4:
                return False

        return True

    def propagation(root, direction: int=-1, limit=True):
        nodes = search(root, root)
        if len(nodes) <= 1:
            return 

        for i, node_i in enumerate(nodes):
            for node_j in nodes[i+1:]:
                cell_i_idx, cell_j_idx = node_i//4, node_j//4
                local_pi_idx, local_pj_idx = node_i%4, node_j%4
                if local_pi_idx > local_pj_idx:
                    cell_i_idx, cell_j_idx = cell_j_idx, cell_i_idx
                    local_pi_idx, local_pj_idx = local_pj_idx, local_pi_idx

                flag = None
                # vertical connect
                if local_pi_idx == 0 and local_pj_idx == 3:
                    local_pk_idx, local_pl_idx = 1, 2
                    next_dir = 0
                    flag = abs(ws[cell_i_idx] - ws[cell_j_idx]) <= dist_thresh
                elif local_pi_idx == 1 and local_pj_idx == 2:
                    local_pk_idx, local_pl_idx = 0, 3
                    next_dir = 1
                    flag = abs(ws[cell_i_idx] - ws[cell_j_idx]) <= dist_thresh
                # horizontal connect
                elif local_pi_idx == 0 and local_pj_idx == 1:
                    local_pk_idx, local_pl_idx = 3, 2
                    next_dir = 2
                    flag = abs(hs[cell_i_idx] - hs[cell_j_idx]) <= h_dist_thresh
                elif local_pi_idx == 2 and local_pj_idx == 3:
                    local_pk_idx, local_pl_idx = 1, 0
                    next_dir = 3
                    flag = abs(hs[cell_i_idx] - hs[cell_j_idx]) <= h_dist_thresh

                if flag is not None:
                    global_pk_idx, global_pl_idx = 4*cell_i_idx+local_pk_idx, 4*cell_j_idx+local_pl_idx
                    root_k, root_l = find_root(global_pk_idx), find_root(global_pl_idx)
                    if root_k == root_l:
                        continue

                    if not limit and not flag:
                        if (next_dir <= 1 and ws[cell_i_idx] > ws[cell_j_idx]) or \
                            (next_dir > 1 and hs[cell_i_idx] > hs[cell_j_idx]):
                                local_pk_idx, local_pl_idx = local_pl_idx, local_pk_idx
                                global_pk_idx, global_pl_idx = global_pl_idx, global_pk_idx
                                root_k, root_l = root_l, root_k

                        if len(group[root_k]) == 1:
                            flag = True
                            # check same local point
                            nodes_l = search(root_l, root_l)
                            for node in nodes_l:
                                if node % 4 == local_pk_idx:
                                    flag = False
                    
                    if flag and valid_merge(root_k, root_l):
                        merge(root_k, root_l)
                        # if not limit:
                        #     cells[global_pk_idx // 4]['bbox'][global_pk_idx % 4] = deepcopy(cells[global_pl_idx // 4]['bbox'][global_pl_idx % 4])

                        if direction == -1 or next_dir == direction:
                            propagation(root_l, next_dir, limit)

    def splice_points():
        for i in range(len(fa)):
            if fa[i] == i:
                nodes = search(i, i)
                if len(nodes) > 1:
                    points = [cells[idx // 4]['bbox'][idx % 4] for idx in nodes]
                    ave_point = average_points(points)
                    for idx in nodes:
                        cells[idx // 4]['bbox'][idx % 4] = deepcopy(ave_point)

    fa = []
    group = []
    for i in range(4 * len(cells)):
        fa.append(i)
        group.append([i])

    ws, hs, centers, xoffsets, yoffsets = [], [], [], [], []
    for cell in cells:
        bbox = cell['bbox']

        center = get_center_point(bbox)
        centers.append(center)

        w = (distance(bbox[0], bbox[1]) + distance(bbox[2], bbox[3])) / 2
        h = (distance(bbox[0], bbox[3]) + distance(bbox[1], bbox[2])) / 2
        ws.append(w)
        hs.append(h)

    h_mean = np.mean(hs)
    ratio = 1.4 * dist_thresh / h_mean
    h_min_offset = min(max(5.0, min_offset / ratio), min_offset)
    h_dist_thresh = min(max(10.0, dist_thresh / ratio), dist_thresh)

    for i in range(len(cells)):
        xoffset = max(r * ws[i], min_offset)
        yoffset = max(r * hs[i], h_min_offset)
        xoffsets.append(xoffset)
        yoffsets.append(yoffset)

    matched = [([], []) for i in range(len(cells))]
    for i in range(len(cells)):
        for j in range(len(cells)):
            if abs(centers[i][0] - centers[j][0]) <= ws[i] + ws[j] + dist_thresh and \
                abs(centers[i][1] - centers[j][1]) <= hs[i] + hs[j] + h_dist_thresh:
                if j < i:
                    matched[i][0].append(j)
                elif j > i:
                    matched[i][1].append(j)

    # step 1: 
    # for each vertex of cells, 
    # calculate the distance from all the vertices of other cells to this vertex, 
    # if the distance is within a certain range, 
    # calculate the mean value of these vertex coordinates, and then assign the value to all these vertices.

    edges = []
    for i, cell_i in enumerate(cells):
        bbox_i = cell_i['bbox']
        for j in matched[i][1]:
            bbox_j = cells[j]['bbox']
            for k, p_i in enumerate(bbox_i):
                for l, p_j in enumerate(bbox_j):  
                    xdist = abs(p_i[0] - p_j[0])
                    ydist = abs(p_i[1] - p_j[1])
                    dist = distance(p_i, p_j)
                    if xdist <= min(xoffsets[i], xoffsets[j]) and ydist <= min(yoffsets[i], yoffsets[j]) and dist <= dist_thresh:
                        global_pi_idx, global_pj_idx = 4*i+k, 4*j+l 
                        edges.append([global_pi_idx, global_pj_idx, dist])

    edges.sort(key=lambda edge: edge[2])
    for edge in edges:
        global_pi_idx, global_pj_idx = edge[0], edge[1]
        if valid_merge(global_pi_idx, global_pj_idx):
            merge(global_pi_idx, global_pj_idx)

    for i, f in enumerate(fa):
        if i == f:
            propagation(f)

    splice_points()

    # step 2: 
    # for all the points that have not been refined in step 1, 
    # find the nearest refined vertex, 
    # if a vertex satisfies the condition in step 1, 
    # the points that have not been refined are spelled onto the output from step 1.

    for i, cell_i in enumerate(cells):
        bbox_i = cell_i['bbox']
        for k, p_i in enumerate(bbox_i):
            global_pi_idx = 4 * i + k
            if not (fa[global_pi_idx] == global_pi_idx and len(group[global_pi_idx]) == 1):
                continue

            min_dist = inf
            nearest_p_idx = None
            for j in matched[i][0] + matched[i][1]:
                bbox_j = cells[j]['bbox']
                for l, p_j in enumerate(bbox_j):
                    global_pj_idx = 4 * j + l
                    if fa[global_pj_idx] == global_pj_idx and len(group[global_pj_idx]) == 1:
                        continue

                    xdist = abs(p_i[0] - p_j[0])
                    ydist = abs(p_i[1] - p_j[1])
                    dist = distance(p_i, p_j)
                    if xdist <= min(xoffsets[i], xoffsets[j]) and ydist <= min(yoffsets[i], yoffsets[j]) and dist <= dist_thresh:
                        if dist < min_dist:
                            if valid_merge(global_pi_idx, global_pj_idx):
                                min_dist = dist
                                nearest_p_idx = global_pj_idx
            
            if nearest_p_idx is not None:
                merge(global_pi_idx, nearest_p_idx)
                # nearest_p = cells[nearest_p_idx // 4]['bbox'][nearest_p_idx % 4]
                # cells[i]['bbox'][k] = deepcopy(nearest_p)

    for i, f in enumerate(fa):
        if i == f:
            propagation(f)
            
    for i, f in enumerate(fa):
        if i == f:
            propagation(f, limit=False)

    splice_points()
    
    return cells


def merge_lines(lines, direction, min_dist_thresh=5.0, max_dist_thresh=10.0, min_angle_thresh=1.0, max_angle_thresh=2.0):
    axis = int(direction == 'vertical')

    def merge(line_i, line_j):
        new_line = line_i + line_j
        new_line.sort(key=lambda seg: seg[0][axis])
        return new_line

    def merge_line(line_i, line_j, idx_i, idx_j):
        # check validity
        for seg_i in line_i:
            for seg_j in line_j:
                if seg_i[2] // 4 == seg_j[2] // 4: # same cell
                    return None

        for seg_i in line_i:
            for seg_j in line_j:
                if check_is_same_point(seg_i[0], seg_j[0]) or check_is_same_point(seg_i[1], seg_j[1]):
                    return merge(line_i, line_j)

        left_p_i = line_i[0][0]
        right_p_i = line_i[-1][1]
        left_p_j = line_j[0][0]
        right_p_j = line_j[-1][1]

        angles = []
        for p_i in [left_p_i, right_p_i]:
            for p_j in [left_p_j, right_p_j]:
                p1, p2 = p_i, p_j
                if p_i[axis] > p_j[axis]:
                    p1, p2 = p2, p1

                angle = calc_angle(p1, p2, axis)
                angles.append(angle)
        
        ave_angle_i = sum([calc_angle(seg[0], seg[1], axis) for seg in line_i]) / len(line_i)
        ave_angle_j = sum([calc_angle(seg[0], seg[1], axis) for seg in line_j]) / len(line_j)
        angles = [abs(angle_k - angle) for angle_k in angles for angle in [ave_angle_i, ave_angle_j]]
        min_len = min(distance(left_p_i, right_p_i), distance(left_p_j, right_p_j))
        dynamic_angle_thresh = min(max(0.001*min_len, min_angle_thresh), max_angle_thresh)
        if sum([1 for angle in angles if angle <= dynamic_angle_thresh]) >= 5:
            return merge(line_i, line_j)

        if coefficients[idx_i] is None:
            coefficients[idx_i] = linregress(line_i)
        if coefficients[idx_j] is None:
            coefficients[idx_j] = linregress(line_j)
        coeff_i = coefficients[idx_i]
        coeff_j = coefficients[idx_j]

        distances = []
        for p in [left_p_i, right_p_i]:
            dist = abs(p[0] - coeff_j[1]) if coeff_j[0] == inf else \
                abs(coeff_j[0] * p[0] - p[1] + coeff_j[1]) / sqrt(coeff_j[0]**2 + 1)
            distances.append(dist)
        for p in [left_p_j, right_p_j]:
            dist = abs(p[0] - coeff_i[1]) if coeff_i[0] == inf else \
                abs(coeff_i[0] * p[0] - p[1] + coeff_i[1]) / sqrt(coeff_i[0]**2 + 1)
            distances.append(dist)

        dynamic_dist_thresh = min(max(0.05*min_len, min_dist_thresh), max_dist_thresh)
        if sum([1 for dist in distances if dist <= dynamic_dist_thresh]) >= 2:
            return merge(line_i, line_j)
        
        return None

    lines.sort(key=lambda line: line[0][0][axis])

    coefficients = len(lines) * [None]
    keep = len(lines) * [False]
    for i, line_i in enumerate(lines):
        keep[i] = True
        for j, line_j in enumerate(lines[i+1:]):
            j += i + 1
            new_line = merge_line(line_i, line_j, i, j)
            if new_line is not None:
                lines[j] = new_line
                coefficients[j] = None
                keep[i] = False
                break
    lines = [line for i, line in enumerate(lines) if keep[i]]

    return lines


def merge_edges(edges, direction):
    keep = len(edges) * [False]
    for i, edge_i in enumerate(edges):
        keep[i] = True
        for j, edge_j in enumerate(edges[i+1:]):
            j += i + 1
            if check_is_same_point(edge_i[-1][1], edge_j[0][0]):
                edges[j] = edge_i + edge_j
                keep[i] = False
                break
            elif check_is_same_point(edge_j[-1][1], edge_i[0][0]):
                edges[j] = edge_j + edge_i
                keep[i] = False
                break
    lines = [edge for i, edge in enumerate(edges) if keep[i]]
    lines = merge_lines(lines, direction)

    return lines


def calc_logical_coordinates(cells):
    # First, split every cell into 4 bounding edges
    left_and_right_edges = []
    up_and_down_edges = []
    for i, cell in enumerate(cells):
        bbox = cell['bbox']
        left_edge = [[bbox[0], bbox[3], 4*i]]
        right_edge = [[bbox[1], bbox[2], 4*i+1]]
        left_and_right_edges.extend([left_edge, right_edge])

        up_edge = [[bbox[0], bbox[1], 4*i+2]]
        down_edge = [[bbox[3], bbox[2], 4*i+3]]
        up_and_down_edges.extend([up_edge, down_edge])

    # then merge the up edges and down edges to horizontal lines and merge left edges and right edges to vertical lines according to cell connectivity
    horizontal_lines = merge_edges(up_and_down_edges, direction='horizontal')
    vertical_lines = merge_edges(left_and_right_edges, direction='vertical')

    # Next, sort horizontal lines, vertical lines and index them from 0
    horizontal_lines.sort(key=lambda line: sum([seg[0][1]+seg[1][1] for seg in line]) / len(line))
    vertical_lines.sort(key=lambda line: sum([seg[0][0]+seg[1][0] for seg in line]) / len(line))

    # Finally, ranks cells by line index and outputs row/column information
    for rk, line in enumerate(horizontal_lines + vertical_lines):
        for seg in line:
            for index in seg[2:]:
                cell_index = index // 4
                edge_index = index % 4

                if cells[cell_index].get('logical_coordinates') is None:
                    cells[cell_index]['logical_coordinates'] = {
                        'startrow': None,
                        'endrow': None,
                        'startcol': None,
                        'endcol': None
                    }

                if edge_index == 0: # leftedge
                    cells[cell_index]['logical_coordinates']['startcol'] = rk - len(horizontal_lines)
                elif edge_index == 1: # rightedge
                    cells[cell_index]['logical_coordinates']['endcol'] = rk - len(horizontal_lines) - 1
                elif edge_index == 2: # upedge
                    cells[cell_index]['logical_coordinates']['startrow'] = rk
                elif edge_index == 3: # downedge
                    cells[cell_index]['logical_coordinates']['endrow'] = rk - 1

    return cells


def pcs_filter(cells, table_rect=None, max_offset=4.0):
    new_cells = []

    for cell in cells:
        bbox = cell['bbox']
        if table_rect is not None:
            center = get_center_point(bbox)
            if not (table_rect[0] <= center[0] and center[0] <= table_rect[2] and \
                table_rect[1] <= center[1] and center[1] <= table_rect[3]):
                continue

        has_same_point = False
        for i, kp_i in enumerate(bbox):
            for kp_j in bbox[i+1:]:
                if check_is_same_point(kp_i, kp_j):
                    has_same_point = True
                    break
            if has_same_point:
                break
        if has_same_point:
            continue

        if min([
            bbox[1][0] - bbox[0][0], 
            bbox[2][0] - bbox[3][0], 
            bbox[3][1] - bbox[0][1],
            bbox[2][1] - bbox[1][1]]) <= max_offset:
            continue

        new_cells.append(cell)

    return new_cells


def adjust_cells(cells, proposal_cells):
    angles = []
    for cell in cells:
        bbox = cell['bbox']
        angle = (calc_angle(bbox[0], bbox[1], axis=0) + calc_angle(bbox[3], bbox[2], axis=0)) / 2
        angles.append(angle)
    if cells==[]:
        return cells, proposal_cells
    angles = np.asarray(angles)
    mean = angles.mean()
    std = angles.std()
    theta = mean / 180 * pi
    slope = sin(theta)

    angles = angles.tolist()
    for cell in proposal_cells:
        bbox = cell['bbox']
        angle = (calc_angle(bbox[0], bbox[1], axis=0) + calc_angle(bbox[3], bbox[2], axis=0)) / 2
        angles.append(angle)

    for i, angle in enumerate(angles):
        if angle <= mean - 3 * std or mean + 3 * std <= angle and abs(angle - mean) >= 4:
            bbox = cells[i]['bbox'] if i < len(cells) else proposal_cells[i - len(cells)]['bbox']
            up_xoffset = bbox[1][0] - bbox[0][0]
            down_xoffset =  bbox[2][0] - bbox[3][0]
            bbox[1] = [bbox[0][0] + up_xoffset, bbox[0][1] + slope * up_xoffset]
            bbox[2] = [bbox[3][0] + down_xoffset, bbox[3][1] + slope * down_xoffset]

    return cells, proposal_cells


def lcs_filter(cells):
    new_cells = []
    for cell in cells:
        lcs = cell['logical_coordinates']
        if lcs['startrow'] <= lcs['endrow'] and lcs['startcol'] <= lcs['endcol']:
            new_cells.append(cell)
    return new_cells


def overlapping_filter(cells):
    cells.sort(
        key=lambda cell: (
            cell['logical_coordinates']['startrow'], 
            -cell['logical_coordinates']['endrow'],
            cell['logical_coordinates']['startcol'],
            -cell['logical_coordinates']['endcol']
        )
    )

    new_cells = []
    for i, cell in enumerate(cells):
        if i == 0:
            new_cells.append(cell)
        else:
            last_cell = cells[i-1]
            if cell['logical_coordinates'] != last_cell['logical_coordinates']:
                new_cells.append(cell)
    cells = new_cells
    
    sub_cells = len(cells) * [None]
    for i, cell_i in enumerate(cells):
        lcs_i = cell_i['logical_coordinates']
        for j, cell_j in enumerate(cells[i+1:]):
            j = i + 1 + j
            lcs_j = cell_j['logical_coordinates']      

            if lcs_j['startrow'] > lcs_i['endrow']:
                break
            
            if lcs_j['startrow'] >= lcs_i['startrow'] and \
                lcs_j['endrow'] <= lcs_i['endrow'] and \
                lcs_j['startcol'] >= lcs_i['startcol'] and \
                lcs_j['endcol'] <= lcs_i['endcol']:
                    diff_idx_cnt = 0
                    for k, v in lcs_i.items():
                        if lcs_j[k] != v:
                            diff_idx_cnt += 1
                    if diff_idx_cnt == 1:
                        if sub_cells[i] is None:
                            sub_cells[i] = [j]
                        else:
                            sub_cells[i].append(j)     
    
    new_cells = []
    for i, sub_cell in enumerate(sub_cells):
        new_cell = None
            
        if sub_cell is None:
            new_cell = cells[i]

        elif len(sub_cell) == 1:
            pcs_i = cells[i]['bbox']
            lcs_i = cells[i]['logical_coordinates']

            j = sub_cell[0]
            pcs_j = cells[j]['bbox']
            lcs_j = cells[j]['logical_coordinates']

            bbox, sr, er, sc, ec = None, None, None, None, None
            if lcs_i['startrow'] != lcs_j['startrow']:
                bbox = deepcopy([pcs_i[0], pcs_i[1], pcs_j[1], pcs_j[0]])
                sr, er, sc, ec = lcs_i['startrow'], lcs_j['startrow'] - 1, lcs_i['startcol'], lcs_i['endcol']
            elif lcs_i['endrow'] != lcs_j['endrow']:
                bbox = deepcopy([pcs_j[3], pcs_j[2], pcs_i[2], pcs_i[3]])
                sr, er, sc, ec = lcs_j['endrow'] + 1, lcs_i['endrow'], lcs_i['startcol'], lcs_i['endcol']
            elif lcs_i['startcol'] != lcs_j['startcol']:
                bbox = deepcopy([pcs_i[0], pcs_j[0], pcs_j[3], pcs_i[3]])
                sr, er, sc, ec = lcs_i['startrow'], lcs_i['endrow'], lcs_i['startcol'], lcs_j['startcol'] - 1
            elif lcs_i['endcol'] != lcs_j['endcol']:
                bbox = deepcopy([pcs_j[1], pcs_i[1], pcs_i[2], pcs_j[2]])
                sr, er, sc, ec = lcs_i['startrow'], lcs_i['endrow'], lcs_j['endcol'] + 1, lcs_i['endcol']

            new_cell = {
                'bbox': bbox,
                'logical_coordinates': {
                    'startrow': sr,
                    'endrow': er,
                    'startcol': sc, 
                    'endcol': ec
                },
                'texts': cells[i]['texts'],
                'confidence': cells[i]['confidence']
            }

        if new_cell is not None:
            new_cells.append(new_cell)

    return new_cells


def cells2html(cells, transpose=False):
    if len(cells) == 0:
        return ''

    cells = deepcopy(cells)
    if transpose:
        for cell in cells:
            lcs = cell['logical_coordinates']
            lcs['startrow'], lcs['endrow'], lcs['startcol'], lcs['endcol'] = lcs['startcol'], lcs['endcol'], lcs['startrow'], lcs['endrow']

    cells.sort(
        key=lambda cell: (
            cell['logical_coordinates']['startrow'], 
            # -cell['logical_coordinates']['endrow'],
            cell['logical_coordinates']['startcol']
        )
    )

    cur_startrow = -1
    cur_endrow = -1
    table_html_strings = []
    for cell in cells:
        lcs = cell['logical_coordinates']
        texts = cell['texts']

        head = ''
        if lcs['startrow'] > cur_startrow:
            head='</tr><tr>' if cur_startrow != -1 else '<tr>'
            cur_startrow = lcs['startrow']
            cur_endrow = lcs['endrow']
        # elif lcs['startrow'] == cur_startrow and lcs['endrow'] < cur_endrow:
        #     head='</tr><tr>'
        #     cur_endrow = lcs['endrow']

        rowspan = lcs['endrow'] - lcs['startrow'] + 1
        colspan = lcs['endcol'] - lcs['startcol'] + 1
        text = '<br>'.join(['&lt;sep&gt;'.join([col['content'] for col in row]) for row in texts])
        table_html_string = head + f'<td rowspan="{rowspan}" colspan="{colspan}">{text}</td>'
        table_html_strings.append(table_html_string)
    table_html_strings[-1] = table_html_strings[-1] + '</tr>'
            
    html_string = '<table frame="hsides" rules="groups" width="100%%">%s</table>' % ''.join(table_html_strings)
    return html_string.replace('\n', '')


def assign_texts(cells, meta_data):
    table_rect = meta_data['table_rect_ex']
    text_cells = meta_data['text_cells']

    saved_text_cells = []
    proposal_text_cells = []

    for text_cell in text_cells:
        text_bbox = text_cell['bbox']
        center = get_center_point(text_bbox)
        if not (table_rect[0] <= center[0] and center[0] <= table_rect[2] and \
            table_rect[1] <= center[1] and center[1] <= table_rect[3]):
            continue

        cell_idx = None
        for i, cell in enumerate(cells):
            cell_bbox = cell['bbox']
            cell_rect = bbox2rect(cell_bbox)
            if cell_rect[0] <= center[0] and center[0] <= cell_rect[2] and \
                cell_rect[1] <= center[1] and center[1] <= cell_rect[3]:        
                    cell_idx = i
                    break
        
        if cell_idx is None:
            proposal_text_cells.append(text_cell)
        else:
            cells[cell_idx]['texts'].append(text_cell)
            saved_text_cells.append(text_cell)            

    return cells, saved_text_cells, proposal_text_cells


def add_cells(cells, proposal_cells, saved_text_cells, proposal_text_cells):
    cells_rect = [inf, inf, 0, 0]
    for cell in cells:
        bbox = cell['bbox']
        for p in bbox:
            cells_rect[0] = min(cells_rect[0], p[0])
            cells_rect[1] = min(cells_rect[1], p[1])
            cells_rect[2] = max(cells_rect[2], p[0])
            cells_rect[3] = max(cells_rect[3], p[1])

    # proposal_cells filtering
    keep = len(proposal_cells) * [False]

    for i, cell in enumerate(proposal_cells):
        cell_bbox = cell['bbox']
        cell_rect = bbox2rect(cell_bbox)
        # proposal_cell in cells_rect
        if cells_rect[0] - 0 <= cell_rect[0] and cell_rect[2] <= cells_rect[2] + 0 and \
            cells_rect[1] - 0 <= cell_rect[1] and cell_rect[3] <= cells_rect[3] + 0:
            keep[i] = True

    for i, cell in enumerate(proposal_cells):
        if keep[i]:
             for text_cell in saved_text_cells:
                text_bbox = text_cell['bbox']
                center = get_center_point(text_bbox)
                cell_bbox = cell['bbox']
                cell_rect = bbox2rect(cell_bbox)
                # saved_text_cells in proposal_cell
                if cell_rect[0] <= center[0] and center[0] <= cell_rect[2] and \
                    cell_rect[1] <= center[1] and center[1] <= cell_rect[3]:
                    keep[i] = False
                    break

    for i, proposal_cell in enumerate(proposal_cells):
        if keep[i]:
            proposal_cell_bbox = proposal_cell['bbox']
            proposal_cell_poly = Polygon(proposal_cell_bbox).convex_hull
            for cell in cells:
                cell_bbox = cell['bbox']
                cell_poly = Polygon(cell_bbox).convex_hull
                overlap_rate = proposal_cell_poly.intersection(cell_poly).area / proposal_cell_poly.area
                if overlap_rate > 0.25:
                    keep[i] = False
                    break
                
    proposal_cells = [cell for i, cell in enumerate(proposal_cells) if keep[i]]

    # proposal_text_cells filtering
    keep = len(proposal_text_cells) * [False]
    for i, text_cell in enumerate(proposal_text_cells):
        text_bbox = text_cell['bbox']
        center = get_center_point(text_bbox)
        # proposal_text_cell in cells_rect
        if cells_rect[0] <= center[0] and center[0] <= cells_rect[2] and \
            cells_rect[1] <= center[1] and center[1] <= cells_rect[3]:
            keep[i] = True
    proposal_text_cells = [cell for i, cell in enumerate(proposal_text_cells) if keep[i]]

    # matching
    keep_cells = len(proposal_cells) * [False]
    keep_texts = len(proposal_text_cells) * [False]
    for i, text_cell in enumerate(proposal_text_cells):
        text_bbox = text_cell['bbox']
        text_poly = Polygon(text_bbox).convex_hull
        center = get_center_point(text_bbox)

        max_iou = 0.1
        cell_idx = None
        for j, cell in enumerate(proposal_cells):
            cell_bbox = cell['bbox']
            cell_rect = bbox2rect(cell_bbox)
            # proposal_text_cell in proposal_cell
            if cell_rect[0] <= center[0] and center[0] <= cell_rect[2] and \
                cell_rect[1] <= center[1] and center[1] <= cell_rect[3]:
                cell_poly = Polygon(cell_bbox).convex_hull
                iter_area = text_poly.intersection(cell_poly).area
                iou = iter_area / text_poly.union(cell_poly).area
                if iou > max_iou:
                    max_iou = iou
                    cell_idx = j
                    break

        if cell_idx is not None:
            proposal_cells[cell_idx]['texts'].append(text_cell)
            keep_cells[cell_idx] = True
            keep_texts[i] = True

    cells += [cell for i, cell in enumerate(proposal_cells) if keep_cells[i]]
    proposal_text_cells = [cell for i, cell in enumerate(proposal_text_cells) if not keep_texts[i]]

    return cells, proposal_text_cells


def add_cells_by_lcs(cells, table_bbox):
    first_row, last_row, first_col, last_col = inf, -1, inf, -1
    for cell in cells:
        lcs = cell['logical_coordinates']
        first_row = min(first_row, lcs['startrow'])
        last_row = max(last_row, lcs['endrow'])
        first_col = min(first_col, lcs['startcol'])
        last_col = max(last_col, lcs['endcol'])

    master = [(last_col+1) * [None] for _ in range(last_row+1)]
    for i, cell in enumerate(cells):
        lcs = cell['logical_coordinates']
        for r in range(lcs['startrow'], lcs['endrow']+1):
            for c in range(lcs['startcol'], lcs['endcol']+1):
                master[r][c] = i

    num_tails = sum([1 for r in range(first_row, last_row+1) if master[r][last_col] is not None])
    num_rows = last_row - first_row + 1
    add_tail = num_tails != num_rows and num_tails >= 0.75*num_rows
    
    fc_bboxes, lc_bboxes = [], []
    fc_x_mean, fc_y_mean, lc_x_mean, lc_y_mean = [], [], [], []
    for r in range(first_row, last_row+1):
        if master[r][first_col] is not None:
            bbox = cells[master[r][first_col]]['bbox']
            fc_bboxes.append(bbox)
            fc_x_mean.extend([bbox[0][0], bbox[3][0]])
            fc_y_mean.extend([bbox[0][1], bbox[3][1]])
        if master[r][last_col] is not None:
            bbox = cells[master[r][last_col]]['bbox']
            lc_bboxes.append(bbox)
            lc_x_mean.extend([bbox[1][0], bbox[2][0]])
            lc_y_mean.extend([bbox[1][1], bbox[2][1]])
    fc_x_mean = np.mean(fc_x_mean)
    fc_y_mean = np.mean(fc_y_mean)
    lc_x_mean = np.mean(lc_x_mean)
    lc_y_mean = np.mean(lc_y_mean)

    fc_slope = inf if table_bbox[0][0] == table_bbox[3][0] else \
        (table_bbox[3][1] - table_bbox[0][1]) / (table_bbox[3][0] - table_bbox[0][0])
    fc_intercept = fc_x_mean if fc_slope == inf else \
        fc_y_mean - fc_slope * fc_x_mean
    lc_slope = inf if table_bbox[1][0] == table_bbox[2][0] else \
        (table_bbox[2][1] - table_bbox[1][1]) / (table_bbox[2][0] - table_bbox[1][0])
    lc_intercept = lc_x_mean if lc_slope == inf else \
        lc_y_mean - lc_slope * lc_x_mean

    fc_fr_bbox = fc_bboxes[0]
    fc_lr_bbox = fc_bboxes[-1]
    fc_line = [[0, fc_intercept], [1, (0 if fc_slope == inf else fc_slope) + fc_intercept]]
    fc_line_p1 = get_line_cross_point(fc_line, [fc_fr_bbox[0], fc_fr_bbox[1]])
    fc_line_p2 = get_line_cross_point(fc_line, [fc_lr_bbox[3], fc_lr_bbox[2]])

    lc_fr_bbox = lc_bboxes[0]
    lc_lr_bbox = lc_bboxes[-1]
    lc_line = [[0, lc_intercept], [1, (0 if lc_slope == inf else lc_slope) + lc_intercept]]
    lc_line_p1 = get_line_cross_point(lc_line, [lc_fr_bbox[0], lc_fr_bbox[1]])
    lc_line_p2 = get_line_cross_point(lc_line, [lc_lr_bbox[3], lc_lr_bbox[2]])

    new_cells = []
    for cell in cells:
        bbox = cell['bbox']
        lcs = cell['logical_coordinates']

        num_empty_cols = 0
        left_cell = None
        for c in range(lcs['startcol']-1, first_col-1, -1):
            if master[lcs['startrow']][c] is None:
                num_empty_cols += 1
            else:
                left_cell = cells[master[lcs['startrow']][c]]
                break
        if num_empty_cols:
            if left_cell is None:
                left_cell = {
                    'bbox': [[-1,-1], fc_line_p1, fc_line_p2, [-1,-1]],
                    'logical_coordinates': {
                        'startrow': lcs['startrow'],
                        'endrow': lcs['endrow'],
                        'startcol': first_col - 1,
                        'endcol': first_col - 1
                    }
                }

            new_tl, new_tr, new_bl, new_br = None, bbox[0], None, bbox[3]
            new_sr, new_er, new_sc, new_ec = lcs['startrow'], lcs['endrow'], left_cell['logical_coordinates']['endcol']+1, lcs['startcol']-1

            left_cell_right_edge = [left_cell['bbox'][1], left_cell['bbox'][2]]
            cell_up_edge = [bbox[0], bbox[1]]
            cell_down_edge = [bbox[3], bbox[2]]

            new_tl = get_line_cross_point(left_cell_right_edge, cell_up_edge)
            new_bl = get_line_cross_point(left_cell_right_edge, cell_down_edge)

            new_cell = {
                'bbox': [new_tl, new_tr, new_br, new_bl],
                'logical_coordinates': {
                    'startrow': new_sr,
                    'endrow': new_er,
                    'startcol': new_sc,
                    'endcol': new_ec
                },
                'texts': [],
                'confidence': None
            }
            new_cells.append(new_cell)

        if add_tail:
            num_empty_cols = 0
            right_cell = None
            for c in range(lcs['endcol']+1, last_col+1):
                if master[lcs['startrow']][c] is None:
                    num_empty_cols += 1
                else:
                    right_cell = cells[master[lcs['startrow']][c]]
                    break
            if num_empty_cols and right_cell is None:
                right_cell = {
                    'bbox': [lc_line_p1, [-1,-1], [-1,-1], lc_line_p2],
                    'logical_coordinates': {
                        'startrow': lcs['startrow'],
                        'endrow': lcs['endrow'],
                        'startcol': last_col + 1,
                        'endcol': last_col + 1
                    }
                }

                new_tl, new_tr, new_bl, new_br = bbox[1], None, bbox[2], None
                new_sr, new_er, new_sc, new_ec = lcs['startrow'], lcs['endrow'], lcs['endcol']+1, right_cell['logical_coordinates']['startcol']-1, 

                right_cell_left_edge = [right_cell['bbox'][0], right_cell['bbox'][3]]
                cell_up_edge = [bbox[0], bbox[1]]
                cell_down_edge = [bbox[3], bbox[2]]

                new_tr = get_line_cross_point(right_cell_left_edge, cell_up_edge)
                new_br = get_line_cross_point(right_cell_left_edge, cell_down_edge)

                new_cell = {
                    'bbox': [new_tl, new_tr, new_br, new_bl],
                    'logical_coordinates': {
                        'startrow': new_sr,
                        'endrow': new_er,
                        'startcol': new_sc,
                        'endcol': new_ec
                    },
                    'texts': [],
                    'confidence': None
                }
                new_cells.append(new_cell)

    return cells + new_cells


def sort_texts(cells):
    for cell in cells:
        texts = cell['texts']
        if len(texts) > 0:            
            mean_char_height = []
            for text in texts:
                bbox = text['bbox']
                height = (distance(bbox[0], bbox[3]) + distance(bbox[1], bbox[2])) / 2
                mean_char_height.append(height)
            mean_char_height = sum(mean_char_height) / len(mean_char_height)

            mean_y = -100
            new_texts = []
            for text in texts:
                x, y = get_center_point(text['bbox'])
                if mean_y - mean_char_height / 2 <= y <= mean_y + mean_char_height / 2:
                    num = len(new_texts[-1])
                    mean_y = (num * mean_y + y) / (num + 1)
                    new_texts[-1].append(text)
                else:
                    new_texts.append([text])
                    mean_y = y
            cell['texts'] = new_texts

    return cells


def outlier_filter(cells, meta_data):
    last_row, last_col = -1, -1
    for cell in cells:
        lcs = cell['logical_coordinates']
        last_row = max(last_row, lcs['endrow'])
        last_col = max(last_col, lcs['endcol'])

    max_er, min_sr, max_ec, min_sc = -1, inf, -1, inf
    for cell in cells:
        lcs = cell['logical_coordinates']
        if lcs['startrow'] == 0:
            max_er = max(max_er, lcs['endrow'])
        if lcs['endrow'] == last_row:
            min_sr = min(min_sr, lcs['startrow'])
        if lcs['startcol'] == 0:
            max_ec = max(max_ec, lcs['endcol'])
        if lcs['endcol'] == last_col:
            min_sc = min(min_sc, lcs['startcol'])

    first_row_cell_indices = []
    last_row_cell_indices = []
    first_col_cell_indices = []
    last_col_cell_indices = []
    fr_flag, lr_flag, fc_flag, lc_flag = False, False, False, False

    area = [(last_col+1) * [0] for _ in range(last_row+1)]
    for i, cell in enumerate(cells):
        lcs = cell['logical_coordinates']
        for r in range(lcs['startrow'], lcs['endrow']+1):
            for c in range(lcs['startcol'], lcs['endcol']+1):
                area[r][c] = 1
    if last_row > 0 and sum([area[last_row][i] for i in range(last_col+1)]) == sum([area[last_row-1][i] for i in range(last_col+1)]):
        lr_flag = True

    for i, cell in enumerate(cells):
        lcs = cell['logical_coordinates']
        has_text = len(cell['texts']) > 0

        if lcs['endrow'] <= max_er:
            first_row_cell_indices.append(i)
            if has_text:
                fr_flag = True

        if lcs['startrow'] >= min_sr:
            last_row_cell_indices.append(i)
            if has_text:
                lr_flag = True

        if lcs['endcol'] <= max_ec:
            first_col_cell_indices.append(i)
            if has_text:
                fc_flag = True

        if lcs['startcol'] >= min_sc:
            last_col_cell_indices.append(i)
            if has_text:
                lc_flag = True
    
    outlier_indices = []
    for cell_indices, flag in zip([first_row_cell_indices, last_row_cell_indices, first_col_cell_indices, last_col_cell_indices], [fr_flag, lr_flag, fc_flag, lc_flag]):
        if not flag:
            outlier_indices += cell_indices
    
    new_cells = [cell for i, cell in enumerate(cells) if i not in outlier_indices]
    return new_cells


def post_process(ret, meta_data: Dict):
    cells = []
    proposal_cells = []

    for dets in ret:
        score = dets[4]
        keypoints = np.array(dets[5:13], dtype=np.float32).reshape(-1,2).tolist()

        cell = {
            'bbox': keypoints,
            'texts': [],
            'confidence': score
        }
        if score >= 0.3:
            cells.append(cell)
        elif score >= 0.1:
            proposal_cells.append(cell)
        else:
            break
    num_cells = len(cells)
    num_text_cells = len(meta_data["text_cells"])
    if num_cells == 0 or num_cells > 300:
        return [], 0
    #if len(meta_data["text_cells"])
    table_rect = meta_data['table_rect_ex']
    x_offset = table_rect[0]
    y_offset = table_rect[1]
    for cell in cells + proposal_cells:
        bbox = cell['bbox']
        for p in bbox:
            p[0] += x_offset
            p[1] += y_offset

    cells = pcs_filter(cells, table_rect)
    proposal_cells = pcs_filter(proposal_cells, table_rect)

    cells, proposal_cel = adjust_cells(cells, proposal_cells)

    cells, saved_text_cells, proposal_text_cells = assign_texts(cells, meta_data)
    cells, proposal_text_cells = add_cells(cells, proposal_cells, saved_text_cells, proposal_text_cells)

    cells = splice_cells(cells)

    cells = pcs_filter(cells)
    cells = calc_logical_coordinates(cells)

    cells = lcs_filter(cells)
    cells = overlapping_filter(cells)

    cells = pcs_filter(cells)
    cells = outlier_filter(cells, meta_data)

    confidence = sum([cell['confidence'] for cell in cells]) / len(cells) if len(cells) else 0

    if len(cells):
        # cells = add_cells_by_lcs(cells, meta_data['table_bbox'])
        # cells = pcs_filter(cells)
        meta_data['text_cells'] = proposal_text_cells
        cells, _, _ = assign_texts(cells, meta_data)

        cells = sort_texts(cells)

    return cells, confidence