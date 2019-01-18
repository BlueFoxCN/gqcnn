import numpy as np
from plyfile import PlyData, PlyElement
import random
import scipy.linalg
import pdb
import cv2
import os
# import matplotlib.pyplot as plt

def load_cloud_point(ply_path):
    """
    load cloud point from a ply file.

    Args:
        ply_path (str) : path str to ply file.

    Returns:
        peach_cloud (np.ndarray, (point_num, 3)): array of [x, y, z]
    """
    plydata = PlyData.read(ply_path)
    point_num = plydata['vertex']['x'].shape[0]
    peach_cloud = np.empty([point_num, 3], dtype=np.float32)
    peach_cloud[:, 0] = plydata['vertex']['x']
    peach_cloud[:, 1] = plydata['vertex']['y']
    peach_cloud[:, 2] = plydata['vertex']['z']
    return peach_cloud

def compute_cloud_dict(point_cloud, vec):
    """
    compute coordinates of points.

    Args:
        point_cloud (np.ndarray, (point_num, 3)): array of [x, y, z]
        vec (list, 4): [cx, cy, fx, fy]

    Returns:
        uv_dict (dict, (u, v): np.array([x, y, z])) : a dict map coordinates to points.
    """
    cx, cy, fx, fy = vec
    point_num = point_cloud.shape[0]

    u = (point_cloud[:, 0] * fx) / point_cloud[:, 2] + cx
    v = (point_cloud[:, 1] * fy) / point_cloud[:, 2] + cy
    z = point_cloud[:,2]

    u_a = np.around(u).astype(np.int32)
    v_a = np.around(v).astype(np.int32)
    depth_img = np.transpose(np.array([u_a, v_a, z]))
    # import pdb
    # pdb.set_trace()

    uv_dict = {}
    uv_around = {}

    for i in range(point_num):
        uv_dict[(u[i], v[i])] = point_cloud[i]
        if not (u_a[i], v_a[i]) in uv_around:
            uv_around[(u_a[i], v_a[i])] = point_cloud[i]

    return uv_dict, uv_around, depth_img

def smooth_uv_dict(uv_dict):
    """
    laplacian smoothing to a uv_dict.
    average of 8 adj points + 1 point itself.

    Args:
        uv_dict (dict, (u, v): np.array([x, y, z])) : dict of cloud points.

    Returns:
        uv_dict_smooth (dict, (u, v): np.array([x, y, z])) : smooth dict.
    """
    uv_dict_smooth = {}
    for (ku, kv) in uv_dict:
        smooth_sum = uv_dict[(ku, kv)].copy()
        smooth_count = 1
        if (ku - 1, kv - 1) in uv_dict:
            smooth_sum += uv_dict[(ku - 1, kv - 1)]
            smooth_count += 1
        if (ku, kv - 1) in uv_dict:
            smooth_sum += uv_dict[(ku, kv - 1)]
            smooth_count += 1
        if (ku + 1, kv - 1) in uv_dict:
            smooth_sum += uv_dict[(ku + 1, kv - 1)]
            smooth_count += 1
        if (ku - 1, kv) in uv_dict:
            smooth_sum += uv_dict[(ku - 1, kv)]
            smooth_count += 1
        if (ku + 1, kv) in uv_dict:
            smooth_sum += uv_dict[(ku + 1, kv)]
            smooth_count += 1
        if (ku - 1, kv + 1) in uv_dict:
            smooth_sum += uv_dict[(ku - 1, kv + 1)]
            smooth_count += 1
        if (ku, kv + 1) in uv_dict:
            smooth_sum += uv_dict[(ku, kv + 1)]
            smooth_count += 1
        if (ku + 1, kv + 1) in uv_dict:
            smooth_sum += uv_dict[(ku + 1, kv + 1)]
            smooth_count += 1
        uv_dict_smooth[(ku, kv)] = smooth_sum / smooth_count
    return uv_dict_smooth

def compute_normal(use_dict, uv_around, distance=1):
    """
    compute center.

    Args:
        use_dict (dict, (u, v): np.array([x, y, z])) : dict of cloud points.
        sample_num (int) number of normals to fit.
        distance (int): distance to use while compute normal.
    Returns:
        r (tuple, 1) : r[0] is center point [x, y, z]
    """
    # sampled_uv = random.sample(use_dict.keys(), sample_num)
    depth = use_dict.keys()
    num_point = len(depth)
    able_compute = []
    normal = []

    # compute normal and add line to A_m, b_v for each point.
    for (ind, key) in enumerate(depth):
        ku, kv = key
        au = np.around(ku).astype(np.int32)
        av = np.around(kv).astype(np.int32)
        
        # if a point can compute normal
        if  (au - distance, av) in uv_around\
        and (au + distance, av) in uv_around\
        and (au, av - distance) in uv_around\
        and (au, av + distance) in uv_around:

            able_compute.append(1)
            vec_x = uv_around[(au + distance, av)] - uv_around[(au - distance, av)]
            vec_y = uv_around[(au, av + distance)] - uv_around[(au, av - distance)]

            # compute normal
            ni = np.cross(vec_x, vec_y)
            normal.append(ni)

        else:
            able_compute.append(0)
            normal.append('none')
        n_able = sum(able_compute)
    # print("%d/%d points can compute normal" % (n_able, num_point))
    return normal, able_compute, n_able

def get_distance(cloud, point):
    distance = np.linalg.norm((cloud - point), ord = 2, axis = 1)
    return distance

def get_adjacent_point(cloud, threshold):
    adj_point = []
    n_adj_point = []
    num_p = np.shape(cloud)[0]
    for i in range(num_p):
        distance = get_distance(cloud, cloud[i])
        adj = np.where(distance <= threshold)[0]
        adj_point.append(adj)
        n_adj_point.append(np.shape(adj)[0])
    return adj_point, n_adj_point

def get_angle_residual(ad_point, normal):
    unit_normal = []
    for v in normal:
        if len(v) == 4:
            unit_normal.append('none')
        else:
            unit_normal.append(v / np.linalg.norm(v, 2))
    
    num_point = len(normal)
    # print('Total: %d points' % num_point)
    ad_angle = []
    mean = np.zeros([num_point], dtype = float)
    n_ad = np.zeros([num_point], dtype = int)

    idx = 0
    for cluster in ad_point:
        ad_n = []
        v_sum = [0, 0, 0]
        num_ad = 0
        
        for p in cluster:
            if len(unit_normal[p]) == 3:
                ad_n.append(unit_normal[p])
                v_sum += unit_normal[p]
                num_ad += 1
        v_aver = np.array(v_sum / np.linalg.norm(np.array(v_sum), 2))
        ad_n = np.squeeze(np.array(ad_n))
        
        if np.shape(ad_n)[0] == 0:
            angle = 3.14
        else:
            cos_angle = np.abs(np.dot(np.squeeze(ad_n), v_aver))
            angle = np.arccos(cos_angle)
        ad_angle.append(angle)
        mean[idx] = np.mean(angle)
        n_ad[idx] = num_ad
        idx += 1

    return ad_angle, mean, n_ad
    
if __name__ == '__main__':
    base_dir = 'point_candidate/'
    img_name = '0_seg'
    cloud_dir = os.path.join(base_dir, 'cloud/')
    depth_dir = os.path.join(base_dir, 'depth/')
    point_dir = os.path.join(base_dir, 'point/')
    visua_dir = os.path.join(base_dir, 'visualization/')

    cloud_path = os.path.join(cloud_dir, img_name + '.ply')
    depth_path = os.path.join(depth_dir, img_name + '.jpg')
    visua_path = os.path.join(visua_dir, img_name + '_v0.jpg')

    intrinsic  = [323.388000, 252.487000, 578.500977, 578.500977] # [cx, cy, fx, fy]
    radius = 3
    min_n_ad_point = 10
    max_mean_angle = 0.3

    # load cloud point
    print('#step1: load cloud point', end = '\t')
    cloud = load_cloud_point(cloud_path)
    cloud_z = np.array(sorted(cloud[:,2]))
    cloud_diff = cloud_z[1:] - cloud_z[:-1]
    print('total points: %d' % np.shape(cloud)[0])
    pdb.set_trace()

    # transport cloud to depth
    print('#step2: transform to depth image and smooth')
    uv_dict, uv_around, depth_img = compute_cloud_dict(cloud, intrinsic)
    uv_dict_smooth = smooth_uv_dict(uv_dict)
    depth_unit = np.around(depth_img[:,2] / (np.max(depth_img[:,2]) / 255)).astype(int)
    img = np.zeros((480, 640, 1), np.uint8)
    for i in range(np.shape(cloud)[0]):
        cv2.circle(img, (int(depth_img[i][0]), int(depth_img[i][1])), 1, int(depth_unit[i]), 4)
    cv2.imwrite(depth_path, img)

    # find adjacent points
    print('#step3: find the adjacent points for each point')
    d_1 = get_distance(cloud, cloud[0])
    ad, n_ad = get_adjacent_point(cloud, radius)
    
    # compute normal vector
    print('#step4: compute the normal vector', end = '\t')
    normal, able, n_able = compute_normal(uv_dict_smooth, uv_around, 1)
    print('able compute: %d' % n_able)

    # compute mean angle residual for each cluster
    print('#step5: compute mean angle residual')
    angle_residual, angle_mean, n_ad_normal = get_angle_residual(ad, normal)

    # get candidate point
    print('step6: get point candidate')
    condition_1 = np.where(n_ad_normal >= min_n_ad_point)
    condition_2 = np.where(angle_mean <= max_mean_angle)
    point_idx = np.intersect1d(condition_1, condition_2)
    point = cloud[point_idx]
    
    # visualization
    img_color = cv2.imread(depth_path, cv2.IMREAD_COLOR)
    for idx in point_idx:
        cv2.circle(img_color, (int(depth_img[idx][0]), int(depth_img[idx][1])), 1, (255, 0, 0), 0)
    cv2.imwrite(visua_path, img_color)
    # import pdb
    # pdb.set_trace()
