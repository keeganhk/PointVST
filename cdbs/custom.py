from .pkgs import *
from .general import *



################################################################################
def geog2cart(d, lat, lon):
    assert lat>=(-90) and lat<=(90)
    assert lon>=0 and lon<=360
    x, y, z = spherical_to_cartesian(d, np.deg2rad(lat), np.deg2rad(lon))
    return x.value, y.value, z.value


def cart2geog(x, y, z):
    d, lat, lon = cartesian_to_spherical(x, y, z)
    d = d.value
    lat = np.rad2deg(lat.value)
    lon = np.rad2deg(lon.value)
    return d, lat, lon


def generate_views_pool_r():
    d = 5.0
    lat_choices = [+70, +30, -30, -70]
    lon_choices = list(np.arange(0, 360, 45))
    num_views = len(lat_choices) * len(lon_choices)
    views_pool = []
    for lat_id in range(len(lat_choices)):
        for lon_id in range(len(lon_choices)):
            lat = lat_choices[lat_id]
            lon = lon_choices[lon_id]
            x, y, z = geog2cart(d, lat, lon)
            views_pool.append(np.asarray([x, y, z]).reshape(1, 3))      
    views_pool = np.concatenate(views_pool, axis=0)
    return views_pool


def generate_views_pool_r_with_onehot():
    d = 5.0
    lat_choices = [+70, +30, -30, -70]
    lon_choices = list(np.arange(0, 360, 45))
    num_lat = len(lat_choices)
    num_lon = len(lon_choices)
    num_views = num_lat * num_lon
    dim_one_hot = num_lat + num_lon
    views_pool = []
    lat_one_hot = np.zeros((num_views, num_lat)).astype(np.float32)
    lon_one_hot = np.zeros((num_views, num_lon)).astype(np.float32)
    j = 0
    for lat_id in range(num_lat):
        for lon_id in range(num_lon):
            lat = lat_choices[lat_id]
            lon = lon_choices[lon_id]
            x, y, z = geog2cart(d, lat, lon)
            views_pool.append(np.asarray([x, y, z]).reshape(1, 3))
            lat_one_hot[j, lat_id] = 1.0
            lon_one_hot[j, lon_id] = 1.0
            j += 1   
    views_pool = np.concatenate(views_pool, axis=0)
    views_pool_one_hot = np.concatenate((lat_one_hot, lon_one_hot), axis=-1)
    assert views_pool.shape[0] == views_pool_one_hot.shape[0]
    return views_pool, views_pool_one_hot


def generate_views_pool_v():
    d = 2.0
    lift = 3
    lat_choices = [+70+lift, +30+lift, -30-lift, -70-lift]
    lon_choices = list(np.arange(0, 360, 45))
    num_views = len(lat_choices) * len(lon_choices)
    views_pool = []
    for lat_id in range(len(lat_choices)):
        for lon_id in range(len(lon_choices)):
            lat = lat_choices[lat_id]
            lon = lon_choices[lon_id]
            x, y, z = geog2cart(d, lat, lon)
            views_pool.append(np.asarray([x, y, z]).reshape(1, 3))      
    views_pool = np.concatenate(views_pool, axis=0)
    return views_pool


def tm2dm(v, f, x, y, z, ort, h, w, empty_value=-1):
    assert empty_value < 0
    params = [{}]
    params[0]['cam_pos'] = [x, y, z]
    params[0]['cam_lookat'] = [0, 0, 0]
    params[0]['cam_up'] = ort
    params[0]['x_fov'] = 0.60
    params[0]['near'] = 0.1
    params[0]['far'] = 10
    params[0]['height'] = h
    params[0]['width'] = w
    params[0]['is_depth'] = True
    dm = mesh2depth(v, f, params, empty_pixel_value=empty_value)[0]
    return dm


def empty_cropping(img):
    h, w = img.shape
    col_flag = img.max(axis=0)
    row_flag = img.max(axis=1)
    h_min, h_max, w_min, w_max = 0, h-1, 0, w-1
    for k in np.linspace(0, h-1, h, dtype=np.uint16):
        if row_flag[k] >= 0:
            h_min = k
            break
    for k in np.linspace(h-1, 0, h, dtype=np.uint16):
        if row_flag[k] >= 0:
            h_max = k
            break
    for k in np.linspace(0, w-1, w, dtype=np.uint16):
        if col_flag[k] >= 0:
            w_min = k
            break
    for k in np.linspace(w-1, 0, w, dtype=np.uint16):
        if col_flag[k] >= 0:
            w_max = k
            break
    img_c = img[h_min:h_max+1, w_min:w_max+1]
    return img_c


def square_padding(img, pad_value=-1):
    assert pad_value < 0
    h, w = img.shape
    if h == w:
        img_p = img
    else:
        if abs(h - w) == 1:
            if h < w:
                pad = np.ones((1, w)) * pad_value
                img_p = np.concatenate((img, pad), axis=0)
            else:
                pad = np.ones((h, 1)) * pad_value
                img_p = np.concatenate((img, pad), axis=1)
        else:
            pad_len = abs(h - w) // 2
            if h < w:
                pad = np.ones((pad_len, w)) * pad_value
                img_p = np.concatenate((pad, img, pad), axis=0)
            else:
                pad = np.ones((h, pad_len)) * pad_value
                img_p = np.concatenate((pad, img, pad), axis=1)
    if img_p.shape[0] != img_p.shape[1]:
        if img_p.shape[0] < img_p.shape[1]:
            s = img_p.shape[0]
        else:
            s = img_p.shape[1]
        img_p = img_p[0:s, 0:s]
    t = img_p.shape[0]
    z = int(np.sqrt(h*w) * 0.15)
    container = np.zeros((t+z, t+z)).astype(np.float32)
    container[z//2:(t+z//2), z//2:(t+z//2)] = img_p
    container[0:z//2, :] = pad_value
    container[t+z//2:t+z, :] = pad_value
    container[:, 0:z//2] = pad_value
    container[:, t+z//2:t+z] = pad_value
    return container


def resize_dm(img, target_h, target_w):
    img_resized = cv2.resize(img, dsize=(target_h, target_w), interpolation=cv2.INTER_NEAREST)
    return img_resized


def post_process_dm(dm_raw, h, w):
    dm_ppr = resize_dm(square_padding(empty_cropping(dm_raw)), h, w)
    if dm_ppr.max() < 0:
        dm_ppr = np.zeros((h, w))
    else:
        emp_idx = np.where(dm_ppr<0)
        fil_idx = np.where(dm_ppr>=0)
        num_emp = emp_idx[0].shape[0]
        num_fil = fil_idx[0].shape[0]
        d_min = dm_ppr[fil_idx].min()
        d_max = dm_ppr[fil_idx].max()
        dm_ppr = (dm_ppr - d_min) / (d_max - d_min + 1e-8)
        margin = 0.05
        dm_ppr = dm_ppr * (1-margin*2) + margin
        dm_ppr = 1 - dm_ppr
        dm_ppr[emp_idx] = 0
    return dm_ppr.astype(np.float32)


def generate_depth_map(v, f, x, y, z, h, w):
    h_raw, w_raw = 512, 512
    ort = np.asarray([0.0, 0.0, 1.0])
    dm_raw = tm2dm(v, f, x, y, z, ort, h_raw, w_raw)
    dm_ppr = post_process_dm(dm_raw, h, w)
    return dm_ppr


def prepare_image_supervisions(dm_gt_init):
    bs, num_lon, init_dm_h, init_dm_w = dm_gt_init.size()
    device = dm_gt_init.device
    dm_gt_init = dm_gt_init.view(bs*num_lon, init_dm_h, init_dm_w)
    with torch.no_grad():
        dm_gt = kornia.morphology.dilation(dm_gt_init.unsqueeze(1), kernel=torch.ones(3, 3).to(device).float()).squeeze(1)
        dm_gt_closed = kornia.morphology.closing(dm_gt.unsqueeze(1), kernel=torch.ones(5, 5).to(device).float()).squeeze(1)
        bm_gt = dm_gt_closed.clone()
        bm_gt[bm_gt>0] = 1.0
        be_gt = kornia.filters.canny(dm_gt_closed.unsqueeze(1))[1].squeeze(1)
        be_gt = kornia.morphology.dilation(be_gt.unsqueeze(1), kernel=torch.ones(3, 3).to(device).float()).squeeze(1)
        dm_gt = F.interpolate(dm_gt.unsqueeze(1), size=(96, 96), mode='nearest').squeeze(1)
        bm_gt = F.interpolate(bm_gt.unsqueeze(1), size=(96, 96), mode='nearest').squeeze(1)
        be_gt = F.interpolate(be_gt.unsqueeze(1), size=(96, 96), mode='nearest').squeeze(1)
        dm_gt = dm_gt.view(bs, num_lon, 96, 96)
        bm_gt = bm_gt.view(bs, num_lon, 96, 96)
        be_gt = be_gt.view(bs, num_lon, 96, 96)
    return dm_gt, bm_gt, be_gt


################################################################################
def extract_visible_points(points, x_v, y_v, z_v):
    num_complete = points.shape[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    cam_pos = [x_v, y_v, z_v]
    r = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())) * (0.05 * num_complete)
    _, pt_map = pcd.hidden_point_removal(cam_pos, r)
    pcd_vis = pcd.select_by_index(pt_map)
    points_vis = np.asarray(pcd_vis.points).astype(np.float32)
    indices_vis = np.asarray(pt_map)
    return indices_vis, points_vis


################################################################################
def unitize_normals(nms_raw):
    assert nms_raw.ndim==2 and nms_raw.shape[1]==3
    scaling = ((nms_raw**2).sum(axis=1, keepdims=True) + 1e-8) ** 0.5
    nms_u = nms_raw / scaling
    return nms_u


def normals_distances(nms_1, nms_2):
    assert nms_1.size(0) == nms_2.size(0)
    assert nms_1.size(1) == nms_2.size(1)
    assert nms_1.size(2)==3 and nms_2.size(2)==3
    bs, num_pts = nms_1.size(0), nms_1.size(1)
    cos_sim = F.cosine_similarity(nms_1, nms_2, dim=-1)
    dists = (1.0 - torch.abs(cos_sim))
    return dists


################################################################################
def smooth_cross_entropy(pred, label, eps=0.2):
    #  Cross Entropy Loss with Label Smoothing
    # label = label.contiguous().view(-1)
    # pred: [batch_size, num_classes]
    # label: [batch_size]
    num_classes = pred.size(1)
    one_hot = torch.zeros_like(pred).scatter(1, label.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (num_classes - 1)
    log_prb = F.log_softmax(pred, dim=1)
    sce_loss = -(one_hot * log_prb).sum(dim=1).mean()
    return sce_loss


def bw_ohem_bce(pr, gt, keep_ratio):
    # batch-wise online hard example mining for binary cross entropy loss
    # pr: [batch_size, xxx], predicted results
    # gt: [batch_size, xxx], ground truths
    assert pr.size(0)==gt.size(0) and pr.device==gt.device
    batch_size, device = pr.size(0), pr.device
    assert keep_ratio>0 and keep_ratio<=1
    num_keep = int(np.around(batch_size * keep_ratio))
    assert num_keep>0 and num_keep<=batch_size
    bw_losses = F.binary_cross_entropy(pr, gt, reduction='none').view(batch_size, -1).mean(dim=-1) # [batch_size]
    bw_losses_sorted, idx_sorted = torch.sort(bw_losses, descending=True)
    loss = bw_losses_sorted[:num_keep].mean()
    return loss


def bw_ohem_l1(pr, gt, keep_ratio):
    # batch-wise online hard example mining for L1 loss
    # pr: [batch_size, xxx], predicted results
    # gt: [batch_size, xxx], ground truths
    assert pr.size(0)==gt.size(0) and pr.device==gt.device
    batch_size, device = pr.size(0), pr.device
    assert keep_ratio>0 and keep_ratio<=1
    num_keep = int(np.around(batch_size * keep_ratio))
    assert num_keep>0 and num_keep<=batch_size
    bw_losses = F.l1_loss(pr, gt, reduction='none').view(batch_size, -1).mean(dim=-1) # [batch_size]
    bw_losses_sorted, idx_sorted = torch.sort(bw_losses, descending=True)
    loss = bw_losses_sorted[:num_keep].mean()
    return loss



