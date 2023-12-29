from ..pkgs import *
from ..general import *
from ..custom import *



def shapenetcore_sub_list_sampling(shapenetcore_dataset_folder, num_perc):
    np.random.seed()
    class_list = []
    num_models_per_class = {}
    for line_info in parse_list_file(os.path.join(shapenetcore_dataset_folder, 'class_num.txt')):
        line_info_split = line_info.split(' ')
        class_list.append(line_info_split[0])
        num_models_per_class[line_info_split[0]] = int(line_info_split[1])
    num_classes = len(class_list)
    num_perc_min = np.asarray(list(num_models_per_class.values())).min()
    num_perc_max = np.asarray(list(num_models_per_class.values())).max()
    assert num_perc <= num_perc_min
    sampled_sub_list = []
    for class_name in class_list:
        for sel_idx in sorted(np.random.choice(num_models_per_class[class_name], num_perc, replace=False)):
            model_name = class_name + '_' + align_number(sel_idx+1, 4)
            sampled_sub_list.append(model_name)
    return sorted(sampled_sub_list)


class ShapeNetCore_PretrainLoader(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, epoch_model_list, num_pts):
        self.dataset_folder = dataset_folder
        self.epoch_model_list = epoch_model_list
        self.num_pts = num_pts
    def __getitem__(self, model_index):
        np.random.seed()
        model_name = self.epoch_model_list[model_index]
        class_name = model_name[:-5]
        tm_path = os.path.join(self.dataset_folder, 'mesh', class_name, model_name + '.obj')
        pc_path = os.path.join(self.dataset_folder, 'point-cloud', str(self.num_pts), class_name, model_name + '.npy')
        tmv, tmf = load_tm(tm_path, False)
        pts = np.load(pc_path).astype(np.float32)[:, 0:3]
        min_range, max_range = 2/3, 3/2
        aniso_scaling_ratios = (np.random.random(3) * (max_range - min_range) + min_range).astype('float32')
        tmv *= aniso_scaling_ratios
        pts *= aniso_scaling_ratios
        rot_angle = np.random.choice(np.deg2rad(np.arange(0, 360, 15)).astype(np.float32))
        tmv = axis_rotation(tmv, rot_angle, 'z')
        pts = axis_rotation(pts, rot_angle, 'z')
        pts = random_translation(pts, 0.20)
        num_lat = 4
        num_lon = 8
        views_pool_r, views_pool_r_ohv = generate_views_pool_r_with_onehot()
        views_pool_v = generate_views_pool_v()
        assert views_pool_r.shape[0]==views_pool_r_ohv.shape[0]
        assert views_pool_r.shape[0]==views_pool_v.shape[0]
        size_vp = views_pool_r.shape[0]
        dim_vpt_ohv = views_pool_r_ohv.shape[1]
        assert (num_lat*num_lon)==size_vp and (num_lat+num_lon)==dim_vpt_ohv
        sel_lat_index = np.random.choice(num_lat)
        vpt_pos_container = []
        vpt_vec_container = []
        vm_container = []
        dm_container = []
        for view_index in np.arange(sel_lat_index*num_lon, (sel_lat_index+1)*num_lon):
            vpt_vec = views_pool_r_ohv[view_index]
            x_r, y_r, z_r = views_pool_r[view_index]
            d_r, lat_r, lon_r = cart2geog(x_r, y_r, z_r)
            init_dm_h, init_dm_w = 256, 256
            dm = generate_depth_map(tmv, tmf, x_r, y_r, z_r, init_dm_h, init_dm_w)
            x_v, y_v, z_v = views_pool_v[view_index]
            vis_idx, vis_pts = extract_visible_points(pts, x_v, y_v, z_v)
            vm = np.zeros((self.num_pts,)).astype(np.float32)
            vm[vis_idx] = 1.0
            vpt_pos = np.asarray([x_r/d_r, y_r/d_r, z_r/d_r]).astype(np.float32)
            vpt_pos_container.append(vpt_pos.reshape(1, -1))
            vpt_vec_container.append(vpt_vec.reshape(1, -1))
            dm_container.append(np.expand_dims(dm, axis=0))
            vm_container.append(np.expand_dims(vm, axis=0))
        vpt_pos_stacked = np.concatenate(vpt_pos_container, axis=0)
        vpt_vec_stacked = np.concatenate(vpt_vec_container, axis=0)
        vm_stacked = np.concatenate(vm_container, axis=0)
        dm_stacked = np.concatenate(dm_container, axis=0)
        return model_name, pts, vpt_pos_stacked, vpt_vec_stacked, vm_stacked, dm_stacked
    def __len__(self):
        return len(self.epoch_model_list)



