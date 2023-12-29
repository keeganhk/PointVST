from ..pkgs import *
from ..general import *
from ..custom import *



class ScanObjectNN_TrainLoader(torch.utils.data.Dataset):
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        self.class_list = [line.strip() for line in open(os.path.join(dataset_folder, 'class_list.txt'), 'r')]
        self.h5_file_path = os.path.join(dataset_folder, 'pc_2048x3_' + 'train' + '.h5')
        fid = h5py.File(self.h5_file_path, 'r')
        self.num_models = fid['points'].shape[0]
        fid.close()
    def __getitem__(self, model_index):
        np.random.seed()
        fid = h5py.File(self.h5_file_path, 'r')
        pc = fid['points'][model_index]
        cid = fid['labels'][model_index]
        fid.close()
        pc = bounding_box_normalization(pc).astype(np.float32)
        if np.random.random() > 0.5:
            pc = random_anisotropic_scaling(pc, 2/3, 3/2)
        pc = random_axis_rotation(pc, 'z')
        pc = random_translation(pc, 0.20)
        if np.random.random() > 0.5:
            pc = random_jittering(pc, 0.02, 0.02)
        return pc, cid
    def __len__(self):
        return self.num_models


class ScanObjectNN_TestLoader(torch.utils.data.Dataset):
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        self.class_list = [line.strip() for line in open(os.path.join(dataset_folder, 'class_list.txt'), 'r')] 
        self.h5_file_path = os.path.join(dataset_folder, 'pc_1024x3_' + 'test' + '.h5')
        fid = h5py.File(self.h5_file_path, 'r')
        self.num_models = fid['points'].shape[0]
        fid.close()
    def __getitem__(self, model_index):
        fid = h5py.File(self.h5_file_path, 'r')
        pc = fid['points'][model_index]
        cid = fid['labels'][model_index]
        fid.close()
        pc = bounding_box_normalization(pc).astype(np.float32)
        return pc, cid
    def __len__(self):
        return self.num_models



