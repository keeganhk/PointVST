from ..pkgs import *
from ..general import *
from ..custom import *



class ModelNet40_SVMClsLoader(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, mode):
        assert mode in ['train', 'test']
        self.dataset_folder = dataset_folder
        self.mode = mode
        self.class_list = [line.strip() for line in open(os.path.join(dataset_folder, 'class_list.txt'), 'r')] 
        self.h5_file_path = os.path.join(dataset_folder, 'pc_1024x6_' + mode + '.h5')
        fid = h5py.File(self.h5_file_path, 'r')
        self.num_models = fid['points'].shape[0]
        fid.close()
    def __getitem__(self, model_index):
        fid = h5py.File(self.h5_file_path, 'r')
        pts = fid['points'][model_index]
        cid = fid['labels'][model_index]
        fid.close()
        pts = bounding_box_normalization(pts).astype(np.float32)[:, 0:3]
        return pts, cid
    def __len__(self):
        return self.num_models


class ModelNet40_TrainLoader(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, ras=True):
        self.dataset_folder = dataset_folder
        self.ras = ras
        self.class_list = [line.strip() for line in open(os.path.join(dataset_folder, 'class_list.txt'), 'r')] 
        self.h5_file_path = os.path.join(dataset_folder, 'pc_1024x6_' + 'train' + '.h5')
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
        if self.ras:
            pc = random_anisotropic_scaling(pc, 2/3, 3/2)
        pc = random_translation(pc, 0.20)
        return pc, cid
    def __len__(self):
        return self.num_models


class ModelNet40_TestLoader(torch.utils.data.Dataset):
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        self.class_list = [line.strip() for line in open(os.path.join(dataset_folder, 'class_list.txt'), 'r')] 
        self.h5_file_path = os.path.join(dataset_folder, 'pc_1024x6_' + 'test' + '.h5')
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



