from ..pkgs import *
from ..general import *
from ..custom import *



def ShapeNetPart_ObjectsParts():
    objects_names = [
        'airplane', 
        'bag', 
        'cap', 
        'car', 
        'chair', 
        'earphone', 
        'guitar', 
        'knife', 
        'lamp', 
        'laptop', 
        'motorbike', 
        'mug', 
        'pistol', 
        'rocket', 
        'skateboard', 
        'table'
    ]
    objects_parts = {
        'airplane': [0, 1, 2, 3], 
        'bag': [4, 5], 
        'cap': [6, 7], 
        'car': [8, 9, 10, 11], 
        'chair': [12, 13, 14, 15], 
        'earphone': [16, 17, 18], 
        'guitar': [19, 20, 21], 
        'knife': [22, 23],
        'lamp': [24, 25, 26, 27], 
        'laptop': [28, 29], 
        'motorbike': [30, 31, 32, 33, 34, 35], 
        'mug': [36, 37], 
        'pistol': [38, 39, 40], 
        'rocket': [41, 42, 43], 
        'skateboard': [44, 45, 46], 
        'table': [47, 48, 49]}
    return objects_names, objects_parts


def ShapeNetPart_PartsColors():
    objects_names, objects_parts = ShapeNetPart_ObjectsParts()
    num_parts = []
    for k, v in objects_parts.items():
        num_parts.append(len(v))
    cmap = cm.jet
    parts_colors = np.zeros((50, 3))
    i = 0
    for num in num_parts:
        base_colors = cmap(np.linspace(0.1, 0.9, num))[:, 0:3] * 255
        for k in range(num):
            parts_colors[i, ...] = base_colors[k, ...]
            i += 1
    return parts_colors


def ShapeNetPart_ColorCode(points_with_labels):
    assert points_with_labels.ndim==2 and points_with_labels.size(-1)==4
    points = points_with_labels[:, 0:3].unsqueeze(0)
    labels = points_with_labels[:, -1].unsqueeze(0).long()
    parts_colors =  torch.tensor(ShapeNetPart_PartsColors()).unsqueeze(0).to(points_with_labels.device)
    color_codes = index_points(parts_colors, labels)
    points_color_coded = torch.cat((points, color_codes), dim=-1).squeeze(0)
    return points_color_coded


class ShapeNetPart_TrainLoader(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, num_pts):
        self.dataset_folder = dataset_folder
        self.num_pts = num_pts
        self.class_list = [line.strip() for line in open(os.path.join(dataset_folder, 'class_list.txt'), 'r')] 
        self.model_list = [line.strip() for line in open(os.path.join(dataset_folder, 'train_list.txt'), 'r')]
        self.num_models = len(self.model_list)
    def __getitem__(self, model_index):
        np.random.seed()
        model_name = self.model_list[model_index]
        class_name = model_name[:-5]
        class_index = self.class_list.index(class_name)
        load_path = os.path.join(self.dataset_folder, 'point-cloud', str(self.num_pts), class_name, model_name + '.npy')
        pc = bounding_box_normalization(np.load(load_path).astype(np.float32))
        points = pc[:, :3]
        labels = pc[:, 3:]
        points = bounding_box_normalization(random_anisotropic_scaling(points, 2/3, 3/2))
        pc = np.concatenate((points, labels), axis=-1)
        return pc, class_index, model_name
    def __len__(self):
        return self.num_models


class ShapeNetPart_TestLoader(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, num_pts):
        self.dataset_folder = dataset_folder
        self.num_pts = num_pts
        self.class_list = [line.strip() for line in open(os.path.join(dataset_folder, 'class_list.txt'), 'r')] 
        self.model_list = [line.strip() for line in open(os.path.join(dataset_folder, 'test_list.txt'), 'r')]
        self.num_models = len(self.model_list)
    def __getitem__(self, model_index):
        model_name = self.model_list[model_index]
        class_name = model_name[:-5]
        class_index = self.class_list.index(class_name)
        load_path = os.path.join(self.dataset_folder, 'point-cloud', str(self.num_pts), class_name, model_name + '.npy')
        pc = bounding_box_normalization(np.load(load_path).astype(np.float32))
        return pc, class_index, model_name
    def __len__(self):
        return self.num_models



