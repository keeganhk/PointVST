import os, sys
sys.path.append(os.path.abspath('../../..'))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from cdbs.pkgs import *
from cdbs.general import *
from cdbs.custom import *
data_root = os.path.abspath(os.path.join('../../..', 'data'))
ckpt_root = os.path.abspath(os.path.join('../../..', 'output', 'ckpt'))
from cdbs.utils_datasets.modelnet40 import *
from cdbs.utils_networks.pointnet_r import *
dataset_folder = os.path.join(data_root, 'ModelNet40')
pret_ckpt_name = 'pret_pointnetr'
ckpt_name = 'ft_r_pointnetr_mn40'


net = PointNetR(in_channels=3, ftr_channels=2048).cuda()
net.load_state_dict(torch.load(os.path.join(ckpt_root, ckpt_name + '.pth')))
net.eval()

test_bs = 32
test_set = ModelNet40_TestLoader(dataset_folder)
test_loader = DataLoader(test_set, batch_size=test_bs, shuffle=False, num_workers=8, worker_init_fn=seed_worker, drop_last=False)

num_samples = 0
mean_ndists = 0
for (pc, _) in tqdm(test_loader):
    pc = pc.cuda()
    pts = pc[:, :, :3]
    nms_gt = pc[:, :, 3:]
    bs = pc.size(0)
    with torch.no_grad():
        _, _, _, _, nms_pr = net(pts)
    ndists = normals_distances(nms_pr, nms_gt)
    num_samples += bs
    mean_ndists += ndists.mean(dim=-1).sum().item()
mean_ndists = np.around(mean_ndists/num_samples, 3)
print('mean ndists: {}'.format(mean_ndists))



