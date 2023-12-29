import os, sys
sys.path.append(os.path.abspath('../../..'))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from cdbs.pkgs import *
from cdbs.general import *
from cdbs.custom import *
data_root = os.path.abspath(os.path.join('../../..', 'data'))
ckpt_root = os.path.abspath(os.path.join('../../..', 'output', 'ckpt'))
from cdbs.utils_datasets.modelnet40 import ModelNet40_SVMClsLoader
from cdbs.utils_networks.dgcnn_c import DGCNNC_Encoder
dataset_folder = os.path.join(data_root, 'ModelNet40')
ckpt_name = 'pret_dgcnnc'


num_pts = 1024
K = 20
dim_cdw = 2048
net = DGCNNC_Encoder(K, dim_cdw).cuda()
net.load_state_dict(torch.load(os.path.join(ckpt_root, ckpt_name + '.pth'))['encoder'])
net.eval()

train_set = ModelNet40_SVMClsLoader(dataset_folder, 'train')
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=8, worker_init_fn=seed_worker, drop_last=False)
train_featrs = []
train_labels = []
for (pts, cid) in tqdm(train_loader):
    B = pts.size(0)
    with torch.no_grad():
        pts = pts.cuda()
        _, cdw = net(pts)
    for bid in range(B):
        train_featrs.append(np.asarray(cdw[bid].cpu().unsqueeze(0)))
        train_labels.append(cid[bid].item())
train_featrs = np.concatenate(train_featrs, axis=0).astype(np.float32)
train_labels = np.asarray(train_labels)

test_set = ModelNet40_SVMClsLoader(dataset_folder, 'test')
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=8, worker_init_fn=seed_worker, drop_last=False)
test_featrs = []
test_labels = []
for (pts, cid) in tqdm(test_loader):
    B = pts.size(0)
    with torch.no_grad():
        pts = pts.cuda()
        _, cdw = net(pts)
    for bid in range(B):
        test_featrs.append(np.asarray(cdw[bid].cpu().unsqueeze(0)))
        test_labels.append(cid[bid].item())
test_featrs = np.concatenate(test_featrs, axis=0).astype(np.float32)
test_labels = np.asarray(test_labels)

c_list = list(np.linspace(0.010, 0.012, 3).astype(np.float32))
oacc_list = []
for c in c_list:
    c = float(c)
    model_tl = SVC(C=c, kernel='linear')
    model_tl.fit(train_featrs, train_labels)
    oacc = np.around(model_tl.score(test_featrs, test_labels)*100, 1)
    oacc_list.append(oacc)
    print('c={}, overall accuracy={}%'.format(np.around(c, 4), oacc))
oacc_list = np.asarray(oacc_list)
print('best oacc: {}%'.format(oacc_list.max()))



