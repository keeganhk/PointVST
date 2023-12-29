import os, sys
sys.path.append(os.path.abspath('../../..'))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from cdbs.pkgs import *
from cdbs.general import *
from cdbs.custom import *
data_root = os.path.abspath(os.path.join('../../..', 'data'))
ckpt_root = os.path.abspath(os.path.join('../../..', 'output', 'ckpt'))
from cdbs.utils_datasets.modelnet40 import *
from cdbs.utils_networks.dgcnn_r import *
dataset_folder = os.path.join(data_root, 'ModelNet40')
pret_ckpt_name = 'pret_dgcnnr'
ckpt_name = 'ft_r_dgcnnr_mn40'


train_bs = 128
train_set = ModelNet40_TrainLoader(dataset_folder, ras=False)
train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True, num_workers=8, worker_init_fn=seed_worker, drop_last=True)

net = DGCNNR(num_knn_neighbors=20, cdw_channels=1024).cuda()
net.encoder.load_state_dict(torch.load(os.path.join(ckpt_root, pret_ckpt_name + '.pth'))['encoder'])
max_lr = 5e-2
min_lr = 1e-4
num_epc = 250
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=max_lr, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epc, eta_min=min_lr)

for epc in range(1, num_epc+1):
    net.train()
    epoch_loss = 0
    num_samples = 0
    for (pc, _) in tqdm(train_loader):
        pc = pc.cuda()
        pts = pc[:, :, :3]
        nms_gt = pc[:, :, 3:]
        bs = pc.size(0)
        optimizer.zero_grad()
        _, _, nms_pr = net(pts)
        ndists = normals_distances(nms_pr, nms_gt)
        loss = ndists.mean()
        loss.backward()
        optimizer.step()
        num_samples += bs
        epoch_loss += (loss.item() * bs)
    scheduler.step()
    epoch_loss = np.around(epoch_loss/num_samples, 5)
    print('epoch: {}, loss: {}'.format(epc, epoch_loss))
    torch.save(net.state_dict(), os.path.join(ckpt_root, ckpt_name + '.pth'))



