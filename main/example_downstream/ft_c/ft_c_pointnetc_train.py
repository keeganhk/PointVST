import os, sys
sys.path.append(os.path.abspath('../../..'))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from cdbs.pkgs import *
from cdbs.general import *
from cdbs.custom import *
data_root = os.path.abspath(os.path.join('../../..', 'data'))
ckpt_root = os.path.abspath(os.path.join('../../..', 'output', 'ckpt'))
from cdbs.utils_datasets.scanobjectnn import *
from cdbs.utils_networks.pointnet_c import *
dataset_folder = os.path.join(data_root, 'ScanObjectNN')
pret_ckpt_name = 'pret_pointnetc'
ckpt_name = 'ft_c_pointnetc_sonn'


train_bs = 64
train_set = ScanObjectNN_TrainLoader(dataset_folder)
train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True, num_workers=8, worker_init_fn=seed_worker, drop_last=True)
test_bs = 32
test_set = ScanObjectNN_TestLoader(dataset_folder)
test_loader = DataLoader(test_set, batch_size=test_bs, shuffle=False, num_workers=8, worker_init_fn=seed_worker, drop_last=False)

net = PointNetC(in_channels=3, ftr_channels=1024, num_classes=15).cuda()
net.encoder.load_state_dict(torch.load(os.path.join(ckpt_root, pret_ckpt_name + '.pth'))['encoder'])
max_lr = 5e-2
min_lr = 5e-4
num_epc = 500
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=max_lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epc, eta_min=min_lr)

best_test_acc = 0
for epc in range(1, num_epc+1):
    net.train()
    epoch_loss_list = [0, 0, 0]
    num_samples = 0
    num_correct = 0
    for (pc, cid) in tqdm(train_loader):
        pc = pc.float().cuda()
        cid = cid.long().cuda()
        pts = pc[:, :, :3]
        pts = index_points(pts, get_fps_idx(pts, 1024))
        bs = pc.size(0)
        optimizer.zero_grad()
        _, ftr_trans_mat, _, _, logits = net(pts)
        loss_cls = smooth_cross_entropy(logits, cid, eps=0.20)
        loss_reg = ftr_trans_regularizer(ftr_trans_mat)
        loss = loss_cls + loss_reg * 0.0001
        loss.backward()
        optimizer.step()
        preds = logits.argmax(dim=-1).detach()
        num_samples += bs
        num_correct += (preds==cid).sum().item()
        epoch_loss_list[0] += (loss.item() * bs)
        epoch_loss_list[1] += (loss_cls.item() * bs)
        epoch_loss_list[2] += (loss_reg.item() * bs)
    scheduler.step()
    epoch_loss_list[0] = np.around(epoch_loss_list[0]/num_samples, 4)
    epoch_loss_list[1] = np.around(epoch_loss_list[1]/num_samples, 4)
    epoch_loss_list[2] = np.around(epoch_loss_list[2]/num_samples, 4)
    train_acc = np.around((num_correct/num_samples)*100, 2)
    print('epoch: {}: train acc: {}%, cls loss: {}, reg loss: {}'.format(epc, train_acc, epoch_loss_list[1], epoch_loss_list[2]))
    if epc<=3 or epc>=(num_epc//2):
        net.eval()
        num_samples = 0
        num_correct = 0
        for (pc, cid) in tqdm(test_loader):
            pc = pc.float().cuda()
            cid = cid.long().cuda()
            pts = pc[:, :, :3]
            bs = pc.size(0)
            with torch.no_grad():
                _, _, _, _, logits = net(pts)
            preds = logits.argmax(dim=-1)
            num_samples += bs
            num_correct += (preds==cid).sum().item()
        test_acc = np.around((num_correct/num_samples)*100, 1)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(net.state_dict(), os.path.join(ckpt_root, ckpt_name + '.pth'))
        print('epoch: {}: test acc: {}%,  best test acc: {}%'.format(epc, test_acc, best_test_acc))



