import os, sys
sys.path.append(os.path.abspath('../../..'))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from cdbs.pkgs import *
from cdbs.general import *
from cdbs.custom import *
data_root = os.path.abspath(os.path.join('../../..', 'data'))
ckpt_root = os.path.abspath(os.path.join('../../..', 'output', 'ckpt'))
from cdbs.utils_datasets.shapenetpart import *
from cdbs.utils_networks.dgcnn_s import *
dataset_folder = os.path.join(data_root, 'ShapeNetPart')
snp_objects_names, snp_objects_parts = ShapeNetPart_ObjectsParts()
pret_ckpt_name = 'pret_dgcnns'
ckpt_name = 'ft_s_dgcnns_snp'


train_bs = 64
train_set = ShapeNetPart_TrainLoader(dataset_folder, 2048)
train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True, num_workers=8, worker_init_fn=seed_worker, drop_last=True)
test_bs = 32
test_set = ShapeNetPart_TestLoader(dataset_folder, 2048)
test_loader = DataLoader(test_set, batch_size=test_bs, shuffle=False, num_workers=8, worker_init_fn=seed_worker, drop_last=False)

net = DGCNNS(num_knn_neighbors=40, cdw_channels=1024, num_object_classes=16, num_part_classes=50).cuda()
net.encoder.load_state_dict(torch.load(os.path.join(ckpt_root, pret_ckpt_name + '.pth'))['encoder'])
max_lr = 5e-2
min_lr = 5e-4
num_epc = 350
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=max_lr, momentum=0.9, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epc, eta_min=min_lr)

best_test_miou = 0
for epc in range(1, num_epc+1):
    net.train()
    epoch_loss = 0
    num_samples = 0
    for (pc, cid, name_list) in tqdm(train_loader):
        pc = pc.cuda()
        cid = cid.long().cuda()
        points = pc[:, :, 0:3]
        labels = pc[:, :, -1]
        bs, num_pts = labels.size(0), labels.size(1)
        optimizer.zero_grad()
        _, _, logits = net(points, cid)
        loss = smooth_cross_entropy(logits.view(bs*num_pts, -1), labels.view(bs*num_pts).long(), eps=0.05)
        loss.backward()
        optimizer.step()
        num_samples += bs
        epoch_loss += (loss.item() * bs)
    scheduler.step()
    epoch_loss = np.around(epoch_loss/num_samples, 4)
    print('epoch: {}, seg loss: {}'.format(epc, epoch_loss))
    cond_1 = (epc<=int(num_epc*0.75) and np.mod(epc, 10)==0)
    cond_2 = (epc>=int(num_epc*0.75) and np.mod(epc, 1)==0)
    if epc<=3 or cond_1 or cond_2:
        net.eval()
        iou_list = []
        for (pc, cid, name_list) in tqdm(test_loader):
            pc = pc.cuda()
            cid = cid.long().cuda()
            bs, num_pts = pc.size(0), pc.size(1)
            points = pc[:, :, 0:3]
            labels = pc[:, :, -1]
            with torch.no_grad():
                _, _, logits = net(points, cid)
            preds = logits.argmax(dim=-1)
            labels = np.asarray(labels.cpu())
            preds = np.asarray(preds.cpu())
            for bid in range(bs):
                L_this = labels[bid]
                P_this = preds[bid]
                class_name = name_list[bid][:-5]
                parts = snp_objects_parts[class_name]
                this_parts_iou = []
                for part_this in parts:
                    if (L_this==part_this).sum() == 0:
                        this_parts_iou.append(1.0)
                    else:
                        I = np.sum(np.logical_and(P_this==part_this, L_this==part_this))
                        U = np.sum(np.logical_or(P_this==part_this, L_this==part_this))
                        this_parts_iou.append(float(I) / float(U))
                this_iou = np.array(this_parts_iou).mean()
                iou_list.append(this_iou)
        test_miou = np.around(np.array(iou_list).mean()*100, 1)
        if test_miou >= best_test_miou:
            best_test_miou = test_miou
            torch.save(net.state_dict(), os.path.join(ckpt_root, ckpt_name + '.pth'))
        print('epoch: {}: test miou: {}%,  best test miou: {}%'.format(epc, test_miou, best_test_miou))



