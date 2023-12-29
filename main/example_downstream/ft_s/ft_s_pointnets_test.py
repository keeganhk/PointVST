import os, sys
sys.path.append(os.path.abspath('../../..'))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from cdbs.pkgs import *
from cdbs.general import *
from cdbs.custom import *
data_root = os.path.abspath(os.path.join('../../..', 'data'))
ckpt_root = os.path.abspath(os.path.join('../../..', 'output', 'ckpt'))
from cdbs.utils_datasets.shapenetpart import *
from cdbs.utils_networks.pointnet_s import *
dataset_folder = os.path.join(data_root, 'ShapeNetPart')
snp_objects_names, snp_objects_parts = ShapeNetPart_ObjectsParts()
pret_ckpt_name = 'pret_pointnets'
ckpt_name = 'ft_s_pointnets_snp'


net = PointNetS(in_channels=3, ftr_channels=2048, num_object_classes=16, num_part_classes=50).cuda()
net.load_state_dict(torch.load(os.path.join(ckpt_root, ckpt_name + '.pth')))
net.eval()

test_bs = 24
test_set = ShapeNetPart_TestLoader(dataset_folder, 2048)
test_loader = DataLoader(test_set, batch_size=test_bs, shuffle=False, num_workers=8, worker_init_fn=seed_worker, drop_last=False)

iou_list = []
for (pc, cid, name_list) in tqdm(test_loader):
    pc = pc.cuda()
    cid = cid.long().cuda()
    bs, num_pts = pc.size(0), pc.size(1)
    points = pc[:, :, 0:3]
    labels = pc[:, :, -1]
    with torch.no_grad():
        _, _, _, _, logits = net(points, cid)
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
print('instance-averaged mIoU: {}%'.format(test_miou))



