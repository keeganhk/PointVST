import os, sys
sys.path.append(os.path.abspath('../..'))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from cdbs.pkgs import *
from cdbs.general import *
from cdbs.custom import *
data_root = os.path.abspath(os.path.join('../..', 'data'))
ckpt_root = os.path.abspath(os.path.join('../..', 'output', 'ckpt'))
from cdbs.utils_datasets.shapenetcore import shapenetcore_sub_list_sampling, ShapeNetCore_PretrainLoader
from cdbs.utils_networks.pretrain_wrappers import DGCNNR_Pretrainer
dataset_folder = os.path.join(data_root, 'ShapeNetCore')
ckpt_name = 'pret_dgcnnr'


num_lat = 4
num_lon = 8
num_pts = 1024
dim_vs_cdw = 2048
K = 20
dim_pwf = 192
dim_cdw = 1024
net = DGCNNR_Pretrainer(K, dim_pwf, dim_cdw, dim_vs_cdw).cuda()
max_lr = 1e-3
min_lr = 5e-6
num_epc = 600
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=max_lr, weight_decay=1e-6)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epc, eta_min=min_lr)

net.train()
for epc in range(1, num_epc+1):
    num_perc = 48
    epoch_model_list = shapenetcore_sub_list_sampling(dataset_folder, num_perc)
    pretr_set = ShapeNetCore_PretrainLoader(dataset_folder, epoch_model_list, num_pts)
    pretr_loader = DataLoader(pretr_set, batch_size=24, shuffle=True, num_workers=12, worker_init_fn=seed_worker, drop_last=True)
    num_processed = 0
    loss_records = [0, 0, 0, 0]
    for (name_list, pts, vpt_pos, vpt_vec, vm_gt, dm_gt_init) in tqdm(pretr_loader):
        bs = len(name_list)
        pts = pts.cuda()
        vpt_pos = vpt_pos.cuda()
        vpt_vec = vpt_vec.cuda()
        vm_gt = vm_gt.cuda()
        dm_gt_init = dm_gt_init.cuda()
        dm_gt, bm_gt, be_gt = prepare_image_supervisions(dm_gt_init)
        optimizer.zero_grad()
        vm_pr, _, dm_trans, bm_trans, be_trans = net(pts, vpt_vec)
        vm_pr = vm_pr.view(bs, num_lon, num_pts)
        dm_trans = dm_trans.view(bs, num_lon, 96, 96)
        bm_trans = bm_trans.view(bs, num_lon, 96, 96)
        be_trans = be_trans.view(bs, num_lon, 96, 96)
        v_loss = F.binary_cross_entropy(vm_pr, vm_gt)
        t_loss_dm = F.l1_loss(dm_trans, dm_gt)
        t_loss_bm = F.binary_cross_entropy(bm_trans, bm_gt)
        t_loss_be = F.binary_cross_entropy(be_trans, be_gt)
        wv, wt_dm, wt_bm, wt_be = 1.0, 1.0, 1.0, 1.0
        overall_loss = v_loss*wv + t_loss_dm*wt_dm + t_loss_bm*wt_bm + t_loss_be*wt_be
        overall_loss.backward()
        optimizer.step()
        num_processed += (bs*num_lon)
        loss_records[0] += (bs*num_lon * v_loss.item())
        loss_records[1] += (bs*num_lon * t_loss_dm.item())
        loss_records[2] += (bs*num_lon * t_loss_bm.item())
        loss_records[3] += (bs*num_lon * t_loss_be.item())
    current_lr = np.around(optimizer.param_groups[0]['lr'], 6)
    scheduler.step()
    loss_records[0] = np.around(loss_records[0]/num_processed, 4)
    loss_records[1] = np.around(loss_records[1]/num_processed, 4)
    loss_records[2] = np.around(loss_records[2]/num_processed, 4)
    loss_records[3] = np.around(loss_records[3]/num_processed, 4)
    print('epoch: {} (lr: {}), v_loss: {}, t_loss_dm: {}, t_loss_bm: {}, t_loss_be: {}'.format(
    align_number(epc, 4), current_lr, loss_records[0], loss_records[1], loss_records[2], loss_records[3]))
    torch.save({'pretrainer': net.state_dict(), 'encoder': net.encoder.state_dict()}, os.path.join(ckpt_root, ckpt_name + '.pth'))



