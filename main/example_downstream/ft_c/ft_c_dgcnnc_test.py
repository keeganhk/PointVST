import os, sys
sys.path.append(os.path.abspath('../../..'))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from cdbs.pkgs import *
from cdbs.general import *
from cdbs.custom import *
data_root = os.path.abspath(os.path.join('../../..', 'data'))
ckpt_root = os.path.abspath(os.path.join('../../..', 'output', 'ckpt'))
from cdbs.utils_datasets.scanobjectnn import *
from cdbs.utils_networks.dgcnn_c import *
dataset_folder = os.path.join(data_root, 'ScanObjectNN')
pret_ckpt_name = 'pret_dgcnnc'
ckpt_name = 'ft_c_dgcnnc_sonn'


net = DGCNNC(K=20, D=2048, Nc=15).cuda()
net.load_state_dict(torch.load(os.path.join(ckpt_root, ckpt_name + '.pth')))
net.eval()

test_bs = 40
test_set = ScanObjectNN_TestLoader(dataset_folder)
test_loader = DataLoader(test_set, batch_size=test_bs, shuffle=False, num_workers=8, worker_init_fn=seed_worker, drop_last=False)

num_samples = 0
num_correct = 0
for (pc, cid) in tqdm(test_loader):
    pc = pc.float().cuda()
    cid = cid.long().cuda()
    pts = pc[:, :, :3]
    bs = pc.size(0)
    with torch.no_grad():
        _, _, logits = net(pts)
    preds = logits.argmax(dim=-1)
    num_samples += bs
    num_correct += (preds==cid).sum().item()
test_acc = np.around((num_correct/num_samples)*100, 1)
print('test acc: {}%'.format(test_acc))



