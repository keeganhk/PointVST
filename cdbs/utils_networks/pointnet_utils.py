from ..pkgs import *
from ..general import *
from ..custom import *



class InpTNet(nn.Module):
    def __init__(self):
        super(InpTNet, self).__init__()
        self.smlp_1 = SMLP(3, 64, True, 'relu')
        self.smlp_2 = SMLP(64, 128, True, 'relu')
        self.smlp_3 = SMLP(128, 1024, True, 'relu')
        self.fc_1 = FC(1024, 512, True, 'relu')
        self.fc_2 = FC(512, 256, True, 'relu')
        self.fc_3 = FC(256, 9, False, 'none')
        self.idt_mat = torch.from_numpy(np.eye(3).flatten().astype(np.float32)).view(1, 3*3)
    def forward(self, pts):
        # pts: [bs, num_pts, 3]
        assert pts.size(2) == 3
        bs, device = pts.size(0), pts.device
        cdw = self.smlp_3(self.smlp_2(self.smlp_1(pts))).max(dim=1)[0] # [bs, 1024]
        base = self.idt_mat.repeat(bs, 1).to(device).view(bs, 3, 3)
        inp_trans_mat = base + self.fc_3(self.fc_2(self.fc_1(cdw))).view(bs, 3, 3)
        return inp_trans_mat # [bs, 3, 3]


class FtrTNet(nn.Module):
    def __init__(self, ftr_channels):
        super(FtrTNet, self).__init__()
        self.ftr_channels = ftr_channels
        self.out_channels = ftr_channels * ftr_channels
        self.smlp_1 = SMLP(ftr_channels, 64, True, 'relu')
        self.smlp_2 = SMLP(64, 128, True, 'relu')
        self.smlp_3 = SMLP(128, 1024, True, 'relu')
        self.fc_1 = FC(1024, 512, True, 'relu')
        self.fc_2 = FC(512, 256, True, 'relu')
        self.fc_3 = FC(256, self.out_channels, False, 'none')
        self.idt_mat = torch.from_numpy(np.eye(ftr_channels).flatten().astype(np.float32)).view(1, self.out_channels)
    def forward(self, pwf):
        # pwf: [bs, num_pts, ftr_channels]
        assert pwf.size(2) == self.ftr_channels
        bs, device = pwf.size(0), pwf.device
        cdw = self.smlp_3(self.smlp_2(self.smlp_1(pwf))).max(dim=1)[0] # [bs, 1024]
        base = self.idt_mat.repeat(bs, 1).to(device).view(bs, self.ftr_channels, self.ftr_channels)
        ftr_trans_mat = base + self.fc_3(self.fc_2(self.fc_1(cdw))).view(bs, self.ftr_channels, self.ftr_channels)
        return ftr_trans_mat # [bs, ftr_channels, ftr_channels]


def ftr_trans_regularizer(ftr_trans_mat):
    # ftr_trans_mat: [bs, num_channels, num_channels]
    assert ftr_trans_mat.size(1) == ftr_trans_mat.size(2)
    bs, num_channels, device = ftr_trans_mat.size(0), ftr_trans_mat.size(1), ftr_trans_mat.device
    I = torch.eye(num_channels).to(device).unsqueeze(0).repeat(bs, 1, 1) # [bs, num_channels, num_channels]
    reg_loss = torch.mean(torch.norm(torch.bmm(ftr_trans_mat, ftr_trans_mat.transpose(2, 1))-I, dim=(1, 2)))
    return reg_loss



