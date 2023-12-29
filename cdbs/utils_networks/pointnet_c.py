from ..pkgs import *
from ..general import *
from ..custom import *
from .pointnet_utils import *



class PointNetC_Encoder(nn.Module):
    def __init__(self, in_channels, ftr_channels):
        super(PointNetC_Encoder, self).__init__()
        self.in_channels = in_channels
        self.ftr_channels = ftr_channels
        self.inp_tnet = InpTNet()
        self.smlp_1 = SMLP(in_channels, 64, True, 'relu')
        self.smlp_2 = SMLP(64, 64, True, 'relu')
        self.ftr_tnet = FtrTNet(64)
        self.smlp_3 = SMLP(64, 64, True, 'relu')
        self.smlp_4 = SMLP(64, 128, True, 'relu')
        self.smlp_5 = SMLP(128, ftr_channels, True, 'relu')
    def forward(self, pc):
        # pc: [bs, num_pts, in_channels]
        assert pc.size(2)==self.in_channels and pc.size(2)>=3
        if self.in_channels > 3:
            pts = pc[:, :, :3] # [bs, num_pts, 3]
            atr = pc[:, :, 3:] # [bs, num_pts, in_channels-3]
            inp_trans_mat = self.inp_tnet(pts) # [bs, 3, 3]
            pts = torch.bmm(pts, inp_trans_mat) # [bs, num_pts, 3]
            pc = torch.cat((pts, atr), dim=-1) # [bs, num_pts, in_channels]
        else:
            inp_trans_mat = self.inp_tnet(pc) # [bs, 3, 3]
            pc = torch.bmm(pc, inp_trans_mat) # [bs, num_pts, 3]
        pwf = self.smlp_2(self.smlp_1(pc)) # [bs, num_pts, 64]
        ftr_trans_mat = self.ftr_tnet(pwf) # [bs, 64, 64]
        pwf = torch.bmm(pwf, ftr_trans_mat) # [bs, num_pts, 64]
        pwf = self.smlp_5(self.smlp_4(self.smlp_3(pwf))) # [bs, num_pts, ftr_channels]
        cdw = pwf.max(dim=1)[0] # [bs, ftr_channels]
        return inp_trans_mat, ftr_trans_mat, pwf, cdw


class PointNetC_Head(nn.Module):
    def __init__(self, ftr_channels, num_classes):
        super(PointNetC_Head, self).__init__()
        self.ftr_channels = ftr_channels
        self.num_classes = num_classes
        self.fc_1 = nn.Sequential(FC(ftr_channels, 512, True, 'relu'), nn.Dropout(p=0.3))
        self.fc_2 = nn.Sequential(FC(512, 256, True, 'relu'), nn.Dropout(p=0.3))
        self.fc_3 = FC(256, num_classes, False, 'none')
    def forward(self, cdw):
        # cdw: [bs, ftr_channels]
        assert cdw.size(1)==self.ftr_channels
        logits = self.fc_3(self.fc_2(self.fc_1(cdw))) # [bs, num_classes]
        return logits


class PointNetC(nn.Module):
    def __init__(self, in_channels, ftr_channels, num_classes):
        super(PointNetC, self).__init__()
        self.in_channels = in_channels
        self.ftr_channels = ftr_channels
        self.num_classes = num_classes
        self.encoder = PointNetC_Encoder(in_channels, ftr_channels)
        self.head = PointNetC_Head(ftr_channels, num_classes)
    def forward(self, pc):
        # pc: [bs, num_pts, in_channels]
        inp_trans_mat, ftr_trans_mat, pwf, cdw = self.encoder(pc)
        # inp_trans_mat: [bs, 3, 3]
        # ftr_trans_mat: [bs, 64, 64]
        # pwf: [bs, num_pts, ftr_channels]
        # cdw: [bs, ftr_channels]
        logits = self.head(cdw) # [bs, num_classes]
        return inp_trans_mat, ftr_trans_mat, pwf, cdw, logits


