from ..pkgs import *
from ..general import *
from ..custom import *
from .pointnet_utils import *



class PointNetR_Encoder(nn.Module):
    def __init__(self, in_channels, ftr_channels):
        super(PointNetR_Encoder, self).__init__()
        self.in_channels = in_channels
        self.ftr_channels = ftr_channels
        self.inp_tnet = InpTNet()
        self.smlp_1 = SMLP(in_channels, 64, True, 'relu')
        self.smlp_2 = SMLP(64, 128, True, 'relu')
        self.smlp_3 = SMLP(128, 128, True, 'relu')
        self.ftr_tnet = FtrTNet(128)
        self.smlp_4 = SMLP(128, 512, True, 'relu')
        self.smlp_5 = SMLP(512, ftr_channels, True, 'relu')
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
        pwf_1 = self.smlp_1(pc) # [bs, num_pts, 64]
        pwf_2 = self.smlp_2(pwf_1) # [bs, num_pts, 128]
        pwf_3 = self.smlp_3(pwf_2) # [bs, num_pts, 128]
        ftr_trans_mat = self.ftr_tnet(pwf_3) # [bs, 128, 128]
        pwf_3_trans = torch.bmm(pwf_3, ftr_trans_mat) # [bs, num_pts, 128]
        pwf_4 = self.smlp_4(pwf_3_trans) # [bs, num_pts, 512]
        pwf_5 = self.smlp_5(pwf_4) # [bs, num_pts, ftr_channels]
        cdw = pwf_5.max(dim=1)[0] # [bs, ftr_channels]
        pwf = torch.cat((pwf_1, pwf_2, pwf_3, pwf_4, pwf_5), dim=-1) # [bs, num_pts, 64+128+128+512+2048=2880]
        return inp_trans_mat, ftr_trans_mat, pwf, cdw


class PointNetR_Head(nn.Module):
    def __init__(self, pwf_channels, cdw_channels):
        super(PointNetR_Head, self).__init__()
        self.pwf_channels = pwf_channels
        self.cdw_channels = cdw_channels
        concat_channels = pwf_channels + cdw_channels
        self.smlp_1 = nn.Sequential(SMLP(concat_channels, 256, True, 'relu'), nn.Dropout(p=0.2))
        self.smlp_2 = nn.Sequential(SMLP(256, 256, True, 'relu'), nn.Dropout(p=0.2))
        self.smlp_3 = SMLP(256, 128, True, 'relu')
        self.smlp_4 = SMLP(128, 3, False, 'none')
    def forward(self, pwf, cdw):
        # pwf: [bs, num_pts, pwf_channels]
        # cdw: [bs, cdw_channels]
        assert pwf.size(2)==self.pwf_channels
        assert cdw.size(1)==self.cdw_channels
        bs, num_pts, device = pwf.size(0), pwf.size(1), pwf.device
        cdw_exp = cdw.unsqueeze(1).repeat(1, num_pts, 1) # [bs, num_pts, cdw_channels]
        concat = torch.cat((pwf, cdw_exp), dim=-1) # [bs, num_pts, pwf_channels+cdw_channels=4928]
        nms = self.smlp_4(self.smlp_3(self.smlp_2(self.smlp_1(concat)))) # [bs, num_pts, 3]
        return nms


class PointNetR(nn.Module):
    def __init__(self, in_channels, ftr_channels):
        super(PointNetR, self).__init__()
        self.encoder = PointNetR_Encoder(in_channels, ftr_channels)
        self.head = PointNetR_Head(64+128+128+512+ftr_channels, ftr_channels)
    def forward(self, pc):
        # pc: [bs, num_pts, in_channels]
        inp_trans_mat, ftr_trans_mat, pwf, cdw = self.encoder(pc)
        # inp_trans_mat: [bs, 3, 3]
        # ftr_trans_mat: [bs, 128, 128]
        # pwf: [bs, num_pts, 64+128+128+512+ftr_channels]
        # cdw: [bs, ftr_channels]
        nms = self.head(pwf, cdw) # [bs, num_pts, 3]
        return inp_trans_mat, ftr_trans_mat, pwf, cdw, nms



