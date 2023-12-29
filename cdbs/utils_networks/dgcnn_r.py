from ..pkgs import *
from ..general import *
from ..custom import *
from .dgcnn_utils import *



class DGCNNR_Encoder(nn.Module):
    def __init__(self, K, D):
        super(DGCNNR_Encoder, self).__init__()
        self.K = K # number of knn neighbors
        self.D = D # codeword dimension
        self.econv_1 = EdgeConv2(3, 64, K)
        self.econv_2 = EdgeConv2(64, 64, K)
        self.econv_3 = EdgeConv1(64, 64, K)
        self.fuse = SMLP(64*3, D, True, 'leakyrelu', 0.20)
    def forward(self, pts):
        # pts: [B, N, 3]
        assert pts.size(2) == 3
        K, D = self.K, self.D
        B, N, device = pts.size(0), pts.size(1), pts.device
        pwf_1 = self.econv_1(pts) # [B, N, C1], this step is actually spatial knn
        pwf_2 = self.econv_2(pwf_1) # [B, N, C2]
        pwf_3 = self.econv_3(pwf_2) # [B, N, C3]
        concat_123 = torch.cat((pwf_1, pwf_2, pwf_3), dim=-1) # [B, N, C1+C2+C3=192]
        pwf_4 = self.fuse(concat_123) # [B, N, D]
        cdw = pwf_4.max(dim=1)[0] # [B, D]
        return concat_123, cdw


class DGCNNR_Head(nn.Module):
    def __init__(self, pwf_channels, cdw_channels):
        super(DGCNNR_Head, self).__init__()
        self.pwf_channels = pwf_channels
        self.cdw_channels = cdw_channels
        concat_channels = pwf_channels + cdw_channels
        self.smlp_1 = nn.Sequential(SMLP(concat_channels, 256, True, 'leakyrelu', 0.20), nn.Dropout(p=0.5))
        self.smlp_2 = nn.Sequential(SMLP(256, 256, True, 'leakyrelu', 0.20), nn.Dropout(p=0.5))
        self.smlp_3 = SMLP(256, 128, True, 'leakyrelu', 0.20)
        self.smlp_4 = SMLP(128, 3, False, 'none')
    def forward(self, pwf, cdw):
        # pwf: [bs, num_pts, pwf_channels]
        # cdw: [bs, cdw_channels]
        assert pwf.size(2)==self.pwf_channels
        assert cdw.size(1)==self.cdw_channels
        bs, num_pts, device = pwf.size(0), pwf.size(1), pwf.device
        cdw_exp = cdw.unsqueeze(1).repeat(1, num_pts, 1) # [bs, num_pts, cdw_channels]
        concat = torch.cat((pwf, cdw_exp), dim=-1) # [bs, num_pts, pwf_channels+cdw_channels=1216]
        nms = self.smlp_4(self.smlp_3(self.smlp_2(self.smlp_1(concat)))) # [bs, num_pts, 3]
        return nms


class DGCNNR(nn.Module):
    def __init__(self, num_knn_neighbors, cdw_channels):
        super(DGCNNR, self).__init__()
        self.num_knn_neighbors = num_knn_neighbors
        self.cdw_channels = cdw_channels
        self.encoder = DGCNNR_Encoder(num_knn_neighbors, cdw_channels)
        self.head = DGCNNR_Head(192, cdw_channels)
    def forward(self, pts):
        # pts: [bs, num_pts, 3]
        B, N, device = pts.size(0), pts.size(1), pts.device
        pwf, cdw = self.encoder(pts) # pwf: [bs, num_pts, 192], cdw: [bs, cdw_channels]
        nms = self.head(pwf, cdw) # [bs, num_pts, 3]
        return pwf, cdw, nms



