from ..pkgs import *
from ..general import *
from ..custom import *
from .dgcnn_utils import *



class DGCNNS_Encoder(nn.Module):
    def __init__(self, K, D):
        super(DGCNNS_Encoder, self).__init__()
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


class DGCNNS_Head(nn.Module):
    def __init__(self, pwf_channels, cdw_channels, num_object_classes, num_part_classes):
        super(DGCNNS_Head, self).__init__()
        self.pwf_channels = pwf_channels
        self.cdw_channels = cdw_channels
        self.num_object_classes = num_object_classes
        self.num_part_classes = num_part_classes
        self.lift = FC(num_object_classes, 64, True, 'leakyrelu', 0.20)
        concat_channels = pwf_channels + cdw_channels + 64
        self.smlp_1 = nn.Sequential(SMLP(concat_channels, 256, True, 'leakyrelu', 0.20), nn.Dropout(p=0.5))
        self.smlp_2 = nn.Sequential(SMLP(256, 256, True, 'leakyrelu', 0.20), nn.Dropout(p=0.5))
        self.smlp_3 = SMLP(256, 128, True, 'leakyrelu', 0.20)
        self.smlp_4 = SMLP(128, num_part_classes, False, 'none')
    def forward(self, pwf, cdw, cid):
        # pwf: [bs, num_pts, pwf_channels]
        # cdw: [bs, cdw_channels]
        # cid: [bs]
        assert pwf.size(2)==self.pwf_channels
        assert cdw.size(1)==self.cdw_channels
        assert cid.max().item() < self.num_object_classes
        bs, num_pts, device = pwf.size(0), pwf.size(1), pwf.device
        cdw_exp = cdw.unsqueeze(1).repeat(1, num_pts, 1) # [bs, num_pts, cdw_channels]
        cid_one_hot = F.one_hot(cid, self.num_object_classes).float().to(device) # [bs, num_object_classes]
        cid_lifted = self.lift(cid_one_hot) # [bs, 64]
        cid_lifted_exp = cid_lifted.unsqueeze(1).repeat(1, num_pts, 1) # [bs, num_pts, 64]
        concat = torch.cat((pwf, cdw_exp, cid_lifted_exp), dim=-1) # [bs, num_pts, pwf_channels+cdw_channels+64=1280]
        seg_logits = self.smlp_4(self.smlp_3(self.smlp_2(self.smlp_1(concat)))) # [bs, num_pts, num_part_classes]
        return seg_logits


class DGCNNS(nn.Module):
    def __init__(self, num_knn_neighbors, cdw_channels, num_object_classes, num_part_classes):
        super(DGCNNS, self).__init__()
        self.num_knn_neighbors = num_knn_neighbors
        self.cdw_channels = cdw_channels
        self.num_object_classes = num_object_classes
        self.num_part_classes = num_part_classes
        self.encoder = DGCNNS_Encoder(num_knn_neighbors, cdw_channels)
        self.head = DGCNNS_Head(192, cdw_channels, num_object_classes, num_part_classes)
    def forward(self, pts, cid):
        # pts: [bs, num_pts, 3]
        # cid: [bs]
        B, N, device = pts.size(0), pts.size(1), pts.device
        pwf, cdw = self.encoder(pts) # pwf: [bs, num_pts, 192], cdw: [bs, cdw_channels]
        seg_logits = self.head(pwf, cdw, cid) # [bs, num_pts, num_part_classes]
        return pwf, cdw, seg_logits



