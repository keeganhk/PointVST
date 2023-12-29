from ..pkgs import *
from ..general import *
from ..custom import *



class EdgeConv1(nn.Module):
    def __init__(self, Ci, Co, K):
        super(EdgeConv1, self).__init__()
        self.Ci = Ci # input channels
        self.Co = Co # output channels
        self.K = K # number of knn neighbors
        self.smlp = SMLP(Ci*2, Co, is_bn=True, nl='leakyrelu', slope=0.2)
    def forward(self, pwf):
        # pwf: [B, N, Ci], point-wise features
        Ci, Co, K = self.Ci, self.Co, self.K
        B, N, device = pwf.size(0), pwf.size(1), pwf.device
        # 1) perform knn in feature space
        # knn_idx = knn_search(pwf.detach().cpu(), pwf.detach().cpu(), K+1)[:, :, 1:] # [B, N, K]
        knn_idx = knn_search(pwf.detach(), pwf.detach(), K+1)[:, :, 1:] # [B, N, K]
        # 2) construct edge features
        ftr_d = pwf.unsqueeze(2).repeat(1, 1, K, 1) # [B, N, K, Ci], duplicated features
        ftr_n = index_points(pwf, knn_idx) # [B, N, K, Ci], neighboring features
        ftr_e = torch.cat((ftr_d, ftr_n - ftr_d), dim=-1) # [B, N, K, 2*Ci], edge features
        # 3) apply MLP on edge features
        ftr_e_updated = self.smlp(ftr_e.view(B, N*K, -1)).view(B, N, K, -1) # [B, N, K, Co]
        # 4) aggregate updated features in local neighborhoods
        ftr_a = torch.max(ftr_e_updated, dim=2)[0] # [B, N, Co]
        return ftr_a


class EdgeConv2(nn.Module):
    def __init__(self, Ci, Co, K):
        super(EdgeConv2, self).__init__()
        self.Ci = Ci # input channels
        self.Co = Co # output channels
        self.K = K # number of knn neighbors
        smlp_1 = SMLP(Ci*2, Co, is_bn=True, nl='leakyrelu', slope=0.2)
        smlp_2 = SMLP(Co, Co, is_bn=True, nl='leakyrelu', slope=0.2)
        self.smlp = nn.Sequential(smlp_1, smlp_2)
    def forward(self, pwf):
        # pwf: [B, N, Ci], point-wise features
        Ci, Co, K = self.Ci, self.Co, self.K
        B, N, device = pwf.size(0), pwf.size(1), pwf.device
        # 1) perform knn in feature space
        # knn_idx = knn_search(pwf.detach().cpu(), pwf.detach().cpu(), K+1)[:, :, 1:] # [B, N, K]
        knn_idx = knn_search(pwf.detach(), pwf.detach(), K+1)[:, :, 1:] # [B, N, K]
        # 2) construct edge features
        ftr_d = pwf.unsqueeze(2).repeat(1, 1, K, 1) # [B, N, K, Ci], duplicated features
        ftr_n = index_points(pwf, knn_idx) # [B, N, K, Ci], neighboring features
        ftr_e = torch.cat((ftr_d, ftr_n - ftr_d), dim=-1) # [B, N, K, 2*Ci], edge features
        # 3) apply MLP on edge features
        ftr_e_updated = self.smlp(ftr_e.view(B, N*K, -1)).view(B, N, K, -1) # [B, N, K, Co]
        # 4) aggregate updated features in local neighborhoods
        ftr_a = torch.max(ftr_e_updated, dim=2)[0] # [B, N, Co]
        return ftr_a



