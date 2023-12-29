from ..pkgs import *
from ..general import *
from ..custom import *
from .dgcnn_utils import *



class DGCNNC_Encoder(nn.Module):
    def __init__(self, K, D):
        super(DGCNNC_Encoder, self).__init__()
        self.K = K # number of knn neighbors
        self.D = D # codeword dimension
        embedding_dims = [3, 64, 64, 128, 256]
        self.econv_1 = EdgeConv1(embedding_dims[0], embedding_dims[1], K)
        self.econv_2 = EdgeConv1(embedding_dims[1], embedding_dims[2], K)
        self.econv_3 = EdgeConv1(embedding_dims[2], embedding_dims[3], K)
        self.econv_4 = EdgeConv1(embedding_dims[3], embedding_dims[4], K)
        self.fuse = SMLP(int(np.array(embedding_dims[1:]).sum()), D//2, is_bn=True, nl='leakyrelu', slope=0.2)
    def forward(self, pts):
        # pts: [B, N, 3]
        assert pts.size(2) == 3
        K, D = self.K, self.D
        B, N, device = pts.size(0), pts.size(1), pts.device
        pwf_1 = self.econv_1(pts) # [B, N, C1], this step is actually spatial knn
        pwf_2 = self.econv_2(pwf_1) # [B, N, C2]
        pwf_3 = self.econv_3(pwf_2) # [B, N, C3]
        pwf_4 = self.econv_4(pwf_3) # [B, N, C4]
        pwf_c = torch.cat((pwf_1, pwf_2, pwf_3, pwf_4), dim=-1) # [B, N, C1+C2+C3+C4], concatenated point-wise features
        pwf_f = self.fuse(pwf_c) # [B, N, D//2], fused point-wise features
        cdw_max_pooled = pwf_f.max(dim=1)[0] # [B, D//2]
        cdw_avg_pooled = pwf_f.mean(dim=1) # [B, D//2]
        cdw = torch.cat((cdw_max_pooled, cdw_avg_pooled), dim=-1) # [B, D], codeword
        return pwf_f, cdw


class DGCNNC_Head(nn.Module):
    def __init__(self, Ci, Nc):
        super(DGCNNC_Head, self).__init__()
        self.Ci = Ci # input channels
        self.Nc = Nc # number of classes
        head_dims = [Ci, 512, 256, Nc]
        # the first fully-connected layer
        linear_1 = nn.Linear(head_dims[0], head_dims[1], bias=False)
        bn_1 = nn.BatchNorm1d(head_dims[1])
        nl_1 = nn.LeakyReLU(True, 0.2)
        dp_1 = nn.Dropout(0.5)
        self.fc_1 = nn.Sequential(linear_1, bn_1, nl_1, dp_1)
        # the second fully-connected layer
        linear_2 = nn.Linear(head_dims[1], head_dims[2], bias=False)
        bn_2 = nn.BatchNorm1d(head_dims[2])
        nl_2 = nn.LeakyReLU(True, 0.2)
        dp_2 = nn.Dropout(0.5)
        self.fc_2 = nn.Sequential(linear_2, bn_2, nl_2, dp_2)
        # the third fully-connected layer
        self.fc_3 = nn.Linear(head_dims[2], head_dims[3], bias=False)
    def forward(self, cdw):
        # cdw: [B, D]
        Ci, Nc = self.Ci, self.Nc
        B, D, device = cdw.size(0), cdw.size(1), cdw.device
        logits = self.fc_3(self.fc_2(self.fc_1(cdw))) # [B, Nc], logits
        return logits


class DGCNNC(nn.Module):
    def __init__(self, K, D, Nc):
        super(DGCNNC, self).__init__()
        self.K = K # number of knn neighbors
        self.D = D # codeword dimension
        self.Nc = Nc # number of classes
        self.encoder = DGCNNC_Encoder(K, D)
        self.head = DGCNNC_Head(D, Nc)
    def forward(self, pts):
        # pts: [B, N, 3]
        B, N, device = pts.size(0), pts.size(1), pts.device
        pwf, cdw = self.encoder(pts) # pwf: [B, N, D//2], cdw: [B, D]
        logits = self.head(cdw) # [B, Nc], logits
        return pwf, cdw, logits




