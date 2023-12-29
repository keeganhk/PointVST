from ..pkgs import *
from ..general import *
from ..custom import *
from .pretrain_utils import *
from .dgcnn_c import DGCNNC_Encoder
from .dgcnn_r import DGCNNR_Encoder
from .dgcnn_s import DGCNNS_Encoder
from .pointnet_c import PointNetC_Encoder
from .pointnet_r import PointNetR_Encoder
from .pointnet_s import PointNetS_Encoder
from .pointnet_utils import ftr_trans_regularizer



class DGCNNC_Pretrainer(nn.Module):
    def __init__(self, K, dim_pwf, dim_cdw, dim_vs_cdw):
        super(DGCNNC_Pretrainer, self).__init__()
        self.K = K
        self.dim_pwf = dim_pwf
        self.dim_cdw = dim_cdw
        self.dim_vs_cdw = dim_vs_cdw
        self.encoder = DGCNNC_Encoder(K, dim_cdw)
        fuse_channels = 1024
        self.vcpw_fusion = ViewConditionedPointWiseFusion(dim_pwf, dim_cdw, fuse_channels)
        self.vs_cdw_embedding = ViewSpecificCodewordEmbedding(fuse_channels, dim_vs_cdw)
        self.image_translt = ImageTranslator(dim_vs_cdw)
    def forward(self, pts, vpt_vec):
        bs, num_pts, num_lon = pts.size(0), pts.size(1), vpt_vec.size(1)
        pwf, cdw = self.encoder(pts)
        pwf_dup = pwf.unsqueeze(1).repeat(1, num_lon, 1, 1).view(bs*num_lon, num_pts, -1)
        cdw_dup = cdw.unsqueeze(1).repeat(1, num_lon, 1).view(bs*num_lon, -1)
        vpt_vec_rsp = vpt_vec.view(bs*num_lon, -1)
        vpe, pwf_fuse = self.vcpw_fusion(pwf_dup, cdw_dup, vpt_vec_rsp)
        vm_pr, vs_cdw = self.vs_cdw_embedding(vpe, pwf_fuse)
        dm_trans, bm_trans, be_trans = self.image_translt(vs_cdw)
        return vm_pr, vs_cdw, dm_trans, bm_trans, be_trans


class DGCNNS_Pretrainer(nn.Module):
    def __init__(self, K, dim_pwf, dim_cdw, dim_vs_cdw):
        super(DGCNNS_Pretrainer, self).__init__()
        self.K = K
        self.dim_pwf = dim_pwf
        self.dim_cdw = dim_cdw
        self.dim_vs_cdw = dim_vs_cdw
        self.encoder = DGCNNS_Encoder(K, dim_cdw)
        fuse_channels = 1024
        self.vcpw_fusion = ViewConditionedPointWiseFusion(dim_pwf, dim_cdw, fuse_channels)
        self.vs_cdw_embedding = ViewSpecificCodewordEmbedding(fuse_channels, dim_vs_cdw)
        self.image_translt = ImageTranslator(dim_vs_cdw)
    def forward(self, pts, vpt_vec):
        bs, num_pts, num_lon = pts.size(0), pts.size(1), vpt_vec.size(1)
        pwf, cdw = self.encoder(pts)
        pwf_dup = pwf.unsqueeze(1).repeat(1, num_lon, 1, 1).view(bs*num_lon, num_pts, -1)
        cdw_dup = cdw.unsqueeze(1).repeat(1, num_lon, 1).view(bs*num_lon, -1)
        vpt_vec_rsp = vpt_vec.view(bs*num_lon, -1)
        vpe, pwf_fuse = self.vcpw_fusion(pwf_dup, cdw_dup, vpt_vec_rsp)
        vm_pr, vs_cdw = self.vs_cdw_embedding(vpe, pwf_fuse)
        dm_trans, bm_trans, be_trans = self.image_translt(vs_cdw)
        return vm_pr, vs_cdw, dm_trans, bm_trans, be_trans


class DGCNNR_Pretrainer(nn.Module):
    def __init__(self, K, dim_pwf, dim_cdw, dim_vs_cdw):
        super(DGCNNR_Pretrainer, self).__init__()
        self.K = K
        self.dim_pwf = dim_pwf
        self.dim_cdw = dim_cdw
        self.dim_vs_cdw = dim_vs_cdw
        self.encoder = DGCNNR_Encoder(K, dim_cdw)
        fuse_channels = 1024
        self.vcpw_fusion = ViewConditionedPointWiseFusion(dim_pwf, dim_cdw, fuse_channels)
        self.vs_cdw_embedding = ViewSpecificCodewordEmbedding(fuse_channels, dim_vs_cdw)
        self.image_translt = ImageTranslator(dim_vs_cdw)
    def forward(self, pts, vpt_vec):
        bs, num_pts, num_lon = pts.size(0), pts.size(1), vpt_vec.size(1)
        pwf, cdw = self.encoder(pts)
        pwf_dup = pwf.unsqueeze(1).repeat(1, num_lon, 1, 1).view(bs*num_lon, num_pts, -1)
        cdw_dup = cdw.unsqueeze(1).repeat(1, num_lon, 1).view(bs*num_lon, -1)
        vpt_vec_rsp = vpt_vec.view(bs*num_lon, -1)
        vpe, pwf_fuse = self.vcpw_fusion(pwf_dup, cdw_dup, vpt_vec_rsp)
        vm_pr, vs_cdw = self.vs_cdw_embedding(vpe, pwf_fuse)
        dm_trans, bm_trans, be_trans = self.image_translt(vs_cdw)
        return vm_pr, vs_cdw, dm_trans, bm_trans, be_trans


class PointNetC_Pretrainer(nn.Module):
    def __init__(self, dim_pwf, dim_cdw, dim_vs_cdw):
        super(PointNetC_Pretrainer, self).__init__()
        self.dim_pwf = dim_pwf
        self.dim_cdw = dim_cdw
        self.dim_vs_cdw = dim_vs_cdw
        self.encoder = PointNetC_Encoder(3, dim_cdw)
        fuse_channels = 1024
        self.vcpw_fusion = ViewConditionedPointWiseFusion(dim_pwf, dim_cdw, fuse_channels)
        self.vs_cdw_embedding = ViewSpecificCodewordEmbedding(fuse_channels, dim_vs_cdw)
        self.image_translt = ImageTranslator(dim_vs_cdw)
    def forward(self, pts, vpt_vec):
        bs, num_pts, num_lon = pts.size(0), pts.size(1), vpt_vec.size(1)
        _, ftr_trans_mat, pwf, cdw = self.encoder(pts)
        pwf_dup = pwf.unsqueeze(1).repeat(1, num_lon, 1, 1).view(bs*num_lon, num_pts, -1)
        cdw_dup = cdw.unsqueeze(1).repeat(1, num_lon, 1).view(bs*num_lon, -1)
        vpt_vec_rsp = vpt_vec.view(bs*num_lon, -1)
        vpe, pwf_fuse = self.vcpw_fusion(pwf_dup, cdw_dup, vpt_vec_rsp)
        vm_pr, vs_cdw = self.vs_cdw_embedding(vpe, pwf_fuse)
        dm_trans, bm_trans, be_trans = self.image_translt(vs_cdw)
        return ftr_trans_mat, vm_pr, vs_cdw, dm_trans, bm_trans, be_trans


class PointNetS_Pretrainer(nn.Module):
    def __init__(self, dim_pwf, dim_cdw, dim_vs_cdw):
        super(PointNetS_Pretrainer, self).__init__()
        self.dim_pwf = dim_pwf
        self.dim_cdw = dim_cdw
        self.dim_vs_cdw = dim_vs_cdw
        self.encoder = PointNetS_Encoder(3, dim_cdw)
        fuse_channels = 1024
        self.vcpw_fusion = ViewConditionedPointWiseFusion(dim_pwf, dim_cdw, fuse_channels)
        self.vs_cdw_embedding = ViewSpecificCodewordEmbedding(fuse_channels, dim_vs_cdw)
        self.image_translt = ImageTranslator(dim_vs_cdw)
    def forward(self, pts, vpt_vec):
        bs, num_pts, num_lon = pts.size(0), pts.size(1), vpt_vec.size(1)
        _, ftr_trans_mat, pwf, cdw = self.encoder(pts)
        pwf_dup = pwf.unsqueeze(1).repeat(1, num_lon, 1, 1).view(bs*num_lon, num_pts, -1)
        cdw_dup = cdw.unsqueeze(1).repeat(1, num_lon, 1).view(bs*num_lon, -1)
        vpt_vec_rsp = vpt_vec.view(bs*num_lon, -1)
        vpe, pwf_fuse = self.vcpw_fusion(pwf_dup, cdw_dup, vpt_vec_rsp)
        vm_pr, vs_cdw = self.vs_cdw_embedding(vpe, pwf_fuse)
        dm_trans, bm_trans, be_trans = self.image_translt(vs_cdw)
        return ftr_trans_mat, vm_pr, vs_cdw, dm_trans, bm_trans, be_trans


class PointNetR_Pretrainer(nn.Module):
    def __init__(self, dim_pwf, dim_cdw, dim_vs_cdw):
        super(PointNetR_Pretrainer, self).__init__()
        self.dim_pwf = dim_pwf
        self.dim_cdw = dim_cdw
        self.dim_vs_cdw = dim_vs_cdw
        self.encoder = PointNetR_Encoder(3, dim_cdw)
        fuse_channels = 1024
        self.vcpw_fusion = ViewConditionedPointWiseFusion(dim_pwf, dim_cdw, fuse_channels)
        self.vs_cdw_embedding = ViewSpecificCodewordEmbedding(fuse_channels, dim_vs_cdw)
        self.image_translt = ImageTranslator(dim_vs_cdw)
    def forward(self, pts, vpt_vec):
        bs, num_pts, num_lon = pts.size(0), pts.size(1), vpt_vec.size(1)
        _, ftr_trans_mat, pwf, cdw = self.encoder(pts)
        pwf_dup = pwf.unsqueeze(1).repeat(1, num_lon, 1, 1).view(bs*num_lon, num_pts, -1)
        cdw_dup = cdw.unsqueeze(1).repeat(1, num_lon, 1).view(bs*num_lon, -1)
        vpt_vec_rsp = vpt_vec.view(bs*num_lon, -1)
        vpe, pwf_fuse = self.vcpw_fusion(pwf_dup, cdw_dup, vpt_vec_rsp)
        vm_pr, vs_cdw = self.vs_cdw_embedding(vpe, pwf_fuse)
        dm_trans, bm_trans, be_trans = self.image_translt(vs_cdw)
        return ftr_trans_mat, vm_pr, vs_cdw, dm_trans, bm_trans, be_trans



