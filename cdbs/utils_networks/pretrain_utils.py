from ..pkgs import *
from ..general import *
from ..custom import *



class ViewPositionalEncoding(nn.Module):
    def __init__(self, num_lat, num_lon, lift_channels):
        super(ViewPositionalEncoding, self).__init__()
        self.num_lat = num_lat
        self.num_lon = num_lon
        self.lift_channels = lift_channels
        assert np.mod(lift_channels, 2) == 0
        self.lat_fc = nn.Sequential(FC(num_lat, 64, is_bn=True, nl='relu'), FC(64, lift_channels//2, is_bn=True, nl='relu'))
        self.lon_fc = nn.Sequential(FC(num_lon, 64, is_bn=True, nl='relu'), FC(64, lift_channels//2, is_bn=True, nl='relu'))
    def forward(self, vpt_vec):
        assert vpt_vec.size(1) == self.num_lat + self.num_lon
        lat_ohv = vpt_vec[:, :self.num_lat]
        lon_ohv = vpt_vec[:, self.num_lat:]
        vpe = torch.cat((self.lat_fc(lat_ohv), self.lon_fc(lon_ohv)), dim=-1)
        return vpe


class ViewConditionedPointWiseFusion(nn.Module):
    def __init__(self, dim_pwf, dim_cdw, fuse_channels):
        super(ViewConditionedPointWiseFusion, self).__init__()
        self.dim_pwf = dim_pwf
        self.dim_cdw = dim_cdw
        self.fuse_channels = fuse_channels
        self.view_pos_enc = ViewPositionalEncoding(num_lat=4, num_lon=8, lift_channels=256)
        self.pwf_smlp = SMLP(dim_pwf, 1024, is_bn=True, nl='relu')
        self.cdw_fc = FC(dim_cdw, 1024, is_bn=True, nl='relu')
        self.fuse = SMLP(1024+1024+256, fuse_channels, is_bn=True, nl='relu')
    def forward(self, pwf, cdw, vpt_vec):
        num_pts = pwf.size(1)
        vpe = self.view_pos_enc(vpt_vec)
        vpe_dup = vpe.unsqueeze(1).repeat(1, num_pts, 1)
        pwf_linear = self.pwf_smlp(pwf)
        cdw_linear_dup = self.cdw_fc(cdw).unsqueeze(1).repeat(1, num_pts, 1)
        concat = torch.cat((pwf_linear, cdw_linear_dup, vpe_dup), dim=-1)
        pwf_fuse = self.fuse(concat)
        return vpe, pwf_fuse


class ViewSpecificCodewordEmbedding(nn.Module):
    def __init__(self, fuse_channels, dim_vs_cdw):
        super(ViewSpecificCodewordEmbedding, self).__init__()
        self.fuse_channels = fuse_channels
        self.dim_vs_cdw = dim_vs_cdw
        smlp_1 = SMLP(fuse_channels, 256, is_bn=True, nl='relu')
        smlp_2 = SMLP(256, 128, is_bn=True, nl='relu')
        smlp_3 = SMLP(128, 1, is_bn=False, nl='sigmoid')
        self.pwv_predictor = nn.Sequential(smlp_1, smlp_2, smlp_3)
        self.fc_1 = FC(fuse_channels, dim_vs_cdw, is_bn=True, nl='relu')
        self.fc_2 = FC(dim_vs_cdw+256, dim_vs_cdw, is_bn=True, nl='relu')
    def forward(self, vpe, pwf_fuse):
        assert pwf_fuse.min() >= 0
        num_pts = pwf_fuse.size(1)
        vm_pr = self.pwv_predictor(pwf_fuse).squeeze(-1)
        pwf_masked_maxpooled = (pwf_fuse * vm_pr.unsqueeze(-1)).max(dim=1)[0]
        vs_cdw = self.fc_2(torch.cat((self.fc_1(pwf_masked_maxpooled), vpe), dim=-1))
        return vm_pr, vs_cdw


class ResConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, leaky_slope=0.0):
        super(ResConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False, padding_mode='zeros'),
            nn.BatchNorm2d(in_channels)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False, padding_mode='zeros'),
            nn.BatchNorm2d(out_channels)
        )
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False), nn.BatchNorm2d(out_channels))
        self.nl = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
    def forward(self, in_ftr):
        out_ftr = self.conv_2(self.nl(self.conv_1(in_ftr)))
        if self.in_channels != self.out_channels:
            out_ftr = self.nl(self.shortcut(in_ftr) + out_ftr)
        else:
            out_ftr = self.nl(in_ftr + out_ftr)
        return out_ftr


class ImageTranslator(nn.Module):
    def __init__(self, dim_vec):
        super(ImageTranslator, self).__init__()
        self.dim_vec = dim_vec
        self.lifting = FC(dim_vec, 8*24*24, is_bn=True, nl='relu')
        self.conv_0 = CU(8, 32, 3, True, 'relu')
        self.upscale_x2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_1 = ResConv2D(32, 64)
        self.conv_2 = ResConv2D(64, 128)
        self.conv_3 = CU(128, 512, 3, True, 'relu')
        self.head_dm = nn.Sequential(CU(512, 128, 3, True, 'relu'), CU(128, 64, 3, True, 'relu'), CU(64, 1, 3, False, 'sigmoid'))
        self.head_bm = nn.Sequential(CU(512, 128, 3, True, 'relu'), CU(128, 64, 3, True, 'relu'), CU(64, 1, 3, False, 'sigmoid'))
        self.head_be = nn.Sequential(CU(512, 128, 3, True, 'relu'), CU(128, 64, 3, True, 'relu'), CU(64, 1, 3, False, 'sigmoid'))
    def forward(self, vec):
        bs = vec.size(0)
        ftr_0 = self.conv_0(self.lifting(vec).view(bs, 8, 24, 24))
        ftr_1 = self.conv_1(self.upscale_x2(ftr_0))
        ftr_2 = self.conv_2(self.upscale_x2(ftr_1))
        ftr_3 = self.conv_3(ftr_2)
        dm_trans = self.head_dm(ftr_3).squeeze(1)
        bm_trans = self.head_bm(ftr_3).squeeze(1)
        be_trans = self.head_be(ftr_3).squeeze(1)
        return dm_trans, bm_trans, be_trans



