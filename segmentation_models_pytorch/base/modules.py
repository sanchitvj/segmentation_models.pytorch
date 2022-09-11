import torch
import torch.nn as nn

try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):

        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()

        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ArgMax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


class Clamp(nn.Module):
    def __init__(self, min=0, max=1):
        super().__init__()
        self.min, self.max = min, max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)


class Activation(nn.Module):
    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif name == "argmax":
            self.activation = ArgMax(**params)
        elif name == "argmax2d":
            self.activation = ArgMax(dim=1, **params)
        elif name == "clamp":
            self.activation = Clamp(**params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )

    def forward(self, x):
        return self.activation(x)


class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        elif name == "cbam":
            self.attention = CBAM_Module(**params)
        elif name == "att":
            self.attention = AttentionBlock(**params)
        elif name == "scag":
            self.attention = SCAGModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)


# NOTE: https://www.mdpi.com/2076-3417/10/17/5729/pdf
class SCAGModule(nn.Module):
    def __init__(self, ch_enc_skip, ch_deconv, out_channels, N=16, scale_factor=2, mode="nearest"):
        super(SCAGModule, self).__init__()
        # 1st pool will return (h,w), 2nd will return (1,1)
        self.avg_pool, self.avg_pool_ch = nn.AdaptiveAvgPool2d((None, None)), nn.AdaptiveAvgPool2d(1)
        self.max_pool, self.max_pool_ch = nn.AdaptiveMaxPool2d((None, None)), nn.AdaptiveMaxPool2d(1)
        self.fc1_enc = nn.Sequential(nn.Conv2d(ch_enc_skip, 1, 1), nn.BatchNorm2d(1))
        self.fc1_dec = nn.Sequential(nn.Conv2d(ch_deconv, 1, 1), nn.BatchNorm2d(1))
        self.conv11_enc = nn.Sequential(nn.Conv2d(2 * ch_enc_skip + 1, 1, 1))
        self.conv11_dec = nn.Sequential(nn.Conv2d(2 * ch_deconv + 1, 1, 1))
        self.fc2 = nn.Sequential(nn.Conv2d(1, 1, 7), nn.BatchNorm2d(1))
        # below ch_deconv // N instead of ch_enc_skip // N to make the output of fc3 layers
        # same for encoder and decoder for summation
        self.fc3_enc = nn.Sequential(nn.Conv2d(ch_enc_skip, ch_deconv // N, 1), nn.BatchNorm2d(ch_deconv // N))
        self.fc3_dec = nn.Sequential(nn.Conv2d(ch_deconv, ch_deconv // N, 1), nn.BatchNorm2d(ch_deconv // N))
        self.fc4 = nn.Sequential(nn.Conv2d(ch_deconv // N, out_channels, 1), nn.BatchNorm2d(out_channels))
        self.fc5 = nn.Sequential(nn.Conv2d(ch_enc_skip, out_channels, 1), nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, e, d):
        enc = e
        e_avg, d_avg = self.avg_pool(e), self.avg_pool(d)
        e_max, d_max = self.max_pool(e), self.max_pool(d)
        e_conv1, d_conv1 = self.fc1_enc(e), self.fc1_dec(d)
        e_x, d_x = torch.cat([e_avg, e_max, e_conv1], 1), torch.cat([d_avg, d_max, d_conv1], 1)
        # e_out, d_out = self.fc2(self.conv11_enc(e_x)), self.fc2(self.conv11_dec(d_x))
        e_out, d_out = self.conv11_enc(e_x), self.conv11_dec(d_x)
        sp_out = e_out + d_out  # torch.cat([e_out, d_out]) (element-wise summation)
        sp_out = self.relu(sp_out)
        sp_out = self.sigmoid(sp_out)
        enc = enc * sp_out

        e_av, d_av = self.avg_pool_ch(e), self.avg_pool_ch(d)
        e_mx, d_mx = self.max_pool_ch(e), self.max_pool_ch(d)
        e_conv3_avg, e_conv3_mx = self.fc3_enc(e_av), self.fc3_enc(e_mx)
        d_conv3_avg, d_conv3_mx = self.fc3_dec(d_av), self.fc3_dec(d_mx)
        e_conv3 = e_conv3_avg + e_conv3_mx  # torch.cat([e_conv3_avg, e_conv3_mx])
        d_conv3 = d_conv3_avg + d_conv3_mx
        ch_out = e_conv3 + d_conv3
        ch_out = self.fc4(ch_out)
        ch_out = self.relu(ch_out)
        ch_out = self.sigmoid(ch_out)

        enc = self.fc5(enc)
        out = ch_out * enc

        return out


class AttentionBlock(nn.Module):
    """
    stride = 2; please set --attention parameter to 1 when activate this block. the result name should include "att2".
    To fit in the structure of the V-net: F_l = F_int
    """

    def __init__(self, F_g, F_l, F_int, kernel_size=2, stride=2, scale_factor=2, mode="nearest"):
        super(AttentionBlock, self).__init__()
        self.mode = mode
        self.scale_factor = scale_factor

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),  # reduce num_channels
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(
                F_l,
                F_int,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                bias=True,
            ),  # downsize
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x, visualize=False):
        """
        :param g: gate signal from coarser scale
        :param x: the output of the l-th layer in the encoder
        :param visualize: enable this when plotting attention matrix
        :return:
        """
        x1 = self.W_x(x)
        g1 = self.W_g(g)
        relu = self.relu(g1 + x1)
        sig = self.psi(relu)

        ################ Modifications possible ###############

        alpha = nn.functional.interpolate(sig, scale_factor=self.scale_factor, mode=self.mode)

        if visualize:
            return alpha
        else:
            return x * alpha


class CBAM_Module(nn.Module):
    def __init__(self, in_channels, reduction=16, attention_kernel_size=3):
        super(CBAM_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        k = 2
        self.conv_after_concat = nn.Conv2d(
            k, 1, kernel_size=attention_kernel_size, stride=1, padding=attention_kernel_size // 2
        )
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention module
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)
        # Spatial attention module
        x = module_input * x
        module_input = x
        b, c, h, w = x.size()
        avg = torch.mean(x, 1, True)
        mx, _ = torch.max(x, 1, True)
        x = torch.cat((avg, mx), 1)
        x = self.conv_after_concat(x)
        x = self.sigmoid_spatial(x)
        x = module_input * x
        return x
