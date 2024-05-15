import torch
import torch.nn as nn
import torch
from networks.WTNet.vgg import *
import math
from networks.Unet.unet import Unet
from thop import profile


class Decoder_block(nn.Module):
    def __init__(self, in_channel, out_channel, attention=False):
        super(Decoder_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)
        self.eca = ECABlock(channels=out_channel)
        self.Spatial = SpatialAttention()
        self.attention = attention

    def forward(self, inputs1, inputs2):
        if self.attention:
            inputs1 = self.eca(inputs1)
            Spatial_map = self.Spatial(inputs1)
            inputs1 = inputs1 * Spatial_map
            outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        else:
            outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class Encoder(nn.Module):
    """
    for input size of (B, 3, 256, 256)
    output size is: feat1, feat2, feat3, feat4, feat5

    torch.Size([1, 64, 256, 256])
    torch.Size([1, 128, 128, 128])
    torch.Size([1, 256, 64, 64])
    torch.Size([1, 512, 32, 32])
    torch.Size([1, 512, 16, 16])
    """

    def __init__(self, in_channel):
        super(Encoder, self).__init__()
        self.backbone = VGG16(pretrained=True, in_channels=in_channel)

    def forward(self, x):
        feat1, feat2, feat3, feat4, feat5 = self.backbone(x)

        return feat1, feat2, feat3, feat4, feat5


class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, bias=1):
        super(ECABlock, self).__init__()

        # 设计自适应卷积核，便于后续做1*1卷积
        kernel_size = int(abs((math.log(channels, 2) + bias) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 全局平局池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 基于1*1卷积学习通道之间的信息
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 首先，空间维度做全局平局池化，[b,c,h,w]==>[b,c,1,1]
        v = self.avg_pool(x)
        # 然后，基于1*1卷积学习通道之间的信息；其中，使用前面设计的自适应卷积核
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # 最终，经过sigmoid 激活函数处理
        v = self.sigmoid(v)
        return x * v


class Tooth_multi_scale(nn.Module):
    """
    all feature map are sampling to 128*128, then concat in channel dimension
    finally, execute channel attention to all channels
    """

    def __init__(self):
        super(Tooth_multi_scale, self).__init__()
        self.input1_down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.input2_out = nn.Identity()
        self.input3_up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.input4_up = nn.UpsamplingBilinear2d(scale_factor=4)
        self.channel_atten = ECABlock(channels=960)

    def forward(self, input1, input2, input3, input4):
        out1 = self.input1_down(input1)
        out2 = self.input2_out(input2)
        out3 = self.input3_up(input3)
        out4 = self.input4_up(input4)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        channel_atten_out = self.channel_atten(out)
        return channel_atten_out


class Bone_multi_scale(nn.Module):
    """
    all feature map are sampling to 64*64, then concat in channel dimension
    finally, execute channel attention to all channels
    """

    def __init__(self):
        super(Bone_multi_scale, self).__init__()
        self.input1_down = nn.MaxPool2d(kernel_size=4, stride=4)
        self.input2_down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.input3_out = nn.Identity()
        self.input4_up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.channel_atten = ECABlock(channels=960)

    def forward(self, input1, input2, input3, input4):
        out1 = self.input1_down(input1)
        out2 = self.input2_down(input2)
        out3 = self.input3_out(input3)
        out4 = self.input4_up(input4)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        channel_atten_out = self.channel_atten(out)
        return channel_atten_out


# class Attention_Shift_Module(nn.Module):
#     def __init__(self):
#         super(Attention_Shift_Module, self).__init__()
#
#     def forward(self, input_binary, input_origin):
#         x1 = torch.multiply(input_binary, input_origin)
#         out = torch.cat([input_origin, x1], dim=1)
#         return out


class SpatialAttention(nn.Module):  # Spatial Attention Module
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out


class multi_scale_feature(nn.Module):
    def __init__(self, zoom=None, input_feature_size=None, in_multi_size=None, channel=None):
        super(multi_scale_feature, self).__init__()
        self.input_feature_size = input_feature_size
        self.in_multi_size = in_multi_size
        self.zoom = zoom
        self.channel = channel
        self.conv = nn.Conv2d(in_channels=960, out_channels=channel, kernel_size=1, stride=1)
        self.atten = SpatialAttention()

        if self.zoom == 'UP':
            self.k = input_feature_size / in_multi_size
            self.up = nn.UpsamplingBilinear2d(scale_factor=self.k)
        elif self.zoom == 'DOWN':
            self.avg = nn.AdaptiveAvgPool2d(self.input_feature_size)
        elif self.zoom == "None":
            self.none = nn.Identity()

    def forward(self, input_feature, in_multi):
        if self.zoom == 'UP':
            out_up = self.up(in_multi)
            out_adjust_channel = self.conv(out_up)
            x_add = torch.add(out_adjust_channel, input_feature)
            spatial_attention_map = self.atten(x_add)
            out = torch.mul(spatial_attention_map, input_feature)
            return out

        if self.zoom == 'DOWN':
            out_down = self.avg(in_multi)
            out_adjust_channel = self.conv(out_down)
            x_add = torch.add(out_adjust_channel, input_feature)
            spatial_attention_map = self.atten(x_add)
            out = torch.mul(spatial_attention_map, input_feature)
            return out
        if self.zoom == 'None':
            out_none = self.none(in_multi)
            out_adjust_channel = self.conv(out_none)
            x_add = torch.add(out_adjust_channel, input_feature)
            spatial_attention_map = self.atten(x_add)
            out = torch.mul(spatial_attention_map, input_feature)
            return out


class Binary_mask(nn.Module):
    def __init__(self, num_classes=2):
        super(Binary_mask, self).__init__()
        self.num_classes = num_classes
        self.encoder = Encoder(in_channel=3)
        self.decoder4 = Decoder_block(in_channel=1024, out_channel=512)
        self.decoder3 = Decoder_block(in_channel=768, out_channel=256)
        self.decoder2 = Decoder_block(in_channel=384, out_channel=128)
        self.decoder1 = Decoder_block(in_channel=192, out_channel=64)
        self.final = nn.Conv2d(64, self.num_classes, 1)

    def forward(self, x):
        feat1, feat2, feat3, feat4, feat5 = self.encoder(x)
        out4 = self.decoder4(feat4, feat5)
        out3 = self.decoder3(feat3, out4)
        out2 = self.decoder2(feat2, out3)
        out1 = self.decoder1(feat1, out2)
        out_last = self.final(out1)
        return out_last


#
# class input_enhancement(nn.Module):
#     def __init__(self):
#         super(input_enhancement, self).__init__()
#         self.conv = nn.Conv2d(5, 3, kernel_size=7, padding=3, bias=False)
#
#     def forward(self, origin, binary_mask):
#         x1 = torch.cat([origin, binary_mask], dim=1)
#         out = self.conv(x1)
#         return out

class input_enhancement(nn.Module):
    def __init__(self):
        super(input_enhancement, self).__init__()
        self.conv = nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, origin, binary_mask):
        x1 = torch.mul(origin, binary_mask)
        out = torch.add(x1, origin)
        out = torch.cat([x1, origin, out], dim=1)
        out = self.conv(out)
        # out = self.relu(out)
        return out


class Tooth_bone_separation(nn.Module):
    def __init__(self):
        super(Tooth_bone_separation, self).__init__()
        self.encoder = Encoder(in_channel=3)

        self.Tdecoder = nn.ModuleList(
            [Decoder_block(in_channel=1024, out_channel=512),
             Decoder_block(in_channel=768, out_channel=256),
             Decoder_block(in_channel=384, out_channel=128),
             Decoder_block(in_channel=192, out_channel=64)]
        )

        self.Bdecoder = nn.ModuleList(
            [Decoder_block(in_channel=1024, out_channel=512),
             Decoder_block(in_channel=768, out_channel=256),
             Decoder_block(in_channel=384, out_channel=128),
             Decoder_block(in_channel=192, out_channel=64)]
        )

        self.Tmulti = nn.ModuleList(
            [
                multi_scale_feature(zoom='UP', input_feature_size=256, in_multi_size=128, channel=64),
                multi_scale_feature(zoom='None', input_feature_size=128, in_multi_size=128, channel=128),
                multi_scale_feature(zoom='DOWN', input_feature_size=64, in_multi_size=128, channel=256),
                multi_scale_feature(zoom='DOWN', input_feature_size=32, in_multi_size=128, channel=512)
            ]
        )

        self.Bmulti = nn.ModuleList(
            [
                multi_scale_feature(zoom='UP', input_feature_size=256, in_multi_size=64, channel=64),
                multi_scale_feature(zoom='UP', input_feature_size=128, in_multi_size=64, channel=128),
                multi_scale_feature(zoom='None', input_feature_size=64, in_multi_size=64, channel=256),
                multi_scale_feature(zoom='DOWN', input_feature_size=32, in_multi_size=64, channel=512)
            ]
        )

        self.Tooth_multi_scale = Tooth_multi_scale()
        self.Bone_multi_scale = Bone_multi_scale()
        self.Tfinal = nn.Conv2d(64, 3, 1)  # background, WT, SM
        self.Bfinal = nn.Conv2d(64, 2, 1)  # background, AB

    def forward(self, x):
        feat1, feat2, feat3, feat4, feat5 = self.encoder(x)

        # Tooth_multi = self.Tooth_multi_scale(feat1, feat2, feat3, feat4)  # (B, 960, 128, 128)
        # Tooth_feat1 = self.Tmulti[0](input_feature=feat1, in_multi=Tooth_multi)
        # Tooth_feat2 = self.Tmulti[1](input_feature=feat2, in_multi=Tooth_multi)
        # Tooth_feat3 = self.Tmulti[2](input_feature=feat3, in_multi=Tooth_multi)
        # Tooth_feat4 = self.Tmulti[3](input_feature=feat4, in_multi=Tooth_multi)
        # Tout4 = self.Tdecoder[0](Tooth_feat4, feat5)
        # Tout3 = self.Tdecoder[1](Tooth_feat3, Tout4)
        # Tout2 = self.Tdecoder[2](Tooth_feat2, Tout3)
        # Tout1 = self.Tdecoder[3](Tooth_feat1, Tout2)

        Tout4 = self.Tdecoder[0](feat4, feat5)
        Tout3 = self.Tdecoder[1](feat3, Tout4)
        Tout2 = self.Tdecoder[2](feat2, Tout3)
        Tout1 = self.Tdecoder[3](feat1, Tout2)

        out_tooth_last = self.Tfinal(Tout1)

        # Bone_multi = self.Bone_multi_scale(feat1, feat2, feat3, feat4)  # (B, 960, 64, 64)
        # Bone_feat1 = self.Bmulti[0](input_feature=feat1, in_multi=Bone_multi)
        # Bone_feat2 = self.Bmulti[1](input_feature=feat2, in_multi=Bone_multi)
        # Bone_feat3 = self.Bmulti[2](input_feature=feat3, in_multi=Bone_multi)
        # Bone_feat4 = self.Bmulti[3](input_feature=feat4, in_multi=Bone_multi)
        # Bout4 = self.Bdecoder[0](Bone_feat4, feat5)
        # Bout3 = self.Bdecoder[1](Bone_feat3, Bout4)
        # Bout2 = self.Bdecoder[2](Bone_feat2, Bout3)
        # Bout1 = self.Bdecoder[3](Bone_feat1, Bout2)

        Bout4 = self.Bdecoder[0](feat4, feat5)
        Bout3 = self.Bdecoder[1](feat3, Bout4)
        Bout2 = self.Bdecoder[2](feat2, Bout3)
        Bout1 = self.Bdecoder[3](feat1, Bout2)

        out_bone_last = self.Bfinal(Bout1)

        return out_tooth_last, out_bone_last


class WTNet(nn.Module):
    def __init__(self):
        super(WTNet, self).__init__()
        self.Binary = Binary_mask()
        self.input_enhancement = input_enhancement()
        self.TBS = Tooth_bone_separation()

    def forward(self, x):
        Binary_out = self.Binary(x)
        Binary_map = torch.nn.functional.softmax(Binary_out, dim=1)
        Binary_map = torch.argmax(Binary_map, dim=1, keepdim=True)
        enhancement = self.input_enhancement(x, Binary_map)
        out_tooth_last, out_bone_last = self.TBS(enhancement)
        return Binary_out, out_tooth_last, out_bone_last


class Unet_SS(nn.Module):
    def __init__(self):
        super(Unet_SS, self).__init__()
        self.Binary = Binary_mask()
        self.input_enhancement = input_enhancement()
        self.encoder = Encoder(in_channel=3)
        self.TBS = Tooth_bone_separation()
        self.Tmulti = nn.ModuleList(
            [
                multi_scale_feature(zoom='UP', input_feature_size=256, in_multi_size=128, channel=64),
                multi_scale_feature(zoom='None', input_feature_size=128, in_multi_size=128, channel=128),
                multi_scale_feature(zoom='DOWN', input_feature_size=64, in_multi_size=128, channel=256),
                multi_scale_feature(zoom='DOWN', input_feature_size=32, in_multi_size=128, channel=512)
            ]
        )
        self.Bmulti = nn.ModuleList(
            [
                multi_scale_feature(zoom='UP', input_feature_size=256, in_multi_size=64, channel=64),
                multi_scale_feature(zoom='UP', input_feature_size=128, in_multi_size=64, channel=128),
                multi_scale_feature(zoom='None', input_feature_size=64, in_multi_size=64, channel=256),
                multi_scale_feature(zoom='DOWN', input_feature_size=32, in_multi_size=64, channel=512)
            ]
        )
        self.Tdecoder = nn.ModuleList(
            [Decoder_block(in_channel=1024, out_channel=512),
             Decoder_block(in_channel=768, out_channel=256),
             Decoder_block(in_channel=384, out_channel=128),
             Decoder_block(in_channel=192, out_channel=64)]
        )

        self.Bdecoder = nn.ModuleList(
            [Decoder_block(in_channel=1024, out_channel=512),
             Decoder_block(in_channel=768, out_channel=256),
             Decoder_block(in_channel=384, out_channel=128),
             Decoder_block(in_channel=192, out_channel=64)]
        )
        self.Tfinal = nn.Conv2d(64, 3, 1)  # background, WT, SM
        self.Bfinal = nn.Conv2d(64, 2, 1)  # background, AB

    def forward(self, x):
        # Binary_out = self.Binary(x)
        # Binary_map = torch.nn.functional.softmax(Binary_out, dim=1)
        # Binary_map = torch.argmax(Binary_map, dim=1, keepdim=True)
        # enhancement = self.input_enhancement(x, Binary_map)

        feat1, feat2, feat3, feat4, feat5 = self.encoder(x)
        Tout4 = self.Tdecoder[0](feat4, feat5)
        Tout3 = self.Tdecoder[1](feat3, Tout4)
        Tout2 = self.Tdecoder[2](feat2, Tout3)
        Tout1 = self.Tdecoder[3](feat1, Tout2)
        out_tooth_last = self.Tfinal(Tout1)

        Bout4 = self.Bdecoder[0](feat4, feat5)
        Bout3 = self.Bdecoder[1](feat3, Bout4)
        Bout2 = self.Bdecoder[2](feat2, Bout3)
        Bout1 = self.Bdecoder[3](feat1, Bout2)
        out_bone_last = self.Bfinal(Bout1)
        return out_tooth_last, out_bone_last


class UNet_IE(nn.Module):
    def __init__(self):
        super(UNet_IE, self).__init__()
        self.unet = Unet()
        self.input_enhancement = input_enhancement()
        self.Binary = Binary_mask()

    def forward(self, x):
        Binary_out = self.Binary(x)
        Binary_map = torch.nn.functional.softmax(Binary_out, dim=1)
        Binary_map = torch.argmax(Binary_map, dim=1, keepdim=True)
        enhancement = self.input_enhancement(x, Binary_map)
        out = self.unet(enhancement)
        return Binary_out, out


if __name__ == '__main__':
    model = WTNet()
    a = torch.rand(size=(1, 3, 256, 256))
    b, c, d = model(a)
    print(b.shape, c.shape, d.shape)

    # flop, para = profile(model, inputs=(a,))
    # print('Flops:', "%.2fM" % (flop / 1e6), 'Params:', "%.2fM" % (para / 1e6))
