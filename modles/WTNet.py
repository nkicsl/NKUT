from collections import OrderedDict
import torch
import torch.nn as nn
import math


class Decoder_block(nn.Module):
    def __init__(self, num_classes=2, init_features=64):
        super(Decoder_block, self).__init__()
        features = init_features
        out_channels = num_classes

        self.upconv4 = nn.ConvTranspose3d(
            features * 8, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = Decoder_block._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = Decoder_block._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = Decoder_block._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = Decoder_block._block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, fea1, fea2, fea3, fea4, fea5):
        dec4 = self.upconv4(fea5)
        dec4 = torch.cat((dec4, fea4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, fea3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, fea2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, fea1), dim=1)
        dec1 = self.decoder1(dec1)
        outputs = self.conv(dec1)
        return outputs

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(  # 有序字典
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class Encoder(nn.Module):
    """
    for input size of (B, 1, 64, 64, 64)
    output size is: feat1, feat2, feat3, feat4, feat5

    torch.Size([1, 64, 256, 256])
    torch.Size([1, 128, 128, 128])
    torch.Size([1, 256, 64, 64])
    torch.Size([1, 512, 32, 32])
    torch.Size([1, 512, 16, 16])
    """

    def __init__(self, in_channels=1, init_features=64):
        super(Encoder, self).__init__()

        features = init_features
        self.encoder1 = Encoder._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = Encoder._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = Encoder._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = Encoder._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = Encoder._block(features * 8, features * 8, name="bottleneck")

    def forward(self, x):
        feat1 = self.encoder1(x)
        feat2 = self.encoder2(self.pool1(feat1))
        feat3 = self.encoder3(self.pool2(feat2))
        feat4 = self.encoder4(self.pool3(feat3))
        feat5 = self.bottleneck(self.pool4(feat4))

        return feat1, feat2, feat3, feat4, feat5

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(  # 有序字典
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, bias=1):
        super(ECABlock, self).__init__()

        # 设计自适应卷积核，便于后续做1*1卷积
        kernel_size = int(abs((math.log(channels, 2) + bias) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 全局平局池化
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # 基于1*1卷积学习通道之间的信息
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 首先，空间维度做全局平局池化，[b,c,h,w,d]==>[b,c,1,1,1]
        v = self.avg_pool(x)
        # 然后，基于1*1卷积学习通道之间的信息；其中，使用前面设计的自适应卷积核
        v = self.conv(v.squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        # 最终，经过sigmoid 激活函数处理
        v = self.sigmoid(v)
        return x * v


class Tooth_multi_scale(nn.Module):
    """
    all feature map are sampling to 32*32*32, then concat in channel dimension
    finally, execute channel attention to all channels
    """

    def __init__(self):
        super(Tooth_multi_scale, self).__init__()
        self.input1_down = nn.MaxPool3d(kernel_size=2, stride=2)
        self.input2_out = nn.Identity()
        self.input3_up = nn.Upsample(scale_factor=2)
        self.input4_up = nn.Upsample(scale_factor=4)
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
    all feature map are sampling to 16*16*16, then concat in channel dimension
    finally, execute channel attention to all channels
    """

    def __init__(self):
        super(Bone_multi_scale, self).__init__()
        self.input1_down = nn.MaxPool3d(kernel_size=4, stride=4)
        self.input2_down = nn.MaxPool3d(kernel_size=2, stride=2)
        self.input3_out = nn.Identity()
        self.input4_up = nn.Upsample(scale_factor=2)
        self.channel_atten = ECABlock(channels=960)

    def forward(self, input1, input2, input3, input4):
        out1 = self.input1_down(input1)
        out2 = self.input2_down(input2)
        out3 = self.input3_out(input3)
        out4 = self.input4_up(input4)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        channel_atten_out = self.channel_atten(out)
        return channel_atten_out


class SpatialAttention(nn.Module):  # Spatial Attention Module
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)
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
        self.conv = nn.Conv3d(in_channels=960, out_channels=channel, kernel_size=1, stride=1)
        self.atten = SpatialAttention()

        if self.zoom == 'UP':
            self.k = input_feature_size / in_multi_size
            self.up = nn.Upsample(scale_factor=self.k)
        elif self.zoom == 'DOWN':
            self.avg = nn.AdaptiveAvgPool3d(self.input_feature_size)
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
        self.encoder = Encoder(in_channels=1, init_features=64)
        self.decoder = Decoder_block(num_classes=2, init_features=64)

    def forward(self, x):
        feat1, feat2, feat3, feat4, feat5 = self.encoder(x)
        out = self.decoder(feat1, feat2, feat3, feat4, feat5)
        return out


class input_enhancement(nn.Module):
    def __init__(self):
        super(input_enhancement, self).__init__()
        self.conv = nn.Conv3d(3, 1, kernel_size=1, stride=1, padding=0)
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
        self.encoder = Encoder()

        self.Tdecoder = Decoder_block()
        self.Bdecoder = Decoder_block()

        self.Tmulti = nn.ModuleList(
            [
                multi_scale_feature(zoom='UP', input_feature_size=64, in_multi_size=32, channel=64),
                multi_scale_feature(zoom='None', input_feature_size=32, in_multi_size=32, channel=128),
                multi_scale_feature(zoom='DOWN', input_feature_size=16, in_multi_size=32, channel=256),
                multi_scale_feature(zoom='DOWN', input_feature_size=8, in_multi_size=32, channel=512)
            ]
        )

        self.Bmulti = nn.ModuleList(
            [
                multi_scale_feature(zoom='UP', input_feature_size=64, in_multi_size=16, channel=64),
                multi_scale_feature(zoom='UP', input_feature_size=32, in_multi_size=16, channel=128),
                multi_scale_feature(zoom='None', input_feature_size=16, in_multi_size=16, channel=256),
                multi_scale_feature(zoom='DOWN', input_feature_size=8, in_multi_size=16, channel=512)
            ]
        )

        self.Tooth_multi_scale = Tooth_multi_scale()
        self.Bone_multi_scale = Bone_multi_scale()
        self.Tfinal = nn.Conv3d(2, 3, 1)  # background, WT, SM
        self.Bfinal = nn.Conv3d(2, 2, 1)  # background, AB

    def forward(self, x):
        feat1, feat2, feat3, feat4, feat5 = self.encoder(x)
        Tooth_multi = self.Tooth_multi_scale(feat1, feat2, feat3, feat4)  # (B, 960, 32, 32, 32)
        Tooth_feat1 = self.Tmulti[0](input_feature=feat1, in_multi=Tooth_multi)
        Tooth_feat2 = self.Tmulti[1](input_feature=feat2, in_multi=Tooth_multi)
        Tooth_feat3 = self.Tmulti[2](input_feature=feat3, in_multi=Tooth_multi)
        Tooth_feat4 = self.Tmulti[3](input_feature=feat4, in_multi=Tooth_multi)

        Tout1 = self.Tdecoder(Tooth_feat1, Tooth_feat2, Tooth_feat3, Tooth_feat4, feat5)
        out_tooth_last = self.Tfinal(Tout1)

        Bone_multi = self.Bone_multi_scale(feat1, feat2, feat3, feat4)  # (B, 960, 16, 16, 16)
        Bone_feat1 = self.Bmulti[0](input_feature=feat1, in_multi=Bone_multi)
        Bone_feat2 = self.Bmulti[1](input_feature=feat2, in_multi=Bone_multi)
        Bone_feat3 = self.Bmulti[2](input_feature=feat3, in_multi=Bone_multi)
        Bone_feat4 = self.Bmulti[3](input_feature=feat4, in_multi=Bone_multi)

        Bout1 = self.Bdecoder(Bone_feat1, Bone_feat2, Bone_feat3, Bone_feat4, feat5)
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


if __name__ == '__main__':
    a = torch.rand(size=(2, 1, 64, 64, 64))
    model = WTNet()
    Binary_out, out_tooth_last, out_bone_last = model(a)
    print(Binary_out.shape)
    print(out_tooth_last.shape)
    print(out_bone_last.shape)
    # print(feat4.shape)
    # print(feat5.shape)
    #
    # eca = SpatialAttention()
    # out = eca(feat1)
    # print(out.shape)
