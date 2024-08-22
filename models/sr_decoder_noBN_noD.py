import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class DynamicConvLayer(nn.Module):
    def __init__(self, out_channels):
        super(DynamicConvLayer, self).__init__()
        self.out_channels = out_channels
        self.conv1x1 = None  # 初始时不创建卷积层

    def forward(self, x):
        # 根据x的通道数动态创建1x1卷积层
        if self.conv1x1 is None:
            in_channels = x.shape[1]  # 提取输入特征图的通道数
            self.conv1x1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, bias=False).to(x.device)

        return self.conv1x1(x)

##############原
# class Decoder(nn.Module):
#     def __init__(self, c1, c2):
#         super(Decoder, self).__init__()
#
#         self.conv1 = nn.Conv2d(c1, c1//2, 1, bias=False)
#         self.conv2 = nn.Conv2d(c2, c2//2, 1, bias=False)
#         self.relu = nn.ReLU()
#         self.last_conv = nn.Sequential(nn.Conv2d((c1+c2)//2, 256, kernel_size=3, stride=1, padding=1, bias=False),
#                                        nn.ReLU(),
#                                        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
#                                        nn.ReLU(),
#                                        nn.Conv2d(128, 64, kernel_size=1, stride=1))
#         self._init_weight()
#
#     def forward(self, x, low_level_feat, factor):
#         low_level_feat = self.conv1(low_level_feat)
#         low_level_feat = self.relu(low_level_feat)
#
#         x = self.conv2(x)
#         x = self.relu(x)
#
#         x = F.interpolate(x, size=[i*(factor//2) for i in low_level_feat.size()[2:]], mode='bilinear', align_corners=True)
#         if factor>1:
#             low_level_feat = F.interpolate(low_level_feat, size=[i*(factor//2) for i in low_level_feat.size()[2:]], mode='bilinear', align_corners=True)
#         x = torch.cat((x, low_level_feat), dim=1) #直接超分
#
#         x = self.last_conv(x)
#
#         return x
#
#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, SynchronizedBatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
###########################

#################################切块超分，简单线性插值和反卷积
# class Decoder(nn.Module):
#     def __init__(self, c1, c2):
#         super(Decoder, self).__init__()
#
#         self.conv1 = nn.Conv2d(c1, c1//2, 1, bias=False)
#         self.conv2 = nn.Conv2d(c2, c2//2, 1, bias=False)
#         self.relu = nn.ReLU()
#         self.last_conv = nn.Sequential(nn.Conv2d((c1+c2)//2, 256, kernel_size=3, stride=1, padding=1, bias=False),
#                                        nn.ReLU(),
#                                        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
#                                        nn.ReLU(),
#                                        nn.Conv2d(128, 64, kernel_size=1, stride=1))
#         self._init_weight()
#         # self.sr = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
#         self.upconv = nn.ConvTranspose2d(c1 // 2, c1 // 2, kernel_size=2, stride=2)
#
#     def forward(self, x, low_level_feat, factor):
#         low_level_feat = self.conv1(low_level_feat)
#         low_level_feat = self.relu(low_level_feat)
#
#         height_chunks = torch.chunk(low_level_feat, chunks=2, dim=2)  # 沿着高度方向切成4块
#         sr_chunks = []
#         for chunk in height_chunks:
#             width_chunks = torch.chunk(chunk, chunks=2, dim=3)  # 沿着宽度方向切成两块
#             sr_chunks.extend([self.upconv(c) for c in width_chunks])  # 对每个块应用上采样
#         sr_low_level_feat = torch.cat((torch.cat((sr_chunks[0], sr_chunks[1]), dim=3),
#                                        torch.cat((sr_chunks[2], sr_chunks[3]), dim=3)), dim=2)
#
#         x = self.conv2(x)
#         x = self.relu(x)
#
#         # print(x.shape) #无像素重排torch.size([1,256,16,16]) #像素重排torch.size([1,256,32,32])
#         # print(sr_low_level_feat.shape) #torch.size([1,256,128,128])
#
#         x = F.interpolate(x, size=[i * (factor // 2) for i in sr_low_level_feat.size()[2:]], mode='bilinear',
#                           align_corners=True)
#         # print(x.shape) #torch.size([1,256,128,128])
#         x = torch.cat((x, sr_low_level_feat), dim=1) #切4块反卷积
#
#         x = F.interpolate(x, scale_factor=0.5, mode='bicubic', align_corners=False)
#         x = self.last_conv(x)
#
#         return x
#
#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, SynchronizedBatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
############################

#############################子像素重排+反卷积（上采样）
# class Decoder(nn.Module):
#     def __init__(self, c1, c2):
#         super(Decoder, self).__init__()
#
#         self.conv1 = nn.Conv2d(c1, c1//2, 1, bias=False)
#         self.conv2 = nn.Conv2d(c2, c2//2, 1, bias=False)
#         self.relu = nn.ReLU()
#         self.last_conv = nn.Sequential(nn.Conv2d((c1+c2)//2, 256, kernel_size=3, stride=1, padding=1, bias=False),
#                                        nn.ReLU(),
#                                        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
#                                        nn.ReLU(),
#                                        nn.Conv2d(128, 64, kernel_size=1, stride=1))
#         self._init_weight()
#         self.sr = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.upconv = nn.ConvTranspose2d(c1 // 2, c1 // 2, kernel_size=2, stride=2)
#         # 增加的卷积层用于扩展通道数，为 PixelShuffle 准备
#         self.upconv1 = nn.Conv2d(c2 // 2, (c2 // 2) * 16, 1, bias=False)
#         self.pixel_shuffle1 = nn.PixelShuffle(4)
#         self.upconv3 = nn.ConvTranspose2d(c2 // 2, c2 // 2, kernel_size=2, stride=2)
#
#         # 如果有多个 F.interpolate，对于每个操作重复上面的步骤
#         # 注意：可能需要根据合并后的通道数调整这里的通道数
#         self.upconv2 = nn.Conv2d(c1 // 2 + c2 // 2, (c1 // 2 + c2 // 2) * 4, 1, bias=False)
#         self.pixel_shuffle2 = nn.PixelShuffle(2)
#
#
#     def forward(self, x, low_level_feat, factor):
#         low_level_feat = self.conv1(low_level_feat)
#         low_level_feat = self.relu(low_level_feat)
#
#         height_chunks = torch.chunk(low_level_feat, chunks=2, dim=2)  # 沿着高度方向切成4块
#         sr_chunks = []
#         for chunk in height_chunks:
#             width_chunks = torch.chunk(chunk, chunks=2, dim=3)  # 沿着宽度方向切成两块
#             sr_chunks.extend([self.upconv(c) for c in width_chunks])  # 对每个块应用上采样
#         sr_low_level_feat = torch.cat((torch.cat((sr_chunks[0], sr_chunks[1]), dim=3),
#                                        torch.cat((sr_chunks[2], sr_chunks[3]), dim=3)), dim=2)
#
#
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.upconv1(x)
#         # print(x.shape)
#         x = self.pixel_shuffle1(x)
#         # print(x.shape)
#         # print(sr_low_level_feat.shape)
#         x = self.upconv3(x)
#         # print(x.shape)
#
#         x = torch.cat((x, sr_low_level_feat), dim=1)
#
#         # print(x.shape)
#         # 融合 x 和 low_level_feat，然后进行第二次上采样
#         # x = self.upconv2(x)
#         #
#         # # print(x.shape)
#         # x = self.pixel_shuffle2(x)
#         # print(x.shape)  #torch.Size([1, 320, 256, 256]) 但应该是torch.Size([1, 320, 64, 64])
#
#         x = F.interpolate(x, scale_factor=0.5, mode='bicubic', align_corners=False)
#         # print(x.shape) #torch.Size([1, 320, 64, 64])
#
#         x = self.last_conv(x)
#
#         return x

# ############upsample+subpixelconv
# class Decoder(nn.Module):
#     def __init__(self, c1, c2):
#         super(Decoder, self).__init__()
#
#         self.conv1 = nn.Conv2d(c1, c1 // 2, 1, bias=False)
#         self.conv2 = nn.Conv2d(c2, c2 // 2, 1, bias=False)
#         self.relu = nn.ReLU()
#         self.last_conv = nn.Sequential(
#             nn.Conv2d((c1 + c2) // 2, 256, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(128, 64, kernel_size=1, stride=1))
#         self._init_weight()
#         self.sr = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
#         # self.upconv = nn.ConvTranspose2d(c1 // 2, c1 // 2, kernel_size=2, stride=2)
#         # 增加的卷积层用于扩展通道数，为 PixelShuffle 准备
#         self.upconv1 = nn.Conv2d(c2 // 2, (c2 // 2) * 16, 1, bias=False)
#         self.pixel_shuffle1 = nn.PixelShuffle(4)
#
#     def forward(self, x, low_level_feat, factor):
#         low_level_feat = self.conv1(low_level_feat)
#         low_level_feat = self.relu(low_level_feat)
#
#         height_chunks = torch.chunk(low_level_feat, chunks=2, dim=2)  # 沿着高度方向切成4块
#         sr_chunks = []
#         for chunk in height_chunks:
#             width_chunks = torch.chunk(chunk, chunks=2, dim=3)  # 沿着宽度方向切成两块
#             sr_chunks.extend([self.sr(c) for c in width_chunks])  # 对每个块应用上采样
#         sr_low_level_feat = torch.cat((torch.cat((sr_chunks[0], sr_chunks[1]), dim=3),
#                                        torch.cat((sr_chunks[2], sr_chunks[3]), dim=3)), dim=2)
#
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.upconv1(x)
#         # print(x.shape)
#         x = self.pixel_shuffle1(x)
#         # print(x.shape)
#         # print(sr_low_level_feat.shape)
#         x = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
#         # print(x.shape)
#
#         x = torch.cat((x, sr_low_level_feat), dim=1)
#         x = F.interpolate(x, scale_factor=0.5, mode='bicubic', align_corners=False)
#         # print(x.shape) #torch.Size([1, 320, 64, 64])
#
#         x = self.last_conv(x)
#
#         return x

##################消融###########
class Decoder(nn.Module):
    def __init__(self, c1, c2):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(c1, c1 // 2, 1, bias=False)
        self.conv2 = nn.Conv2d(c2, c2 // 2, 1, bias=False)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(
            nn.Conv2d((c1 + c2) // 2, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1, stride=1))
        self._init_weight()
        self.sr = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upconv = nn.ConvTranspose2d(c1 // 2, c1 // 2, kernel_size=2, stride=2)
        # 增加的卷积层用于扩展通道数，为 PixelShuffle 准备
        self.upconv1 = nn.Conv2d(c2 // 2, (c2 // 2) * 16, 1, bias=False)
        self.pixel_shuffle1 = nn.PixelShuffle(4)
        self.upconv3 = nn.ConvTranspose2d(c2 // 2, c2 // 2, kernel_size=2, stride=2)

        # 如果有多个 F.interpolate，对于每个操作重复上面的步骤
        # 注意：可能需要根据合并后的通道数调整这里的通道数
        # self.upconv2 = nn.Conv2d(c1 // 2 + c2 // 2, (c1 // 2 + c2 // 2) * 4, 1, bias=False)
        # self.pixel_shuffle2 = nn.PixelShuffle(2)

    def forward(self, x, low_level_feat, factor):
        # print(low_level_feat.shape)
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        # low_level_feat = self.sr(low_level_feat)
        # print(low_level_feat.shape)

        height_chunks = torch.chunk(low_level_feat, chunks=2, dim=2)  # 沿着高度方向切成4块
        sr_chunks = []
        for chunk in height_chunks:
            width_chunks = torch.chunk(chunk, chunks=2, dim=3)  # 沿着宽度方向切成两块
            sr_chunks.extend([self.upconv(c) for c in width_chunks])  # 对每个块应用上采样
        sr_low_level_feat = torch.cat((torch.cat((sr_chunks[0], sr_chunks[1]), dim=3),
                                       torch.cat((sr_chunks[2], sr_chunks[3]), dim=3)), dim=2)

        # print(sr_low_level_feat.shape)
        x = self.conv2(x)
        x = self.relu(x)
        # print(x.shape)
        x = F.interpolate(x, scale_factor=8, mode='bicubic', align_corners=False)
        # x = self.upconv1(x)
        # print(x.shape)
        # x = self.pixel_shuffle1(x)
        # # print(x.shape)
        # # print(sr_low_level_feat.shape)
        # x = self.upconv3(x)
        # # print(x.shape)

        x = torch.cat((x, sr_low_level_feat), dim=1)

        # print(x.shape)
        # 融合 x 和 low_level_feat，然后进行第二次上采样
        # x = self.upconv2(x)
        #
        # # print(x.shape)
        # x = self.pixel_shuffle2(x)
        # print(x.shape)  #torch.Size([1, 320, 256, 256]) 但应该是torch.Size([1, 320, 64, 64])

        x = F.interpolate(x, scale_factor=0.5, mode='bicubic', align_corners=False)
        # print(x.shape) #torch.Size([1, 320, 64, 64])

        x = self.last_conv(x)

        return x
################################################


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
#############################