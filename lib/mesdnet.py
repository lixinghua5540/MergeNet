import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms.functional import rgb_to_grayscale
from lib.res2net import res2net50_v1b_26w_4s
from lib.modules import *
from lib.HBGNet.pvtv2 import pvt_v2_b2
from efficientnet_pytorch import EfficientNet


class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            Conv(in_channels, in_channels // 4), 
            Conv(in_channels // 4, out_channels)

        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)
        return self.up(out)

class Out(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Out, self).__init__()
        self.conv1 = Conv(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

#Multi-Scale Edge-Guided Attention Network
class EGANet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(EGANet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # --- 修改 2: 在 EGANet.__init__ 中替换 ---
        # 替换原来的 resnet 定义
        # self.encoder = pvt_v2_b2()

        # # 加载你的预训练权重
        # checkpoint = torch.load('/home/czh/sudujieshu/MEGANet/HBGNet/pvt_v2_b2.pth', map_location='cpu')
        # if 'model' in checkpoint: checkpoint = checkpoint['model']
        # self.encoder.load_state_dict(checkpoint, strict=False)
        # # --- 修改 3: 修改 dem_1 的输入通道 (关键！) ---
        # # PVT-B2 的 Stage 4 输出是 512, Stage 3 是 320, Stage 2 是 128, Stage 1 是 64
        # self.x5_dem_1 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        # self.x4_dem_1 = nn.Sequential(nn.Conv2d(320, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        # self.x3_dem_1 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        # self.x2_dem_1 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        # Encoder
        # #----------------------------------------------------------------------------------------#
        resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.encoder1_conv = resnet.conv1
        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4
        #----------------------------------------------------------------------------------------#
        # resnext = models.resnext50_32x4d(pretrained=True) 
        # self.encoder1_conv = resnext.conv1 # 输出通道 64
        # self.encoder1_bn = resnext.bn1
        # self.encoder1_relu = resnext.relu
        # self.maxpool = resnext.maxpool
        # self.encoder2 = resnext.layer1    # 输出通道 256
        # self.encoder3 = resnext.layer2    # 输出通道 512
        # self.encoder4 = resnext.layer3    # 输出通道 1024
        # self.encoder5 = resnext.layer4    # 输出通道 2048
        #----------------------------------------------------------------------------------------#
        #----------------------------------------------------------------------------------------#
        # # 1. 加载 EfficientNet-B0
        # self.encoder = EfficientNet.from_pretrained('efficientnet-b0')
        
        # # 2. 适配 DEM 模块的输入通道 (根据 B0 的 Reduction 层调整)
        # # B0 通道分布: reduction_5=320, r4=112, r3=40, r2=24, r1=16
        # self.x5_dem_1 = nn.Sequential(nn.Conv2d(320, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        # self.x4_dem_1 = nn.Sequential(nn.Conv2d(112, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        # self.x3_dem_1 = nn.Sequential(nn.Conv2d(40, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        # self.x2_dem_1 = nn.Sequential(nn.Conv2d(24, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        
        # # 3. 适配 e1 (用于 BEM/EAM 模块)
        # self.e1_bottleneck = nn.Sequential(nn.Conv2d(16, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        #----------------------------------------------------------------------------------------#
        # res2net和resnext时使用
        self.x5_dem_1 = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.x4_dem_1 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.x3_dem_1 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.x2_dem_1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        #----------------------------------------------------------------------------------------#
        
        self.up5 = nn.Sequential(
            Conv(512, 512),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        ) 
        # self.up4 = Up(768, 256)
        # self.up3 = Up(384, 128)
        # self.up2 = Up(192, 64)
        # self.up1 = Up(128, 64)
        self.up4 = nn.Sequential(
            Conv(256, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.up3 = nn.Sequential(
            Conv(128, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.up2 = nn.Sequential(
            Conv(64, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.up1 = nn.Sequential(
            Conv(64, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        # self.bem = BEM(64,512)
        # self.sobel_x1, self.sobel_y1 = get_sobel(64, 1)
        # self.sobel_x4, self.sobel_y4 = get_sobel(512, 1)
        self.bem = EAM()

        self.bgm1 = BGM(64)
        self.bgm2 = BGM(64)
        self.bgm3 = BGM(128)
        self.bgm4 = BGM(256)
        
        self.out5 = Out(512, n_classes)
        self.out4 = Out(256, n_classes)
        self.out3 = Out(128, n_classes)
        self.out2 = Out(64, n_classes)
        self.out1 = Out(64, n_classes)

        self.gcmsff5 = MFEblock(in_channels = 512,atrous_rates = [5,7,9])
        self.gcmsff4 = MFEblock(in_channels = 256,atrous_rates = [5,7,9])
        self.gcmsff3 = MFEblock(in_channels = 128,atrous_rates = [3,5,7])
        self.gcmsff2 = MFEblock(in_channels = 64,atrous_rates = [1,2,3])
        self.gcmsff1 = MFEblock(in_channels = 64,atrous_rates = [1,2,3])
        # # test different rates
        # self.gcmsff5 = MFEblock(in_channels = 512,atrous_rates = [1,2,3])
        # self.gcmsff4 = MFEblock(in_channels = 256,atrous_rates = [1,2,3])
        # self.gcmsff3 = MFEblock(in_channels = 128,atrous_rates = [1,2,3])
        # self.gcmsff2 = MFEblock(in_channels = 64,atrous_rates = [1,2,3])
        # self.gcmsff1 = MFEblock(in_channels = 64,atrous_rates = [1,2,3])

        # self.gcmsff5 = MFEblock(in_channels = 512,atrous_rates = [5,7,9])
        # self.gcmsff4 = MFEblock(in_channels = 256,atrous_rates = [5,7,9])
        # self.gcmsff3 = MFEblock(in_channels = 128,atrous_rates = [5,7,9])
        # self.gcmsff2 = MFEblock(in_channels = 64,atrous_rates = [5,7,9])
        # self.gcmsff1 = MFEblock(in_channels = 64,atrous_rates = [5,7,9])

        # self.gcmsff5 = MFEblock(in_channels = 512,atrous_rates = [3,5,7])
        # self.gcmsff4 = MFEblock(in_channels = 256,atrous_rates = [3,5,7])
        # self.gcmsff3 = MFEblock(in_channels = 128,atrous_rates = [3,5,7])
        # self.gcmsff2 = MFEblock(in_channels = 64,atrous_rates = [3,5,7])
        # self.gcmsff1 = MFEblock(in_channels = 64,atrous_rates = [3,5,7])

        self.gldff4 = GLDFF(256)
        self.gldff3 = GLDFF(128)
        self.gldff2 = GLDFF(64)
        self.gldff1 = GLDFF(64)

        self.conv_skip4 = nn.Conv2d(512, 256, kernel_size=1, stride=1) 
        self.conv_skip3 = nn.Conv2d(256, 128, kernel_size=1, stride=1) 
        self.conv_skip2 = nn.Conv2d(128, 64, kernel_size=1, stride=1)
        self.conv_skip1 = nn.Conv2d(64, 64, kernel_size=1, stride=1) 

        self.gate_conv4 = nn.Sequential(
                            nn.Conv2d(256, 128, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(128, 1, kernel_size=1),
                            nn.Sigmoid()
                        )
        self.gate_conv3 = nn.Sequential(
                            nn.Conv2d(128, 64, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 1, kernel_size=1),
                            nn.Sigmoid()
                        )
        self.gate_conv2 = nn.Sequential(
                            nn.Conv2d(64, 32, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(32, 1, kernel_size=1),
                            nn.Sigmoid()
                        )
        self.gate_conv1 = nn.Sequential(
                            nn.Conv2d(64, 32, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(32, 1, kernel_size=1),
                            nn.Sigmoid()
                        )




    def forward(self, x):
        grayscale_img = rgb_to_grayscale(x)
        edge_feature = make_laplace_pyramid(grayscale_img, 5, 1)
        shuchu = edge_feature[0]
        edge_feature = edge_feature[1]

        attn_np = shuchu.squeeze().cpu().detach().numpy()  # [256, 256]
        attn_np = (attn_np * 255).astype(np.uint8)  # 归一化为 0-255

        #Encoder
        #-----------------------------------------------------------------------------------------------#
        e1 = self.encoder1_conv(x)  
        e1 = self.encoder1_bn(e1)
        e1 = self.encoder1_relu(e1)
        e1_pool = self.maxpool(e1)  
        
        e2 = self.encoder2(e1_pool)

        e3 = self.encoder3(e2)  
        
        e4 = self.encoder4(e3)  

        e5 = self.encoder5(e4)  
        #-----------------------------------------------------------------------------------------------#
        # # --- 修改 4: Encoder 获取方式 ---
        # pvt_outs = self.encoder(x) # 返回 [Stage1, Stage2, Stage3, Stage4]
        # e2 = pvt_outs[0]  # H/4, 64 ch
        # e3 = pvt_outs[1]  # H/8, 128 ch
        # e4 = pvt_outs[2]  # H/16, 320 ch
        # e5 = pvt_outs[3]  # H/32, 512 ch
        # # --- 修改 5: 构造 e1 (用于 BEM 模块) ---
        # # 因为 PVT 最小只到 H/4，这里通过上采样模拟 ResNet 原本的 H/2 层 (e1)
        # e1 = F.interpolate(e2, scale_factor=2, mode='bilinear', align_corners=True) # H/2
        #-----------------------------------------------------------------------------------------------#
        #-----------------------------------------------------------------------------------------------#
        # # 提取各个阶段的特征
        # endpoints = self.encoder.extract_endpoints(x)
        
        # e1_raw = endpoints['reduction_1'] # H/2, 16ch
        # e2_raw = endpoints['reduction_2'] # H/4, 24ch
        # e3_raw = endpoints['reduction_3'] # H/8, 40ch
        # e4_raw = endpoints['reduction_4'] # H/16, 112ch
        # e5_raw = endpoints['reduction_5'] # H/32, 320ch

        # # 映射到统一维度
        # e1 = self.e1_bottleneck(e1_raw)
        # e2 = e2_raw # 后续在 x2_dem_1 中处理
        # e3 = e3_raw # 后续在 x3_dem_1 中处理
        # e4 = e4_raw # 后续在 x4_dem_1 中处理
        # e5 = e5_raw # 后续在 x5_dem_1 中处理
        #-----------------------------------------------------------------------------------------------#
        e5_dem_1 = self.x5_dem_1(e5)
        e4_dem_1 = self.x4_dem_1(e4)
        e3_dem_1 = self.x3_dem_1(e3)
        e2_dem_1 = self.x2_dem_1(e2)

        #Decoder
        # e5_dem_1 e4_dem_1 e3_dem_1 e2_dem_1 e1
        # e5_dem_1 e1 联合计算 boundary loss
        gc5 = self.gcmsff5(e5_dem_1) # H/32 W/32 512
        d5 = self.up5(gc5) # H/16 W/16 512
        out5 = self.out5(d5) # H/16 W/16 1 --> loss5

        # s1 = run_sobel(self.sobel_x1, self.sobel_y1, e1)
        # s4 = run_sobel(self.sobel_x4, self.sobel_y4, e5_dem_1)
        edge = self.bem(e5,e1) # H/2 W/2 1 --> boundary loss
        # boundary = self.bem(e1, gc5) # H W 1 --> boundary loss
        edge_attention = torch.sigmoid(edge) # H/2 W/2 1
        boundaryforloss = F.interpolate(edge, scale_factor=2, mode='bilinear', align_corners=False) # H W 1

        # 对boundary_attention降采样以满足输入BGM模块
        edge_attention1 = edge_attention
        edge_attention2 = F.interpolate(edge_attention1, scale_factor=0.5, mode='bilinear', align_corners=False) # H W 1
        edge_attention3 = F.interpolate(edge_attention2, scale_factor=0.5, mode='bilinear', align_corners=False)
        edge_attention4 = F.interpolate(edge_attention3, scale_factor=0.5, mode='bilinear', align_corners=False)

        skip4 = self.conv_skip4(d5) # H/16 W/16 256
        gl4 = self.gldff4(e4_dem_1, skip4) # H/16 W/16 256
        bgm4, A_final4, Background4 = self.bgm4(edge_feature, gl4, out5, edge_attention4) # H/16 W/16 256
        fusion4 = skip4 + bgm4 # H/16 W/16 256
        gc4 = self.gcmsff4(fusion4) # H/16 W/16 256
        d4 = self.up4(gc4) # H/8 W/8 256
        out4 = self.out4(d4) # H/8 W/8 1 --> loss4

        skip3 = self.conv_skip3(d4) # H/8 W/8 128
        gl3 = self.gldff3(e3_dem_1, skip3) # H/8 W/8 128
        bgm3, A_final3, Background3 = self.bgm3(edge_feature, gl3, out4, edge_attention3) # H/8 W/8 128
        fusion3 = skip3 + bgm3 # H/8 W/8 128
        gc3 = self.gcmsff3(fusion3) # H/8 W/8 128
        d3 = self.up3(gc3) # H/4 W/4 128
        out3 = self.out3(d3) # H/4 W/4 1 --> loss3

        skip2 = self.conv_skip2(d3) # H/4 W/4 64
        gl2 = self.gldff2(e2_dem_1, skip2) # H/4 W/4 64
        bgm2, A_final2, Background2 = self.bgm2(edge_feature, gl2, out3, edge_attention2) # H/4 W/4 64
        fusion2 = skip2 + bgm2 # H/4 W/4 64
        gc2 = self.gcmsff2(fusion2) # H/4 W/4 64
        d2 = self.up2(gc2) # H/2 W/2 64
        out2 = self.out2(d2) # H/2 W/2 1 --> loss2

        skip1 = self.conv_skip1(d2) # H/2 W/2 64
        gl1 = self.gldff1(e1, skip1) # H/2 W/2 64
        bgm1, A_final1, Background1 = self.bgm1(edge_feature, gl1, out2, edge_attention1) # H/2 W/2 64
        fusion1 = skip1 + bgm1 # H/2 W/2 64
        gc1 = self.gcmsff1(fusion1) # H/2 W/2 64
        d1 = self.up1(gc1) # H W 64
        out1 = self.out1(d1) # H W 1 --> loss1
        
        return out1, out2, out3, out4, out5, boundaryforloss

class EGANetModel(nn.Module): 
    def __init__(self, n_channels=3, n_classes=1):
        super(EGANetModel,self).__init__()
        self.channel = n_channels
        self.num_classes = n_classes
        self.net = EGANet(self.channel, self.num_classes)

    def forward(self, images):
        out1, out2, out3, out4, out5, boundary = self.net(images)
        return out1, out2, out3, out4, out5, boundary

if __name__ == '__main__':
    # (B,C,H,W)
    x = torch.rand(1, 3, 512, 512).to(device="cuda:1")
    # atrous_rates: 扩张率
    Model = EGANetModel(3,1).to(device="cuda:1")
    out1, out2, out3, out4, out5, boundary = Model(x)
    print(boundary.shape)
