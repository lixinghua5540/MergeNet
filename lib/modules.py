import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np 
import cv2
from torchvision.ops import DeformConv2d
device = torch.device("cuda:3")

class oneConv(nn.Module):
    # 卷积+ReLU函数
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_sizes, padding = paddings, dilation = dilations, bias=False),###, bias=False
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),#groups = in_channels
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class MFEblock(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(MFEblock, self).__init__()
        out_channels = in_channels
        # modules = []
        # modules.append(nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU()))
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1, bias=False),#groups = in_channels , bias=False
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.layer2 = ASPPConv(in_channels, out_channels, rate1)
        self.layer3 = ASPPConv(in_channels, out_channels, rate2)
        self.layer4 = ASPPConv(in_channels, out_channels, rate3)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),)
            #nn.Dropout(0.5))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim = 2)
        self.Sigmoid = nn.Sigmoid()

        self.conv0_s = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.conv1_s = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.conv2_s = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.conv3_s = nn.Conv2d(in_channels, in_channels // 4, 1)

        self.conv_squeeze = nn.Conv2d(2, 4, rate3*2+1, padding=rate3)
        self.conv_m = nn.Conv2d(in_channels//4, in_channels, 1)

    def forward(self, x):
        ### 多特征提取: Multi-Features Extraction block, MFEblock
        y0 = self.layer1(x)    # 第一个分支的输入只有x: (B,C,H,W)-->(B,C,H,W)
        y1 = self.layer2(y0+x) # 第二个分支的输入是y0和x: (B,C,H,W)-->(B,C,H,W)
        y2 = self.layer3(y1+x) # 第三个分支的输入是y1和x: (B,C,H,W)-->(B,C,H,W)
        y3 = self.layer4(y2+x) # 第四个分支的输入是y2和x: (B,C,H,W)-->(B,C,H,W)

        # 通过池化聚合全局信息,然后通过1×1conv建模通道相关性: (B,C,H,W)-->GAP-->(B,C,1,1)-->SE1-->(B,C,1,1)
        y0_weight = self.conv0_s(y0)
        y1_weight = self.conv1_s(y1)
        y2_weight = self.conv2_s(y2)
        y3_weight = self.conv0_s(y3)
        
        # # 将多个尺度的全局信息进行拼接: (B,C,4,1)
        attention = torch.cat([y0_weight,y1_weight,y2_weight,y3_weight], dim = 1)
        avg_attn = torch.mean(attention, dim=1, keepdim=True) # 应用全局平均池化: (B,C,H,W)-->(B,1,H,W)

        max_attn, _ = torch.max(attention, dim=1, keepdim=True) # 应用全局最大池化: (B,C,H,W)-->(B,1,H,W)
        agg = torch.cat([avg_attn, max_attn], dim=1)  # 将平均池化和最大池化特征进行拼接: (B,2,H,W)

        sig = self.conv_squeeze(agg).sigmoid()  # 将2个通道映射为N个通道, N是尺度的个数, 并通过sigmoid函数得到每个尺度对应的权重表示: (B,N,H,W), 在这里N==2
        
        attn = y0_weight * sig[:, 0, :, :].unsqueeze(1) + y1_weight * sig[:, 1, :, :].unsqueeze(1)\
            + y2_weight * sig[:, 2, :, :].unsqueeze(1) + y3_weight * sig[:, 3, :, :].unsqueeze(1) # 对多个尺度的信息进行加权求和: (B,C/2,H,W)
        attn = self.conv_m(attn) # 将通道恢复为原通道数量: (B,C/2,H,W)-->(B,C,H,W)
        
        return x * attn
    
#------------------------------------------------------------------------------#
class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
    
class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class EAM(nn.Module):
    def __init__(self):
        super(EAM, self).__init__()
        self.reduce1 = Conv1x1(64, 64)
        self.reduce4 = Conv1x1(2048, 256)
        #--------------------------------------------------------------#
        # PVTV2使用
        # self.reduce4 = Conv1x1(512, 256) # 原来是 Conv(2048, 256, 1)
        #--------------------------------------------------------------#
        #--------------------------------------------------------------#
        # # efficientnet使用
        # self.reduce4 = Conv1x1(320, 256) # 原来是 Conv(2048, 256, 1)
        #--------------------------------------------------------------#
        self.block = nn.Sequential(
            ConvBNR(256 + 64, 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1))

    def forward(self, x4, x1):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)
        return out
#------------------------------------------------------------------------------------------------------------------------------------# 

class GLDFF(nn.Module):
    def __init__(self, dim):
        super(GLDFF, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_atten = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_redu = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.nonlin = nn.Sigmoid()

    def forward(self, x, skip):
        output = torch.cat([x, skip], dim=1)
        
        att = self.conv_atten(self.avg_pool(output))
        output = output * att
        output = self.conv_redu(output)

        att = self.conv1(x) + self.conv2(skip)
        att = self.nonlin(att)

        output = output * att
        return output

def gauss_kernel(channels=3, cuda=True):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                            [4., 16., 24., 16., 4.],
                            [6., 24., 36., 24., 6.],
                            [4., 16., 24., 16., 4.],
                            [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    if cuda:
        kernel = kernel.to(device)
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def conv_gauss(img, kernel):
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    out = F.conv2d(img, kernel, groups=img.shape[1])
    return out

def upsample(x, channels):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels))

def make_laplace(img, channels):
    filtered = conv_gauss(img, gauss_kernel(channels))
    down = downsample(filtered)
    up = upsample(down, channels)
    if up.shape[2] != img.shape[2] or up.shape[3] != img.shape[3]:
        up = nn.functional.interpolate(up, size=(img.shape[2], img.shape[3]))
    diff = img - up
    return diff

def make_laplace_pyramid(img, level, channels):
    current = img
    pyr = []
    for _ in range(level):
        filtered = conv_gauss(current, gauss_kernel(channels))
        down = downsample(filtered)
        up = upsample(down, channels)
        if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
            up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
        diff = current - up
        pyr.append(diff)
        current = down
    pyr.append(current)
    return pyr

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
    def forward(self, x):
        avg_out = self.mlp(F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        max_out = self.mlp(F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        channel_att_sum = avg_out + max_out

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)
    def forward(self, x):
        x_compress = torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting 
        # print('scale',scale.shape)
        return x * scale, scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out, sa = self.SpatialGate(x_out)
        return x_out, sa

class EGA(nn.Module):
    def __init__(self, in_channels):
        super(EGA, self).__init__()

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3 , 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        
        self.cbam = CBAM(in_channels)

    def forward(self, edge_feature, x, pred):
        residual = x
        xsize = x.size()[2:]
        
        pred = torch.sigmoid(pred)
        
        #reverse attention 
        background_att = 1 - pred
        background_x= x * background_att
        
        #boudary attention
        edge_pred = make_laplace(pred, 1)  
        pred_feature = x * edge_pred

        #high-frequency feature
        edge_input = F.interpolate(edge_feature, size=xsize, mode='bilinear', align_corners=True)
        input_feature = x * edge_input

        fusion_feature = torch.cat([background_x, pred_feature, input_feature], dim=1)
        fusion_feature = self.fusion_conv(fusion_feature)

        attention_map = self.attention(fusion_feature)
        fusion_feature = fusion_feature * attention_map

        out = fusion_feature + residual
        out = self.cbam(out)
        return out

class BGM(nn.Module):
    global_count = 0
    def __init__(self, in_channels):
        super(BGM, self).__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3 , 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid())
        
        self.cbam = CBAM(in_channels)

        self.liner = nn.Linear(2,2)

    def forward(self, edge_feature, x, pred, edge_attention):
        residual = x
        xsize = x.size()[2:]
        
        pred = torch.sigmoid(pred)

        background_att = 1 - pred
        background_x= x * background_att
        
        edge_pred = make_laplace(pred, 1)

        concat_feat = torch.cat([edge_pred, edge_attention], dim=1)
        B, C, H, W = concat_feat.shape
        concat_feat = concat_feat.permute(0, 2, 3, 1).reshape(-1, C)
        weights = torch.softmax(self.liner(concat_feat), dim=1)
        weights = weights.view(B, H, W, 2).permute(0, 3, 1, 2)
        A_final = weights[:, 0:1] * edge_pred + weights[:, 1:2] * edge_attention
        pred_feature = x * A_final

        #high-frequency feature
        edge_input = F.interpolate(edge_feature, size=xsize, mode='bilinear', align_corners=True)
        input_feature = x * edge_input

        fusion_feature = torch.cat([background_x, pred_feature, input_feature], dim=1)
        fusion_feature = self.fusion_conv(fusion_feature)
        
        avg_pool = self.avgpool(fusion_feature)
        max_pool = self.maxpool(x)
        combined = avg_pool + max_pool
        
        attention_map = self.attention(combined)

        fusion_feature = fusion_feature * attention_map

        out = fusion_feature + residual
        out,sa = self.cbam(out) 

        return out, A_final, background_att
    
if __name__ == '__main__':
    x = torch.rand(1, 512, 16, 16)
    mfe = MFEblock(in_channels = 512,atrous_rates = [5,7,9])
    out = mfe(x)
    print(out.shape)




