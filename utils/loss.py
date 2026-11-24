import torch.nn as nn
import torch.nn.functional as F
import torch
# from lib.modules import *
from torchvision.transforms.functional import rgb_to_grayscale
import imageio
device = torch.device("cuda:4")


#-------------------------------------------------------------------------------------------------#
class LossBinary:
    def __init__(self, jaccard_weight=0.0, class_weights=None, device=None):
        self.device = device
        if class_weights is not None:
            self.bce_weight = torch.tensor(class_weights, dtype=torch.float32)  # 类别0的权重为0.2，类别1的权重为0.8
        else:
            self.bce_weight = None

        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        """
        Args:
            outputs: (B, 1, H, W) - 模型输出的 logits（未经过 sigmoid）
            targets: (B, H, W) - 真实标签，0 或 1
        Returns:
            loss: 总损失
        """
        targets = targets.float().squeeze(dim=1)  # 目标值转换为 float
        
        # 计算 BCE Loss
        # weight_map = torch.where(targets == 1, 
        #                  self.bce_weight[1].to(device), 
        #                  self.bce_weight[0].to(device))
        bce_loss = nn.BCEWithLogitsLoss()
        loss = (1 - self.jaccard_weight) * bce_loss(outputs.squeeze(dim=1), targets)

        # 计算 Jaccard Loss（如果 jaccard_weight > 0）
        if self.jaccard_weight > 0:
            eps = 1e-7
            probs = torch.sigmoid(outputs.squeeze(1))  # 计算概率
            intersection = (probs * targets).sum(dim=(1, 2))  # 计算交集
            union = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))  # 计算并集
            
            jaccard_loss = 1 - (intersection + eps) / (union - intersection + eps)  # Jaccard Loss
            loss += self.jaccard_weight * jaccard_loss.mean()

        return loss
#-------------------------------------------------------------------------------------------------#

class EdgeAdaptiveLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.5, pos_weight=4.0, smooth=1e-5):
        """
        Args:
            alpha: Focal Loss的平衡因子（正负样本）
            gamma: Focal Loss的难易调节系数
            dice_weight: Dice loss 占比
            pos_weight: BCE中正类（边缘）权重
            smooth: 防止除零
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.pos_weight = torch.tensor([pos_weight])
        self.smooth = smooth

    def forward(self, outputs, targets):
        """
        outputs: logits, shape (B, 1, H, W)
        targets: binary mask, shape (B, 1, H, W) or (B, H, W)
        """
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        # Sigmoid & Focal Loss
        probs = torch.sigmoid(outputs)
        probs = probs.clamp(min=1e-4, max=1-1e-4)  # 避免数值不稳定

        bce = F.binary_cross_entropy(probs, targets, reduction='none')
        focal = self.alpha * (1 - probs) ** self.gamma * bce
        focal_loss = focal.mean()

        # Dice Loss
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1 - ((2. * intersection + self.smooth) / (union + self.smooth))
        dice_loss = dice_loss.mean()

        total_loss = (1 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss
        return total_loss

class EnhancedEdgeAdaptiveLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.5, pos_weight=4.0, 
                 edge_kernel_size=5, smooth=1e-5, dynamic_weight_adjust=False):
        """
        Args:
            alpha: Focal Loss的平衡因子（可扩展为每个样本的alpha）
            gamma: Focal Loss的难易调节系数
            dice_weight: Dice loss初始占比（若dynamic_weight_adjust=True则动态调整）
            pos_weight: BCE中正类（边缘）的全局权重
            edge_kernel_size: 边缘敏感度的卷积核大小（用于边缘增强）
            smooth: 防止除零的小常数
            dynamic_weight_adjust: 是否根据训练阶段动态调整dice_weight
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.base_dice_weight = dice_weight
        self.dice_weight = dice_weight
        self.register_buffer('pos_weight', torch.tensor(pos_weight)) 
        self.smooth = smooth
        self.edge_kernel = self._get_edge_kernel(edge_kernel_size)
        self.dynamic_weight_adjust = dynamic_weight_adjust
        self.register_buffer('edge_kernel_const', self.edge_kernel)

    def _get_edge_kernel(self, ksize):
        """生成边缘检测卷积核（Laplacian-like）"""
        kernel = torch.ones((1, 1, ksize, ksize), dtype=torch.float32) * -1
        center = ksize // 2
        kernel[:, :, center, center] = ksize**2 - 1
        return kernel / (ksize**2)

    def _calculate_edge_weights(self, targets):
        """计算边缘权重图（边缘像素获得更高权重）"""
        with torch.no_grad():
            edge_map = F.conv2d(targets, self.edge_kernel_const.to(targets.device), 
                               padding=self.edge_kernel_const.size(2)//2)
            edge_weights = 1 + torch.abs(edge_map)  # 边缘区域权重>1
        return edge_weights.detach()

    def forward(self, outputs, targets, current_epoch=None, max_epochs=None):
        """
        Args:
            outputs: 模型输出的logits (B, 1, H, W)
            targets: 二值边缘标注 (B, 1, H, W) 或 (B, H, W)
            current_epoch/max_epochs: 用于动态调整权重（可选）
        """
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        # === 动态权重调整 ===
        if self.dynamic_weight_adjust and current_epoch is not None:
            progress = current_epoch / max(max_epochs, 1)
            self.dice_weight = self.base_dice_weight * (1 - 0.5 * progress)  # 随训练逐渐降低Dice权重

        # === 边缘敏感权重 ===
        edge_weights = self._calculate_edge_weights(targets)

        # === 改进的Focal Loss ===
        probs = torch.sigmoid(outputs)
        probs = probs.clamp(self.smooth, 1 - self.smooth)  # 更严格的数值约束

        # 使用binary_cross_entropy_with_logits
        bce_loss = F.binary_cross_entropy_with_logits(
            outputs,  # 直接使用logits
            targets,
            weight=edge_weights,
            pos_weight=self.pos_weight,
            reduction='none'
        )
        
        # 动态alpha：对困难样本更敏感
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_term = (1 - pt) ** self.gamma
        focal_loss = (self.alpha * focal_term * bce_loss).mean()

        # === 改进的Dice Loss ===
        intersection = (probs * targets * edge_weights).sum(dim=(1, 2, 3))  # 边缘加权
        union = (probs * edge_weights).sum(dim=(1, 2, 3)) + (targets * edge_weights).sum(dim=(1, 2, 3))
        dice_loss = 1 - (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = dice_loss.mean()

        # === 组合损失 ===
        total_loss = (1 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss
        return total_loss

'''https://github.com/ntcongvn/CCBANet/blob/main/libraries/CCBANet/utils/loss.py'''

class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(weight=weight, size_average=size_average)

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        loss = self.bceloss(pred_flat, target_flat)

        return loss
    



"""Dice loss"""


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1

        size = pred.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, pred, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #pred = F.sigmoid(pred)       
        
        #flatten label and prediction tensors
        pred = pred.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (pred * targets).sum()
        total = (pred + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

"""BCE + DICE Loss"""


class BceDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss(weight, size_average)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = diceloss + bceloss

        return loss
    
class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, input, target):
        # binary_cross_entropy_with_logits 会自动对 input 做 sigmoid，不需要你手动做
        bce = F.binary_cross_entropy_with_logits(input, target)

        # 手动 sigmoid 用于 Dice 部分
        input = torch.sigmoid(input)

        # 保证 target 是 float 类型
        target = target.float()

        # Flatten
        input = input.view(input.size(0), -1)
        target = target.view(target.size(0), -1)

        intersection = (input * target).sum(1)
        total = input.sum(1) + target.sum(1)
        dice_score = (2 * intersection + self.smooth) / (total + self.smooth)

        # 限制 dice_score 的范围避免反向传播出问题
        dice_loss = 1 - dice_score.clamp(min=0.0, max=1.0)
        dice = dice_loss.mean()

        total_loss = 0.5 * bce + dice

        return total_loss




"""BCE + IoU Loss"""


class BceIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BceIoULoss, self).__init__()
        self.bce = BCELoss(weight, size_average)
        self.iou = IoULoss()

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        iouloss = self.iou(pred, target)

        loss = iouloss + bceloss

        return loss

""" Structure Loss: https://github.com/DengPingFan/PraNet/blob/master/MyTrain.py """
class StructureLoss(nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init__()

    def forward(self, pred, mask):
        weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        inter = ((pred * mask)*weit).sum(dim=(2, 3))
        union = ((pred + mask)*weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1)/(union - inter+1)
        return (wbce + wiou).mean()


""" Deep Supervision Loss"""
# loss for ours methods
class DeepSupervisionLoss(nn.Module):
    def __init__(self, typeloss="BceDiceLoss"):
        super(DeepSupervisionLoss, self).__init__()

        if typeloss=="BceDiceLoss":
            self.criterion = REAUNetLoss()
        elif typeloss=="BceIoULoss":
            self.criterion = BceIoULoss()
        elif typeloss=="StructureLoss":
            self.criterion = StructureLoss()
        else:
            raise Exception("Loss name is unvalid.")
        self.boundrayloss = LossBinary()

    def forward(self, pred, gt, boundary):
        d0, d1, d2, d3, d4, boundaries = pred[0:]
        loss0 = self.criterion(torch.sigmoid(d0), gt)
        lossb = self.boundrayloss(boundaries, boundary)
        gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
        loss1 = self.criterion(torch.sigmoid(d1), gt)
        gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
        loss2 = self.criterion(torch.sigmoid(d2), gt)
        gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
        loss3 = self.criterion(torch.sigmoid(d3), gt)
        gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
        loss4 = self.criterion(torch.sigmoid(d4), gt)

        return loss0 + loss1 + loss2 + loss3 + loss4 + lossb, loss0 + loss1 + loss2 + loss3 + loss4, lossb
        # return loss0 + loss1 + loss2 + loss3 + loss4, loss0 + loss1 + loss2 + loss3 + loss4, lossb
    """
    用于MCCANet
    """
    # def forward(self, pred, gt, boundary):
    #     d0, boundaries = pred[0:]
    #     loss0 = self.criterion(torch.sigmoid(d0), gt)
    #     lossb = self.boundrayloss(boundaries, boundary)

    #     return loss0 + lossb, loss0, lossb
    # 用于没有eem
    # def forward(self, pred, gt, boundary):
    #     d0, d1, d2, d3, d4 = pred[0:]
    #     loss0 = self.criterion(torch.sigmoid(d0), gt)
    #     gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    #     loss1 = self.criterion(torch.sigmoid(d1), gt)
    #     gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    #     loss2 = self.criterion(torch.sigmoid(d2), gt)
    #     gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    #     loss3 = self.criterion(torch.sigmoid(d3), gt)
    #     gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    #     loss4 = self.criterion(torch.sigmoid(d4), gt)

    #     return loss0 + loss1 + loss2 + loss3 + loss4
    """
    no deep supervision
    """
    # def forward(self, pred, gt, boundary):
    #     d0 = pred[0:]
    #     loss0 = self.criterion(torch.sigmoid(d0), gt)

    #     return loss0
    
# loss for reaunet
class REAUNetLoss(nn.Module):
    def __init__(self, frac=0.5, threshold=0.5, eps=1e-10):
        super().__init__()
        self.frac = frac
        self.threshold = threshold
        self.eps = eps

    def forward(self, output, target):
        # bce loss
        loss_bce = nn.BCELoss()(output, target)
        batch = output.shape[0]
        output = output.reshape(batch, -1)
        target = target.reshape(batch, -1)
        # dice loss
        output_t = torch.where(output > self.threshold, torch.ones_like(output), torch.zeros_like(output))
        inter = (output_t * target).sum(-1)
        union = (output_t + target).sum(-1)
        dice = (2 * inter + self.eps) / (union + self.eps)
        loss_dice = 1 - (dice.sum() / batch)
        return self.frac * loss_bce + loss_dice
    
# loss for unet
class UNetAndDeeplabV3Loss(nn.Module):
    def __init__(self):
         super(UNetAndDeeplabV3Loss,self).__init__()
         self.criterion = BceDiceLoss()
    def forward(self,x,gt):
        return self.criterion(torch.sigmoid(x), gt)

# loss for seanet
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice
def rcf_loss(inputs, label):

    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask > 0.5).float()).float() # ==1.
    num_negative = torch.sum((mask == 0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0.
   # inputs= torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='sum')(inputs.float(), label.float())

    return 1.*torch.sum(cost)
class RCFloss:
    def __init__(self):

        self.device = ''

    def __call__(self, outputs, targets):

        criterion = sum([rcf_loss(outpu, targets) for outpu in outputs])     #sum([rcf_loss(outpu, targets) for outpu in outputs])

        return criterion
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=3):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum
awl = AutomaticWeightedLoss(2)
class SEANetLoss(nn.Module):
    def __init__(self):
        super(SEANetLoss, self).__init__()
        self.loss3 = nn.MSELoss() #dist
        self.loss2 = RCFloss() #boundary 
        self.loss1 = BCEDiceLoss() #mask
    def forward(self,predicts,gts,boundaries,distances):
        loss1 = self.loss1(predicts[0],gts)
        loss2 = self.loss2(predicts[1],boundaries)
        loss3 = self.loss3(predicts[2],distances)
        loss = awl(loss1,loss3)
        loss = loss + loss2
        return loss


if __name__ == '__main__':
    gt = torch.rand(1,1,512,512)
    d0 = torch.rand(1,1,512,512)
    d1 = torch.rand(1,1,256,256)
    d2 = torch.rand(1,1,128,128)
    d3 = torch.rand(1,1,64,64)
    d4 = torch.rand(1,1,32,32)
    d5 = torch.rand(1,1,512,512)
    boundary = torch.rand(1,1,512,512)
    criteria_loss = DeepSupervisionLoss()
    lossall, lossm, lossb = criteria_loss([d0,d1,d2,d3,d4,d5], gt, boundary)
    print(lossm, lossb)


