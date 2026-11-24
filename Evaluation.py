import numpy as np
import os
import imageio

def compute_metrics(pred, gt):
    """
    计算 IoU, F1-Score (Dice), Recall
    :param pred: 预测的二值分割结果 (H, W)，0/1
    :param gt: 真实的二值分割掩码 (H, W)，0/1
    :return: IoU, F1-Score, Recall
    """
    intersection = np.logical_and(pred, gt).sum()  # 交集 |S ∩ G|
    union = np.logical_or(pred, gt).sum()  # 并集 |S ∪ G|
    pred_size = pred.sum()  # 预测区域大小 |S|
    gt_size = gt.sum()  # 真实区域大小 |G|
    
    iou = intersection / (union + 1e-8)  # 避免除 0
    f1 = 2 * intersection / (pred_size + gt_size + 1e-8)  # Dice 系数
    recall = intersection / (gt_size + 1e-8)  # 召回率

    return iou, f1, recall

# # 示例
# gt = np.array([[0,1,1,0], [0,1,1,0], [0,0,0,0], [0,0,0,0]])  # 真实掩码
# pred = np.array([[0,1,1,0], [0,1,0,0], [0,0,0,0], [0,0,0,0]])  # 预测掩码

# iou, dice, recall = compute_metrics(pred, gt)
# print(f"IoU: {iou:.4f}, F1-Score: {dice:.4f}, Recall: {recall:.4f}")
path1 = '/home/czh/sudujieshu/MEGANet/MEGANet-Res2Net/predictions/'
path2 = '/home/czh/sudujieshu/Cropland_Datasets/IFLYTEK2021/H512W512/test/gts/'
predictions = os.listdir(path1)
gts = os.listdir(path2)
iou_all, f1_all, recall_all = 0,0,0
for p,g in zip(predictions,gts):
    pred = imageio.imread(os.path.join(path1,p))
    gt = imageio.imread(os.path.join(path2,g))
    pred = (pred > 128).astype(np.uint8)
    gt = (gt > 128).astype(np.uint8)
    iou, f1, recall = compute_metrics(pred,gt)
    iou_all+=iou
    f1_all+=f1
    recall_all+=recall
print(f"Batch IoU: {iou_all/len(gts):.2f}, Batch F1-Score: {f1_all/len(gts):.2f}, Batch Recall: {recall_all/len(gts):.2f}")
