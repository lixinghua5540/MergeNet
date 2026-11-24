import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from lib.DBBANet.model import DBBANet
from lib.Deeplabv3_plus.deeplabv3 import DeepLab
from lib.UNet import UNetWithResnet50Encoder
from lib.MCCANet.model import MCCA
from lib.CMTFNet.model import CMTFNet
from lib.ourslgdffbgm import EGANetModel
from lib.SEANet import SEANet
from lib.BsiNet import BsiNet
from lib.HBGNet.model import Field
from lib.BFINet.model import BFINet
from lib.REAUNet import REAUNet
import imageio
import torch.nn as nn
from utils.dataloader import test_dataset
import matplotlib.pyplot as plt
import cv2
from utils.metrics import Evaluator
from tqdm import tqdm
from utils.gucgou import calculate_guc_gou_instance

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=512, help='testing size')
parser.add_argument('--pth_path', type=str, default='your_pth_path')

# data_path = '/Cropland_Datasets/IFLYTEK2021/test/imgs(test)/'
# save_path = '/Cropland_Datasets/IFLYTEK2021/test/gts(test)/'
opt = parser.parse_args()
device = torch.device("cuda:3")
# model = BFINet('/MEGANet/BFINet/pvt_v2_b2.pth')
model = EGANetModel()
state_dict = torch.load(opt.pth_path, map_location=device)  # 加载时直接指定设备
model.load_state_dict(state_dict['model'])
# model.load_state_dict(state_dict)
model.to(device)
model.eval()

Eva_test = Evaluator(2)

import torch
import numpy as np
import cv2

def remove_border_edges(edge_map, border=3):
    """
    将边界处的边缘像素清零
    """
    edge_map[:border, :] = 0         # 顶部
    edge_map[-border:, :] = 0        # 底部
    edge_map[:, :border] = 0         # 左侧
    edge_map[:, -border:] = 0        # 右侧
    return edge_map

def visualize_confusion(prediction, target, save_path='confusion_map.png'):
    """
    修改版：绘制 TP/TN/FP/FN 的可视化图像，并返回像素统计值。
    """
    if prediction.ndim == 3:
        prediction = prediction.squeeze(0)
    if target.ndim == 3:
        target = target.squeeze(0)

    prediction = prediction.astype(np.uint8)
    target = target.astype(np.uint8)

    h, w = target.shape
    total_pix = h * w
    vis_image = np.zeros((h, w, 3), dtype=np.uint8)

    # 计算四种情况
    tp = (prediction == 1) & (target == 1)
    tn = (prediction == 0) & (target == 0)
    fp = (prediction == 1) & (target == 0) # 误分
    fn = (prediction == 0) & (target == 1) # 漏分

    # 统计像素数量
    fp_count = np.sum(fp)
    fn_count = np.sum(fn)

    # 画图 (OpenCV 颜色通道是 BGR)
    vis_image[tp] = [255, 255, 255]   # 白色
    vis_image[tn] = [0, 0, 0]         # 黑色
    vis_image[fp] = [0, 0, 255]       # 红色 (误分)
    vis_image[fn] = [0, 255, 0]       # 绿色 (漏分)

    # cv2.imwrite(save_path, vis_image)
    
    return fp_count, fn_count, total_pix

def compute_binary_iou(prediction, gt):
    intersection = np.logical_and(prediction == 1, gt == 1).sum()  # TP
    union = np.logical_or(prediction == 1, gt == 1).sum()  # TP + FP + FN
    iou = intersection / (union + 1e-7)  # 避免除 0
    return iou

def calculate_gtc_guc_gou_batch(true_masks, pred_masks):
    true_masks = np.asarray(true_masks)
    pred_masks = np.asarray(pred_masks)
    if true_masks.ndim != 3 or pred_masks.ndim != 3:
        raise ValueError("输入形状必须是 [B, H, W]")
    batch_size = true_masks.shape[0]
    GTC_list = 0.
    GUC_list = 0.
    GOU_list = 0.
    for i in range(batch_size):
        true_mask = true_masks[i]
        pred_mask = pred_masks[i]
        GTC, GUC, GOU = calculate_gtc_guc_gou(true_mask, pred_mask)
        GTC_list += GTC
        GUC_list += GUC
        GOU_list += GOU
    return GTC_list, GUC_list, GOU_list

def calculate_gtc_guc_gou(true_mask, pred_mask):
    true_mask = (true_mask > 0).astype(np.int32)
    pred_mask = (pred_mask > 0).astype(np.int32)
    intersection = np.logical_and(true_mask, pred_mask).sum()
    true_area = np.sum(true_mask)
    pred_area = np.sum(pred_mask)
    GUC = max(0, (true_area - intersection) / true_area) if true_area > 0 else 0
    GOU = max(0, (pred_area - intersection) / pred_area) if pred_area > 0 else 0
    GTC = (GUC + GOU) / 2
    return GTC, GUC, GOU

from scipy.spatial.distance import cdist
from skimage import measure

def average_contour_distance(gt_mask, pred_mask, max_dist=512):
    gt_mask = gt_mask.squeeze()
    pred_mask = pred_mask.squeeze()
    if np.sum(gt_mask) == 0 and np.sum(pred_mask) == 0:
        return 0.0
    elif np.sum(gt_mask) == 0 or np.sum(pred_mask) == 0:
        return max_dist
    gt_contours = measure.find_contours(gt_mask, 0.5)
    pred_contours = measure.find_contours(pred_mask, 0.5)
    if len(gt_contours) == 0 or len(pred_contours) == 0:
        return max_dist
    gt_pts = np.concatenate(gt_contours, axis=0)
    pred_pts = np.concatenate(pred_contours, axis=0)
    d1 = cdist(pred_pts, gt_pts).min(axis=1).mean()
    d2 = cdist(gt_pts, pred_pts).min(axis=1).mean()
    return (d1 + d2) / 2

from skimage.segmentation import find_boundaries

def compute_fbdy(gt_mask, pred_mask):
    gt_mask = gt_mask.squeeze()
    pred_mask = pred_mask.squeeze()
    gt_boundary = find_boundaries(gt_mask, mode='thick').astype(np.uint8)
    pred_boundary = find_boundaries(pred_mask, mode='thick').astype(np.uint8)
    gt_sum = np.sum(gt_boundary)
    pred_sum = np.sum(pred_boundary)
    if gt_sum == 0 and pred_sum == 0:
        return 1.0
    if gt_sum == 0 or pred_sum == 0:
        return 0.0
    intersection = np.logical_and(gt_boundary, pred_boundary)
    intersection_sum = np.sum(intersection)
    com = intersection_sum / gt_sum
    corr = intersection_sum / pred_sum
    if com + corr == 0:
        return 0.0
    fbdy = 2 * com * corr / (com + corr)
    return fbdy

def compute_structural_errors(gt_mask, pred_mask):
    gt_mask = gt_mask.squeeze()
    pred_mask = pred_mask.squeeze()
    intersection = np.logical_and(gt_mask, pred_mask).astype(np.uint8)
    gt_area = np.sum(gt_mask)
    pred_area = np.sum(pred_mask)
    inter_area = np.sum(intersection)
    if gt_area == 0:
        Sover = 0.0 if pred_area == 0 else 1.0
    else:
        Sover = 1.0 - (inter_area / gt_area)
    if pred_area == 0:
        Sunder = 0.0 if gt_area == 0 else 1.0
    else:
        Sunder = 1.0 - (inter_area / pred_area)
    gt_boundary = find_boundaries(gt_mask, mode='thick').astype(np.uint8)
    pred_boundary = find_boundaries(pred_boundary if 'pred_boundary' in locals() else pred_mask, mode='thick').astype(np.uint8)
    boundary_intersection = np.logical_and(gt_boundary, pred_boundary).astype(np.uint8)
    gt_bnd_area = np.sum(gt_boundary)
    bnd_inter_area = np.sum(boundary_intersection)
    if gt_bnd_area == 0:
        Lbe = 0.0 if np.sum(pred_boundary) == 0 else 1.0
    else:
        Lbe = 1.0 - (bnd_inter_area / gt_bnd_area)
    return Sover, Sunder, Lbe

# 路径设置
image_root = '/Cropland_Datasets/FGFD/val/imgs/'
gt_root = '/Cropland_Datasets/FGFD/val/gts/'
test_loader = test_dataset(image_root, gt_root, 512)

# --- 初始化统计变量 ---
GTC, GUC, GOU, sover, sunder, lbe = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
total_fp_pixels = 0
total_fn_pixels = 0
total_all_pixels = 0

for i in tqdm(range(test_loader.size)):
    image, gt, name = test_loader.load_data()
    image = image.to(device)
    predicts = model(image)
    pred = torch.sigmoid(predicts[0]).data.cpu().numpy()
    
    prediction = (pred > 0.5).astype(np.int32)
    target = gt.numpy()
    
    # 指标计算 (保留原逻辑)
    GTC_list, GUC_list, GOU_list = calculate_gtc_guc_gou_batch(target, prediction.squeeze(1))
    sover_list, sunder_list, lbe_list = compute_structural_errors(target, prediction.squeeze(1))
    GUC += GUC_list
    GOU += GOU_list
    sover += sover_list
    sunder += sunder_list
    lbe += lbe_list
    Eva_test.add_batch(target, prediction.squeeze(1))
    
    # 可视化与误分/漏分占比统计
    save_path = 'your_save_path'+ name.replace('image', 'mask').replace('tif', 'png')
    
    # 获取单张图的 FP, FN 像素数
    fp_c, fn_c, p_total = visualize_confusion(prediction.squeeze(1), target, save_path=save_path)
    
    # 累加到全局
    total_fp_pixels += fp_c
    total_fn_pixels += fn_c
    total_all_pixels += p_total

# 最终打印
F1 = Eva_test.F1Score()
IOU = Eva_test.Intersection_over_Union()
OA = Eva_test.OA()

# -------------------- 修改后的打印部分 --------------------
# 假设你的 Evaluator 返回的是 [背景, 前景] 的数组
# 我们通常关注前景（类别1）的指标
f1_val = F1[1] if isinstance(F1, np.ndarray) else F1
iou_val = IOU[1] if isinstance(IOU, np.ndarray) else IOU
oa_val = OA # OA 通常本身就是单数值标量

# 计算百分比
total_all_pixels = max(total_all_pixels, 1e-7) # 防止除零
fp_rate = (total_fp_pixels / total_all_pixels) * 100
fn_rate = (total_fn_pixels / total_all_pixels) * 100

print("\n" + "*"*50)
# 使用提取后的标量值进行格式化打印
print(f"基础指标: F1={float(f1_val):.4f}, IOU={float(iou_val):.4f}, OA={float(oa_val):.4f}")
print(f"结构误差: sover={sover/test_loader.size:.4f}, sunder={sunder/test_loader.size:.4f}, lbe={lbe/test_loader.size:.4f}")
print("-" * 50)
print(f"全局误分率 (False Positive Rate): {fp_rate:.2f}% (把背景误认为目标)")
print(f"全局漏分率 (False Negative Rate): {fn_rate:.2f}% (未识别出目标)")
print(IOU[0])
print("*"*50)