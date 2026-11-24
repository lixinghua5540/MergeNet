import numpy as np
from scipy.ndimage import label

def calculate_guc_gou_instance(true_mask, pred_mask):
    """
    基于实例级的全局欠分割（GUC）和过分割（GOU）计算。

    参数:
        true_mask (np.ndarray): 真实掩码，形状 [H, W]，二值，1 表示目标像素。
        pred_mask (np.ndarray): 预测掩码，形状 [H, W]，二值，1 表示目标像素。

    返回:
        GUC (float), GOU (float)
    """

    # 连通组件标记：区分不同实例
    true_labeled, num_true = label(true_mask)
    pred_labeled, num_pred = label(pred_mask)

    # 统计每个真实实例的像素集合
    true_instances = [set(zip(*np.where(true_labeled == i))) for i in range(1, num_true + 1)]
    # 统计每个预测实例的像素集合
    pred_instances = [set(zip(*np.where(pred_labeled == j))) for j in range(1, num_pred + 1)]

    # 计算每个真实实例与所有预测实例的最大交集比例
    guc_sum = 0
    for G in true_instances:
        if len(G) == 0:
            continue
        max_iou = 0
        for P in pred_instances:
            inter = len(G.intersection(P))
            iou = inter / len(G)  # 计算覆盖率
            if iou > max_iou:
                max_iou = iou
        guc_sum += max_iou
    GUC = 1 - (guc_sum / num_true) if num_true > 0 else 0

    # 计算每个预测实例与所有真实实例的最大交集比例
    gou_sum = 0
    for P in pred_instances:
        if len(P) == 0:
            continue
        max_iou = 0
        for G in true_instances:
            inter = len(P.intersection(G))
            iou = inter / len(P)  # 计算覆盖率
            if iou > max_iou:
                max_iou = iou
        gou_sum += max_iou
    GOU = 1 - (gou_sum / num_pred) if num_pred > 0 else 0

    return GUC, GOU