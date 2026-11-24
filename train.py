import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import os
import argparse
from lib.mesdnet import EGANetModel
from lib.CMTFNet.model import CMTFNet
from lib.MCCANet.model import MCCA
from lib.DBBANet.model import DBBANet
from lib.Deeplabv3_plus.deeplabv3 import DeepLab
from lib.UNet import UNetWithResnet50Encoder
from lib.SEANet import SEANet
from utils.dataloader import PolypDataset, get_loader, test_dataset
from utils.loss import DeepSupervisionLoss, REAUNetLoss, UNetAndDeeplabV3Loss, SEANetLoss
from utils.utils import AvgMeter, clip_gradient
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils.metrics import Evaluator
from tqdm import tqdm
import wandb
from skimage.segmentation import find_boundaries

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

def train(train_loader, model, optimizer, epoch, criteria_loss):
    global best_iou
    global best_epoch
    Eva_train = Evaluator(2)
    Eva_test = Evaluator(2)
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record = AvgMeter()
    lossrecordmask = AvgMeter()
    lossrecordedge = AvgMeter()
    best = 0
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        # ---- data prepare ----
        images, gts, boundaries, distances = pack
        images = Variable(images).to(device)
        gts = Variable(gts).to(device)
        boundaries = Variable(boundaries).to(device)
        distances = Variable(distances).to(device)
            # trainsize = int(round(opt.trainsize*rate/32)*32)
            # if rate != 1:
            #     images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            #     gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            #     boundaries = F.interpolate(boundaries, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            #     distances = F.interpolate(distances, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
        
        predicts = model(images)
        if opt.model=='ours':
            loss, lossm, lossb = criteria_loss(predicts, gts, boundaries)
            # loss = criteria_loss(predicts, gts, boundaries)
        if opt.model=='dbbanet':
            loss = criteria_loss(torch.sigmoid(predicts), gts)
        if opt.model=='unet':
            loss = criteria_loss(predicts, gts)
        if opt.model=='cmtfnet':
            loss = criteria_loss(predicts, gts)
        if opt.model=='mccanet':
            loss, lossm, lossb = criteria_loss(predicts, gts, boundaries)
        if opt.model=='deeplabv3plus':
            loss =  criteria_loss(predicts, gts)
        if opt.model=='seanet':
            loss = criteria_loss(predicts, gts, boundaries, distances)

        writer.add_scalar("Loss/train", loss, epoch)

        loss.backward()
        clip_gradient(optimizer, opt.grad_norm)
        optimizer.step()

        loss_record.update(loss.data, opt.batchsize)
        lossrecordmask.update(lossm.data, opt.batchsize)
        lossrecordedge.update(lossb.data, opt.batchsize)
       
        # Add batch sample into evaluator
        # if rate != 1:
        pred = torch.sigmoid(predicts[0]).data.cpu().numpy()
        # pred = torch.sigmoid(predicts).data.cpu().numpy()
        prediction = (pred > 0.5).astype(np.int32)
        target = gts.cpu().numpy()
        Eva_train.add_batch(target.squeeze(), prediction.squeeze())
        if i % 20 == 0 or i == total_step:
            print("{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:.4f}, lossmask: {:.4f}, lossedge: {:.4f}". format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show(),lossrecordmask.show(),lossrecordedge.show()))
            # print("{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:.4f}". format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))  
    # 实时上传到 W&B (Step级别)
    wandb.log({
        "train/loss": loss_record.show(),
        "train/loss_mask": lossrecordmask.show(),
        "train/loss_edge": lossrecordedge.show(),
        "train/lr": optimizer.param_groups[0]['lr']
    })
    F1 = Eva_train.F1Score()
    IOU = Eva_train.Intersection_over_Union()
    print(
        'Epoch [%d/%d], \n[Training] IOU0: %.4f, IOU1: %.4f, F1: %.4f' % (
            epoch, opt.epoch, \
            IOU[0],IOU[1],F1))

    # save_path = 'charpter3_loss_1m_1b/{}'.format('JEGMAMFMshunxu.pth')
    save_path = 'rate_357/dilation_{}.pth'.format(epoch)

    model.train(False)
    model.eval()
    test_loader = test_dataset(image_test_root, gt_test_root, 512)
    # 初始化统计量
    total_sover, total_sunder, total_lbe = 0.0, 0.0, 0.0
    for i in tqdm(range(test_loader.size)):
        image, gt, name = test_loader.load_data()
        image = image.to(device)
        predicts = model(image)
        pred = torch.sigmoid(predicts[0]).data.cpu().numpy()
        # pred = torch.sigmoid(predicts).data.cpu().numpy()
        prediction = (pred > 0.5).astype(np.int32)
        target = gt.numpy()

        Eva_test.add_batch(target, prediction.squeeze(1))
        # 2. 计算结构指标 (Sover, Sunder, Lbe)
        s_o, s_u, l_e = compute_structural_errors(target, prediction)
        total_sover += s_o
        total_sunder += s_u
        total_lbe += l_e
    # 计算平均值
    num_test = test_loader.size
    avg_sover, avg_sunder, avg_lbe = total_sover/num_test, total_sunder/num_test, total_lbe/num_test
    F1 = Eva_test.F1Score()
    IOU = Eva_test.Intersection_over_Union()
    new_iou = IOU[1]
    # ---- 统一上传到 W&B ----
    wandb.log({
        "epoch": epoch,
        "test/IOU_crop": new_iou,
        "test/IOU_back": IOU[0],
        "test/F1_crop": F1,
        "test/Sover": avg_sover,
        "test/Sunder": avg_sunder,
        "test/Lbe": avg_lbe
    })
    torch.save(model.state_dict(), save_path)
    if new_iou >= best_iou:
        best_iou = new_iou
        best_epoch = epoch
        best_net = model.state_dict()
        torch.save(model.state_dict(), 'rate_357/dilation_best.pth')
        print('Best %s Model Iou[0] :%.4f; Model Iou[1] :%.4f; F1 : %.4f; Best epoch : %d' % ('ours', IOU[0], IOU[1], F1, best_epoch))  
        wandb.run.summary["best_iou"] = best_iou
        wandb.run.summary["best_epoch"] = best_epoch

if __name__ == '__main__':

    writer = SummaryWriter()

    if torch.cuda.is_available():
        device = torch.device("cuda:7")
    else:
        device = torch.device("cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default = 200, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--grad_norm', type=float, default=0.5, help='gradient clipping norm')
    parser.add_argument('--batchsize', type=int,
                        default = 4, help='training batch size')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--mt', type=float, default=0.9)
    parser.add_argument('--power', type=float, default=0.9)
    parser.add_argument('--trainsize', type=int,
                        default=512, help='training dataset size')
    parser.add_argument('--train_path', type=str,
                        default='/home/czh/sudujieshu/Cropland_Datasets/IFLYTEK2021/train/imgs/train', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='baselineplusssem')
    parser.add_argument("--mgpu", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--model", type=str, default="ours", help='model name to use')
    opt = parser.parse_args()

    # 初始化 W&B
    wandb.init(
        project="Dilation", # 这里填你的项目名
        name=f"rate_357", # 本次实验的小名
        config=opt  # 自动记录你的 lr, batchsize, model 等参数
    )

    # choose your model (add if 命令)s]
    if opt.model=='ours':
        model = EGANetModel()
    if opt.model=='dbbanet':
        model = DBBANet()
    if opt.model=='mccanet':
        model = MCCA()
    if opt.model=='cmtfnet':
        model = CMTFNet()
    if opt.model=='unet':
        model = UNetWithResnet50Encoder()
    if opt.model=='deeplabv3plus':
        model = DeepLab()
    if opt.model=='seanet':
        model = SEANet()
        
    if opt.mgpu == "true" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    
    params = model.parameters()

    optimizer = torch.optim.Adam(params, lr=opt.lr, weight_decay=opt.weight_decay)

    lr_lambda = lambda epoch: 1.0 - pow((epoch / opt.epoch), opt.power)
    scheduler = LambdaLR(optimizer, lr_lambda)
    if opt.model == 'ours':
        criteria_loss = DeepSupervisionLoss()
    if opt.model == 'reaunet':
        criteria_loss = REAUNetLoss()
    if opt.model=='dbbanet':
        criteria_loss = REAUNetLoss()
    if opt.model == 'mccanet':
        criteria_loss = DeepSupervisionLoss()
    if opt.model=='cmtfnet':
        criteria_loss = UNetAndDeeplabV3Loss()
    if opt.model=='unet':
        criteria_loss = UNetAndDeeplabV3Loss()
    if opt.model=='deeplabv3plus':
        criteria_loss = UNetAndDeeplabV3Loss()
    if opt.model=='seanet':
        criteria_loss = SEANetLoss()


    image_root = '/Cropland_Datasets/FGFD/train/imgs/'
    gt_root = '/Cropland_Datasets/FGFD/train/gts/'

    image_test_root = '/Cropland_Datasets/FGFD/test/imgs/'
    gt_test_root = '/Cropland_Datasets/FGFD/test/gts/'
    
    # image_root = '/Cropland_Datasets/IFLYTEK2021/train/imgs/train/'
    # gt_root = '/Cropland_Datasets/IFLYTEK2021/train/gts/'

    # image_test_root = '/Cropland_Datasets/IFLYTEK2021/test/imgs(test)/'
    # gt_test_root = '/Cropland_Datasets/IFLYTEK2021/test/gts(test)/'

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print(torch.cuda.get_device_name(0))
    print("#"*20, "Start Training", "#"*20)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    best_iou = 0
    best_epoch = 0
    for epoch in range(0, opt.epoch):
        train(train_loader, model, optimizer, epoch, criteria_loss)
        scheduler.step()
        
    writer.close()
