import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import numbers
import numpy as np 
from PIL import Image
import torch
import cv2

palette = [[0,0,0],[128,0,0],[0,128,0]]

"""
自定义的转为onehot编码
"""
def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)  # 单通道有索引使用
        # class_map = equality.astype(int)  #单通道无索引使用
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    semantic_map = np.argmax(semantic_map, axis=-1)
    # semantic_map[semantic_map == 1] = 0
    semantic_map[semantic_map == 2] = 1
    return (semantic_map*255).astype(np.int32)

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask, boundary, distance):
        if random.random() < self.p:
            img = F.vflip(img)
            mask = F.vflip(mask)
            boundary = F.vflip(boundary)
            distance = F.vflip(distance)
        return img, mask, boundary, distance

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask, boundary, distance):
        if random.random() < self.p:
            img = F.hflip(img)
            mask = F.hflip(mask)
            boundary = F.hflip(boundary)
            distance = F.hflip(distance)
        return img, mask, boundary, distance

class RandomRotate(object):
    def __init__(self, degrees, p=0.5, resample=False, expand=False, center=None):
        if isinstance(degrees,numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center
        self.p = p

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img, mask, boundary, distance):
        if random.random() < self.p:
            angle = self.get_params(self.degrees)
            return F.rotate(img, angle, self.resample, self.expand, self.center), F.rotate(mask, angle, self.resample, self.expand, self.center), F.rotate(boundary, angle, self.resample, self.expand, self.center), F.rotate(distance, angle, self.resample, self.expand, self.center)
        return img, mask, boundary, distance

class PolypDataset(Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, trainsize, augmentations=True):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.tif') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.boundaries = [gt_root.replace('gts', 'boundaries') + f for f in os.listdir(gt_root.replace('gts', 'boundaries')) if f.endswith('.tif') or f.endswith('.png')]
        self.distances = [gt_root.replace('gts', 'distances') + f for f in os.listdir(gt_root.replace('gts', 'distances')) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.boundaries = sorted(self.boundaries)
        self.distances = sorted(self.distances)
        self.filter_files()
        self.size = len(self.images)
        self.augmentations = augmentations
        self.resize = transforms.Resize((self.trainsize, self.trainsize))
        self.rotate = RandomRotate(degrees=(0, 360))
        self.vf = RandomVerticalFlip(p=0.5)
        self.hf = RandomHorizontalFlip(p=0.5)
        # self.img_transform = transforms.Compose([
        #     transforms.GaussianBlur((25, 25), sigma=(0.001, 2.0)),
        #     transforms.ColorJitter(
        #         brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406],
        #                         [0.229, 0.224, 0.225])])
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.boundary_transform = transforms.ToTensor()
        self.distance_transform = transforms.ToTensor()


    def __getitem__(self, index):
        # image = self.rgb_loader(self.images[index])
        # gt = self.binary_loader(self.gts[index])

        # image = self.resize(image)
        # gt = self.resize(gt)

        """
        符合FGFD数据集数据处理模式
        """
        image = Image.open(self.images[index]).convert('RGB')
        gt = np.array(Image.open(self.gts[index]).convert('RGB'), dtype=np.uint8)
        gt = mask_to_onehot(gt, palette)
        gt = Image.fromarray(np.uint8(gt))
        boundary = np.array(Image.open(self.boundaries[index]).convert('L'), dtype=np.uint8)
        boundary = Image.fromarray(np.uint8(boundary))  # 恢复为 [0, 255] 的图像
        distance = np.array(Image.open(self.distances[index]).convert('L'), dtype=np.uint8)
        distance = Image.fromarray(np.uint8(distance))

        if self.augmentations == True:
            image, gt, boundary, distance = self.rotate(image, gt, boundary, distance)
            image, gt, boundary, distance = self.vf(image, gt, boundary, distance)
            image, gt, boundary, distance = self.hf(image, gt, boundary, distance)
            image = self.img_transform(image)
            gt = self.gt_transform(gt)
            boundary = self.boundary_transform(boundary)
            distance = self.distance_transform(distance)
            # print(gt.shape,boundary.shape)
            # cv2.imwrite('/home/czh/sudujieshu/Cropland_Datasets/IFLYTEK2021/train/gggg.png',np.array(gt.squeeze(dim=0))*255)
            # cv2.imwrite('/home/czh/sudujieshu/Cropland_Datasets/IFLYTEK2021/train/bbbb.png',np.array(boundary.squeeze(dim=0))*255)
            # distance = torch.from_numpy(np.expand_dims(distance, 0)).float()
        else:
            image = self.img_transform(image)
            gt = self.gt_transform(gt)
            boundary = self.boundary_transform(boundary)
            distance = torch.from_numpy(np.expand_dims(distance, 0)).float
        return image, gt, boundary, distance

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=False, augmentation=True):
# drop_last 仅限最后一个余1的情况下才使用
    dataset = PolypDataset(image_root, gt_root, trainsize, augmentation)
    data_loader = DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last=True)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.tif') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.boundaries = [gt_root.replace('gts', 'boundaries') + f for f in os.listdir(gt_root.replace('gts', 'boundaries')) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.boundaries = sorted(self.boundaries)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.boundary_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        """
        符合FGFD数据集数据处理模式
        """
        image = Image.open(self.images[self.index]).convert('RGB')
        image = self.transform(image).unsqueeze(0)
        # image = torch.from_numpy(np.array(image) / 255.).float().permute(2, 0, 1).unsqueeze(0)
        gt =np.array(Image.open(self.gts[self.index]).convert('RGB'), dtype=np.uint8)
        gt = mask_to_onehot(gt, palette)
        gt = Image.fromarray(np.uint8(gt))  
        gt = self.gt_transform(gt)  
        # gt = torch.from_numpy(np.array(gt) / 255.).float().unsqueeze(0)
        boundary = np.array(Image.open(self.boundaries[self.index]).convert('L'), dtype=np.uint8)
        boundary = Image.fromarray(np.uint8(boundary))  # 恢复为 [0, 255] 的图像
        boundary = self.boundary_transform(boundary)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        # print(name)
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
        

if __name__ == '__main__':
     dataset = PolypDataset('/Cropland_Datasets/IFLYTEK2021/train/imgs/train/', '/Cropland_Datasets/IFLYTEK2021/train/gts/', trainsize=512)
     img, gt, boundary, _ = dataset[0]
     print(boundary.shape)
