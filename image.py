import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2
import torch
from torchvision import datasets, transforms


def load_data(img_path, train=True, unlabeled=False, need_pseudo_label=False, pseudo_label_path=None):
    if unlabeled:
        img = Image.open(img_path).convert('RGB')
        target = 0
        if need_pseudo_label:
            img_name = os.path.basename(img_path)
            target = torch.load(os.path.join(pseudo_label_path, img_name.replace('.jpg', '.pth')))
            img, target = data_augumentation(img, target)

    else:
        img = Image.open(img_path).convert('RGB')
        gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
        gt_file = h5py.File(gt_path)
        target = np.asarray(gt_file['density'])
        target = cv2.resize(target, (target.shape[1]//8, target.shape[0]//8), interpolation=cv2.INTER_CUBIC)*64

    return img, target


def data_augumentation(img, target):
    if random.random() < 0.8:
        img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
    img = transforms.RandomGrayscale(p=0.2)(img)
    img = blur(img, p=0.5)
    img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
    return img, target


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img

# if False:
        #     crop_size = (img.size[0]/2,img.size[1]/2)
        #     if random.randint(0,9)<= -1:
        #
        #
        #         dx = int(random.randint(0,1)*img.size[0]*1./2)
        #         dy = int(random.randint(0,1)*img.size[1]*1./2)
        #     else:
        #         dx = int(random.random()*img.size[0]*1./2)
        #         dy = int(random.random()*img.size[1]*1./2)
        #
        #
        #
        #     img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        #     target = target[dy:crop_size[1]+dy,dx:crop_size[0]+dx]
        #
        #
        #
        #
        #     if random.random()>0.8:
        #         target = np.fliplr(target)
        #         img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #