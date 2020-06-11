import os 
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np


def aug():

    images = os.listdir('data/dataset3/road_car_images')
    anno = os.listdir('data/dataset3/road_car_anno')
    for image_name in images:

        random_angle = np.random.randint(30, 180)
        # 数据增强
        # 不能加概率  否则image和anno加强的不一样
        transform = transforms.Compose([
            transforms.Lambda(lambda image: image.transpose(Image.FLIP_TOP_BOTTOM)),
            transforms.Lambda(lambda image: image.transpose(Image.FLIP_LEFT_RIGHT)),
            transforms.Lambda(lambda image: image.rotate(random_angle, Image.NEAREST))
        ]) 
        
        image = Image.open('data/dataset3/road_car_images/' + image_name)
        anno = Image.open('data/dataset3/road_car_anno/' + image_name)
        image = transform(image)
        anno = transform(anno)
        image.save('data/dataset3/road_car_images_aug/' + 'aug_'+ image_name)
        print(image_name+'：image保存成功')
        anno.save('data/dataset3/road_car_anno_aug/' + 'aug_' + image_name)
        print(image_name+'：anno保存成功')

if __name__ == "__main__":
    aug()
