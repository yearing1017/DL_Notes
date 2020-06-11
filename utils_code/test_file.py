import os
from PIL import Image
import cv2
images = os.listdir('data/dataset3/images_prepped_train')


for image_name in images:
    if not os.path.exists('data/dataset3/anno_prepped_train/' + image_name):
        print(image_name)

print("over")
