import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

image_folder_path = "unlabel_image"
h5_folder_path = "volumes"

image_list = os.listdir(image_folder_path)

def write_image_to_h5(image, mask, h5_path, image_name='image', mask_name='label'):
    # 创建或打开 .h5 文件
    with h5py.File(h5_path, 'w') as f:
        # 将图像数据写入 .h5 文件
        f.create_dataset(image_name, data=image)
        f.create_dataset(mask_name, data=mask)

label_list = []
for i in tqdm(range(len(image_list))):
    image_path = os.path.join(image_folder_path, image_list[i])
    h5_path = os.path.join(h5_folder_path, image_list[i].replace('.jpg', '.h5'))

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = image.astype(np.float32)/255

    imageHeight, imageWidth = image.shape

    mask = np.zeros((imageHeight, imageWidth), dtype=np.uint8)

    write_image_to_h5(image, mask, h5_path)
# plt.imshow(mask)



