import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

mask_folder_path = "label"
image_folder_path = "label_image"
mask_img_folder = "mask"

h5_folder_path = "slices"

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
    mask_path = os.path.join(mask_folder_path, image_list[i].replace('.jpg', '_Mask.json'))
    mask_img_path = os.path.join(mask_img_folder, image_list[i].replace('.jpg', '.png'))
    h5_path = os.path.join(h5_folder_path, image_list[i].replace('.jpg', '.h5'))

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = image.astype(np.float32)/255

    with open(mask_path, 'r') as f:
        data = json.load(f)

    shapes = data['shapes']
    imageHeight = data['imageHeight']
    imageWidth = data['imageWidth']

    mask = np.zeros((imageHeight, imageWidth), dtype=np.uint8)

    for i in range(len(shapes)):
        data0 = shapes[i]
        label = int(data0['label'])
        if label not in label_list:
            label_list.append(label)
    label_list.sort()

    for j in range(len(shapes)):
        data0 = shapes[j]
        label = int(data0['label'])
        idx = label_list.index(label)+1
        points = data0['points']
        # 按照points在imageHeight，imageWidth大小的图像上画出封闭多边形
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], color=idx)

    write_image_to_h5(image, mask, h5_path)
    cv2.imwrite(mask_img_path, mask)
# plt.imshow(mask)



