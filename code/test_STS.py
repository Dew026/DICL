#%%
import h5py
import numpy as np
import torch
# from networks.net_factory import net_factory

from networks.unet_icl import UNet_icl
from networks.unet import UNet

from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom

import matplotlib.pyplot as plt
# from networks.net_factory import net_factory

root_path = "../data/STS"
exp = 'STS/Inherent_Consistent_Learning'
argmodel = 'unet'
num_classes = 53
max_iterations = 30000
batch_size=8
deterministic=1
base_lr=0.01
patch_size=[256, 256]
seed=1337
labeled_num=3
num_tries='1'
labeled_bs=4

def net_factory(net_type="unet", in_chns=1, class_num=3):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "icl_unet":
        net = UNet_icl(in_chns=in_chns, class_num=class_num).cuda()
    return net

net = net_factory(net_type=argmodel, in_chns=1,class_num=num_classes)

snapshot_path = '/home/ubuntu/cxx_projects/ICL/experiments/STS/Inherent_Consistent_Learning_Fully_2000_labeled/unet_exp_1/model/model_iter_100000.pth'

net.load_state_dict(torch.load(snapshot_path))

#%%
import os
from tqdm import tqdm

val_folder = "/home/ubuntu/cxx_projects/ICL/data/STS/val"
val_list = os.listdir(val_folder)

json_folder = "/home/ubuntu/cxx_projects/ICL/data/STS/json_results"

if not os.path.exists(json_folder):
    os.mkdir(json_folder)

for i in tqdm(range(len(val_list))):
    # case_name = val_list[i]
    case_name = val_list[i].split('_')[2]+'_'+val_list[i].split('_')[3]
    json_name = case_name.replace(".h5", "_Mask.json")
    json_name = json_name[:11] + json_name[12:]

    json_path = os.path.join(json_folder, json_name)

    case = os.path.join(val_folder, val_list[i])

    h5f = h5py.File(case, 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]

    # prediction = np.zeros_like(label)
    # for ind in range(image.shape[0]):
    #     slice = image[ind, :, :]
    x, y = image.shape[0], image.shape[1]
    image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=0)
    input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
    net.eval()

    with torch.no_grad():
        out_main = net(input)
        out = torch.argmax(torch.softmax(
            out_main, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = zoom(out, (x / patch_size[0], y / patch_size[0]), order=0)

    # plt.subplot(211)
    # plt.imshow(label)
    # plt.subplot(212)
    # plt.imshow(pred)

    import json
    import cv2

    idx_list = [11,12,13,14,15,16,17,18,21,22,23,24,25,26,27,28,31,32,33,34,35,36,37,38,41,42,43,44,45,46,47,48,51,52,53,54,55,61,62,63,64,65,71,72,73,74,75,81,82,83,84,85]
    # 读取pred中所有像素值，并将其写为轮廓分割模式的json文件
    cv2.imwrite(case_name.replace(".h5", ".png"), pred)

    shapes = []
    for i in range(0,52):
        bin_image = (pred == i+1)

        # 寻找轮廓
        contours, _ = cv2.findContours(bin_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 将轮廓信息转换为字典
        for contour in contours:
            contour_dict = contour.squeeze().tolist()  # 将轮廓转换为列表

            shape = {"label":str(idx_list[i]),
                    "points":contour_dict}
            
            shapes.append(shape)

    result = {"shapes": shapes,
            "imageHeight":int(pred.shape[0]),
            "imageWidth":int(pred.shape[1]),}
    
    # plt.imshow(bin_image)
    # 将轮廓信息保存到 JSON 文件
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=4)


# %%
