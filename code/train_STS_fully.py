import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import BaseDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils import losses

root_path = "../data/STS"
exp = 'STS/Inherent_Consistent_Learning_Fully'
argmodel = 'unet'
num_classes = 53
max_iterations = 100000
batch_size=12
deterministic=1
base_lr=0.01
patch_size=[256, 256]
seed=1337
labeled_num=2000
num_tries='1'
labeled_bs=6

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
        
    elif "STS" in dataset:
        ref_dict = {"30": 30, "1000":1000, "2000":2000}

    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def train(snapshot_path):

    labeled_slice = patients_to_slices(root_path, labeled_num)

    model = net_factory(net_type=argmodel, in_chns=1, class_num=num_classes)

    db_train = BaseDataSets(base_dir=root_path, split="train", num=labeled_slice, transform=transforms.Compose([
        RandomGenerator(patch_size)
    ]))
    db_val = BaseDataSets(base_dir=root_path, split="val_test")

    def worker_init_fn(worker_id):
        random.seed(seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss = loss_dice + loss_ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('Info/lr', lr_, iter_num)
            writer.add_scalar('Loss/loss', loss, iter_num)
            writer.add_scalar('Loss/loss_ce', loss_ce, iter_num)
            writer.add_scalar('Loss/loss_dice', loss_dice, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))


            if iter_num > 0 and iter_num % 5000 == 0:
                save_name = 'model_iter_{}.pth'.format(iter_num)
                save_path = os.path.join(snapshot_path+'/model', save_name)
                torch.save(model.state_dict(), save_path)
                
            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    snapshot_path = "../experiments/{}_{}_labeled/{}_exp_{}".format(
        exp, labeled_num, argmodel, num_tries)
    if not os.path.exists(snapshot_path+'/model'):
        os.makedirs(snapshot_path+'/model')

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    train(snapshot_path)
