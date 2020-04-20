import os
import scipy.misc
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
from torch import optim


class AverageMeter():
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self,val , n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum/self.count

def weights_init(module):
    class_name = module.__class__.__name__
    if 'Conv' in class_name:
        module.weight.data.normal_(mean = 0.0, std=0.02)
    elif 'BatchNorm' in class_name:
        module.weight.data.normal_(mean=1.0,std=0.02)
        module.bias.data.fill_(0)
    return

def anomaly_score(G,D,z,x,__lambda__= 0.1):

    # Residual Loss
    G_z = G(z)
    residual_loss = torch.sum(torch.abs(x - G_z))

    # Discrimination Loss
    feature_G_z , _ = D(G_z)
    feature_x , _ = D(x)
    discrimination_loss = torch.sum(torch.abs(feature_x - feature_G_z))

    total_loss = (1 - __lambda__) * residual_loss + __lambda__ * discrimination_loss
    return G_z, total_loss

def load_data(train_dir, tranform, config):
    return torch.utils.data.DataLoader(datasets.)

