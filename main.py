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
from generator import Generator
from discriminator import Discriminator
import config

def get_args():
    import argparse

    parser =argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description= "Pytorch implementation of 'Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery'"

    parser.add_argument('--pretrained', dest='pretrained', help="switch for using pretrained model",
                        action='store_true', default=False)
    parser.add_argument('--anomaly', dest='anomaly', help="switch for anomaly detecting", action='store_true',
                        default=True)
    parser.add_argument('--root_dir', type=str, dest='root_dir', help='the path of current directory')
    parser.add_argument('--train_dir', type=str, dest='train_dir', help='the path of train data')
    parser.add_argument('--checkpoint_dir', type=str, dest='checkpoint_dir', help='the path of chekcpoint dir',
                        default='checkpoint')
    parser.add_argument('--save_dir', type=str, dest='save_dir', help='the path of generated data dir',
                        default='sample')
    parser.add_argument('--test_dir', type=str, dest='test_dir', help='the path of anomaly test data')
    parser.add_argument('--test_result_dir', type=str, dest='test_result_dir',
                        help='the path of anomaly test result dir')

    args = parser.parse_args()

    return args

def main():
    args = get_args()


    transform = transforms.Compose([
        transforms.Scale()
    ])

    G = Generator()
    D = Discriminator()
