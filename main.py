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
from models.generator import Generator
from models.discriminator import Discriminator
import config

def get_args():
    import argparse

    parser =argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description= "Pytorch implementation of 'Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery'")

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([
        transforms.Scale()
    ])


    transform = transforms.Compose([
        transforms.Scale((512,512,3)),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5,0.5,0.5),
                             std =  (0.5,0.5,0.5))
    ])

    G = Generator().to(device)
    D = Discriminator().to(device)

    loss_func = nn.MSELoss()

    G_optim = torch.optim.adam(G.parameters(), lr = 0.01)
    D_optim = torch.optim.adam(D.parameters(), lr = 0.01)

    epochs = 500
    import tqdm
    for epoch in tqdm(range(epochs)):
        for batch_idx, (x, labels) in tqdm(enumerate(train_loader)):
            # discriminator

            z = init.normal(torch.Tensor(batch_size, 100), mean=0, std=0.1).to(device)
            D_z, _ = D(G(z))
            D_x, _ = D(x)

            zeros = torch.zeros(batch_size,1).to(device)
            D_loss = torch.sum(loss_func(D_z, zeros)) + torch.sum(loss_func(D_imgs,ones_label))

            D_optim.zero_grad()
            D_loss.backward()
            D_optim.step()

            #generator

            z = init.normal(torch.Tensor(batch_size , 100), mean = 0 , std = 0.1 ).to(device)
            G_z = G(z)
            D_z = D(G_z)

            ones = torch.ones(batch_size,1)
            G_loss = torch.sum(loss_func(D_z,ones))
            G_optim.zero_grad()
            G_loss.backward(retain_graph = True)
            G_optim.step()

            # model save

