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

def anomaly_score(G,D, ano_z, test_input):
    """
    Calculate anomaly score
    :param G:
    :param D:
    :param ano_z:
    :param test_input:
    :return:
    """
    ano_G = G(ano_z)
    residual_loss = torch.sum(torch.abs(test_input - ano_G))

    feature_ano_G , _ = D(ano_G)
    feature_input , _ = D(test_input)
    discriminator_loss = torch.sum(torch.abs(feature_ano_G - feature_input))

    total_loss = (1 - ano_coeff) * residual_loss + ano_coeff * discriminator_loss
    return ano_G, total_loss

