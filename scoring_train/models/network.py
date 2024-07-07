import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from torch.utils.data import Sampler
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import time
import pickle
import numpy as np
from torchvision.transforms import Lambda
import argparse
import copy
import random
import numbers
#from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import os


class resnet_lstm4_fle(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm4_fle, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.sequence_length = 11
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 4)
        self.dropout = nn.Dropout(p=0.5)

        init.xavier_normal_(self.lstm.all_weights[0][0])
        init.xavier_normal_(self.lstm.all_weights[0][1])
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        length = x.shape[1]
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, length, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
        y = self.dropout(y)
        y = self.fc1(y)
        y = self.dropout(y)
        y = F.relu(y)
        y = self.fc2(y)
        return y
    
if __name__ == '__main__':

    aaa = torch.randn(2,3,224,224)

    bb = resnet_lstm4_fle()

    print(bb(aaa).shape)