import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class FeatureAlignModule(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU())
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())
        self.conv2 = nn.Conv2d(in_channels//2, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        

    def forward(self,x):
        x0, x1 = torch.chunk(x, 2, dim=1)
        x = self.conv0(x)
        x = self.conv1(x)
        attn0, attn1 = torch.split(x, [x0.shape[1], x1.shape[1]], dim=1)
        attn = attn0 * x0 + attn1 * x1
        attn = self.conv2(attn)
        return attn