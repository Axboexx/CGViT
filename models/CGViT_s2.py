"""
@Project ：ml-cvnets 
@File    ：CGViT_s2.py
@IDE     ：PyCharm 
@Author  ：chengxuLiu
@Date    ：2023/12/28 17:00 
"""
from utils.Ghostmodel import *
from utils.PartialConv import *
from utils.mobilevit2 import *
import torch.nn as nn


class CGViT_s2(nn.Module):
    def __init__(self, num_classes=92):
        super(CGViT_s2, self).__init__()

        self.to_latent = nn.Identity()

        self.Stage1 = nn.Sequential(
            GhostModule(3, 32, kernel_size=1, stride=2, relu=True),
            Partial_block(32),
            Partial_block(32),
        )
        self.Stage2 = nn.Sequential(
            GhostModule(32, 64, kernel_size=1, stride=2, relu=True),
            Partial_block(64),
            Partial_block(64),
            Partial_block(64),
            Partial_block(64)
        )
        self.Stage3 = nn.Sequential(
            GhostModule(64, 96, kernel_size=3, stride=2, relu=True),
            Partial_block(96),
            Partial_block(96),
            Partial_block(96),
            Partial_block(96),
            Partial_block(96),
            Partial_block(96)
        )
        self.Stage4 = nn.Sequential(
            GhostModule(96, 128, kernel_size=3, relu=True),
            Partial_block(128),
            Partial_block(128),
            Partial_block(128),
            Partial_block(128),
            Partial_block(128),
            Partial_block(128)
        )
        self.Stage5 = nn.Sequential(
            GhostModule(128, 256, kernel_size=3, stride=2, relu=True),
            Partial_block(256),
            Partial_block(256),
            Partial_block(256),
            Partial_block(256),
            Partial_block(256)
        )

        self.Transformer = MobileViTBlock(256, 2, 256, 3, (2, 2), 256, 0.2)
        self.conv2 = conv_1x1_bn(256, 960)

        self.pool = nn.AvgPool2d(14, 1)

        self.classifier = nn.Sequential(
            nn.Linear(960, 1280, bias=False),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x,):
        x = self.Stage1(x)
        x = self.Stage2(x)
        x = self.Stage3(x)
        x = self.Stage4(x)
        x = self.Stage5(x)
        x = self.Transformer(x)
        x = self.conv2(x)

        x = self.pool(x).view(x.size(0), -1)

        x = self.classifier(x)
        return x