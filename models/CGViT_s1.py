"""
@Project ：ml-cvnets 
@File    ：CGViT_s1.py
@IDE     ：PyCharm 
@Author  ：chengxuLiu
@Date    ：2024/1/1 16:18 
"""

from utils.Ghostmodel import *
from utils.PartialConv import *
from utils.mobilevit2 import *
# from Transformer import *
from timm.models.mobilevit import MobileVitV2Block as mvv2b


class CGViT_s1(nn.Module):
    def __init__(self, num_classes=92):
        super(CGViT_s1, self).__init__()

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
        self.mv2_3 = mvv2b(96, 96, drop_path_rate=0.2)
        self.Stage4 = nn.Sequential(
            GhostModule(96, 128, kernel_size=3, stride=2, relu=True),
            Partial_block(128),
            Partial_block(128),
            Partial_block(128),
            Partial_block(128),
            Partial_block(128),
            Partial_block(128)
        )
        self.mv2_4 = mvv2b(128, 128, drop_path_rate=0.2)

        self.Stage5 = nn.Sequential(
            GhostModule(128, 256, kernel_size=3, stride=2, relu=True),
            Partial_block(256),
            Partial_block(256),
            Partial_block(256),
            Partial_block(256)
        )

        self.mv2_5 = mvv2b(256, 256, drop_path_rate=0.2)
        self.conv2 = conv_1x1_bn(256, 960)

        self.pool = nn.AvgPool2d(8, 1)
        self.fc = nn.Linear(960, num_classes, bias=False)

    def forward(self, x):
        x = self.Stage1(x)
        x = self.Stage2(x)
        x = self.Stage3(x)
        x = self.mv2_3(x)
        x = self.Stage4(x)
        x = self.mv2_4(x)
        x = self.Stage5(x)
        x = self.mv2_5(x)
        x = self.conv2(x)
        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x
