"""
@Project ：ml-cvnets 
@File    ：CGViT_s1.py
@IDE     ：PyCharm 
@Author  ：chengxuLiu
@Date    ：2024/1/1 16:18 
"""
from .Ghostmodel import *
from .PartialConv import *
# from Transformer import *
from .mobilevit2 import *
from timm.models.mobilevit import MobileVitV2Block as mvv2b
from cvnets.models import MODEL_REGISTRY
from cvnets.models.classification.base_image_encoder import BaseImageEncoder


@MODEL_REGISTRY.register(name="CGViT_s1", type="classification")
class CGViT_s1(BaseImageEncoder):
    def __init__(self, opts, num_classes=92):
        num_classes = getattr(opts, "model.classification.n_classes", 1000)
        super(CGViT_s1, self).__init__(opts)

        # self.pool = 'cls'
        self.to_latent = nn.Identity()

        self.Stage1 = nn.Sequential(
            GhostModule(3, 32, kernel_size=1, stride=2, relu=True),
            Partial_block(32),
            Partial_block(32),

        )
        # self.T1 = MobileViTBlock(32, 2, 32, 3, (2, 2), 32, 0.3)
        # self.mv2_1 = mvv2b(32, 32, drop_path_rate=0.2)
        self.Stage2 = nn.Sequential(
            GhostModule(32, 64, kernel_size=1, stride=2, relu=True),
            Partial_block(64),
            Partial_block(64),
            Partial_block(64),
            Partial_block(64)

        )
        # self.T2 = MobileViTBlock(64, 2, 64, 3, (2, 2), 64, 0.3)

        self.Stage3 = nn.Sequential(
            GhostModule(64, 96, kernel_size=3, stride=2, relu=True),
            Partial_block(96),
            Partial_block(96),
            Partial_block(96),
            Partial_block(96),
            Partial_block(96),
            Partial_block(96)
        )
        # self.T3 = MobileViTBlock(96, 2, 96, 3, (2, 2), 96, 0.3)
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
        # self.T4 = MobileViTBlock(128, 2, 128, 3, (2, 2), 128, 0.3)
        self.mv2_4 = mvv2b(128, 128, drop_path_rate=0.2)

        self.Stage5 = nn.Sequential(
            GhostModule(128, 256, kernel_size=3, stride=2, relu=True),
            Partial_block(256),
            Partial_block(256),
            Partial_block(256),
            Partial_block(256)
        )

        # self.T5 = MobileViTBlock(256, 2, 256, 3, (2, 2), 256, 0.3)
        self.mv2_5 = mvv2b(256, 256, drop_path_rate=0.2)
        self.conv2 = conv_1x1_bn(256, 960)

        self.pool = nn.AvgPool2d(8, 1)
        self.fc = nn.Linear(960, num_classes, bias=False)
        # self.classifier = nn.Sequential(
        #     nn.Linear(960, 1280, bias=False),
        #     nn.BatchNorm1d(1280),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.2),
        #     nn.Linear(1280, num_classes)
        # )

    def forward(self, x):
        x = self.Stage1(x)
        # x = self.T1(x)
        x = self.Stage2(x)
        # x = self.T2(x)
        x = self.Stage3(x)
        # x = self.T3(x)
        x = self.mv2_3(x)
        x = self.Stage4(x)
        # x = self.T4(x)
        x = self.mv2_4(x)
        x = self.Stage5(x)
        # x = self.T5(x)
        x = self.mv2_5(x)
        x = self.conv2(x)
        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        # x = self.classifier(x)
        return x
