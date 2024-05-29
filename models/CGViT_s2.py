"""
@Project ：ml-cvnets 
@File    ：CGViT_s2.py
@IDE     ：PyCharm 
@Author  ：chengxuLiu
@Date    ：2023/12/28 17:00 
"""
from .Ghostmodel import *
from .PartialConv import *
from .mobilevit2 import *
from timm.models.mobilevit import MobileVitV2Block as mvv2b
from cvnets.models import MODEL_REGISTRY
from cvnets.models.classification.base_image_encoder import BaseImageEncoder
from cvnets.models.classification.config.testnet import get_configuration


@MODEL_REGISTRY.register(name="CGViT_s2", type="classification")
class CGViT_s2(BaseImageEncoder):
    def __init__(self, opts, num_classes=92):
        num_classes = getattr(opts, "model.classification.n_classes", 1000)
        super(CGViT_s2, self).__init__(opts)

        # self.pool = 'cls'
        self.to_latent = nn.Identity()
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(960),
        #     nn.Linear(960, 92)
        # )
        self.Stage1 = nn.Sequential(
            GhostModule(3, 32, kernel_size=1, stride=2, relu=True),
            Partial_block(32),
            Partial_block(32),
            # GhostModule(64, 120, stride=2, relu=True)
        )
        self.Stage2 = nn.Sequential(
            GhostModule(32, 64, kernel_size=1, stride=2, relu=True),
            Partial_block(64),
            Partial_block(64),
            Partial_block(64),
            Partial_block(64)
            # Partial_block(120)
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
        # self.PatchEmbed = PatchEmbedding(embed_size=960, patch_size=3, channels=960, img_size=14)
        # self.Transformer = Transformer(dim=960, depth=1, n_heads=4, mlp_expansions=1, dropout=0.2)

        self.classifier = nn.Sequential(
            nn.Linear(960, 1280, bias=False),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
        # self.classifier=nn.Linear()

    def forward(self, x, *args, **kwargs):
        x = self.Stage1(x)
        x = self.Stage2(x)
        x = self.Stage3(x)
        x = self.Stage4(x)
        x = self.Stage5(x)
        x = self.Transformer(x)
        x = self.conv2(x)
        # x.shape(3,960,14,14)
        x = self.pool(x).view(x.size(0), -1)
        # x = self.fc(x)
        # x = self.PatchEmbed(x)
        # x = self.Transformer(x)
        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        # x = self.to_latent(x)
        # x = self.mlp_head(x)
        # x = self.pool(x)
        # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# @MODEL_REGISTRY.register(name="Testnet_mv_4_5M_0_7G", type="classification")
# class Testnet(BaseImageEncoder):
#     def __init__(self, opts):
#         super(Testnet, self).__init__(opts)
#         num_classes = getattr(opts, "model.classification.n_classes", 92)
#
#         cfg = get_configuration(opts)
#
#         stage = cfg["stage1"]
#         # self.Stage1 = nn.Sequential(
#         #     nn.Conv2d(in_channels=stage["gm"][0], out_channels=stage["gm"][1], kernel_size=stage["gm"][2],
#         #               stride=stage["gm"][3]),
#         #     nn.BatchNorm2d(stage["gm"][1]),
#         #     nn.ReLU(inplace=True)
#         # )
#         self.Stage1 = nn.Sequential()
#         self.Stage1.append(
#             GhostModule(inp=stage["gm"][0], oup=stage["gm"][1], kernel_size=stage["gm"][2], stride=stage["gm"][3],
#                         relu=stage["gm"][4])
#         )
#
#         for i in range(stage["pb_num"]):
#             self.Stage1.append(Partial_block(stage["pb_channel"]))
#
#         if cfg["mv2_1"]["use"]:
#             stage = cfg["mv2_1"]
#             self.mv2_1 = mvv2b(stage["in"], stage["out"], drop_path_rate=stage["drop_path_rate"])
#         else:
#             self.mv2_1 = nn.Identity()
#
#         self.Stage2 = nn.Sequential()
#         stage = cfg["stage2"]
#         self.Stage2.append(
#             GhostModule(inp=stage["gm"][0], oup=stage["gm"][1], kernel_size=stage["gm"][2], stride=stage["gm"][3],
#                         relu=stage["gm"][4]))
#         for i in range(stage["pb_num"]):
#             self.Stage2.append(Partial_block(stage["pb_channel"]))
#
#         if cfg["mv2_2"]["use"]:
#             stage = cfg["mv2_2"]
#             self.mv2_2 = mvv2b(stage["in"], stage["out"], drop_path_rate=stage["drop_path_rate"])
#         else:
#             self.mv2_2 = nn.Identity()
#
#         self.Stage3 = nn.Sequential()
#         stage = cfg["stage3"]
#         self.Stage3.append(
#             GhostModule(inp=stage["gm"][0], oup=stage["gm"][1], kernel_size=stage["gm"][2], stride=stage["gm"][3],
#                         relu=stage["gm"][4]))
#         for i in range(stage["pb_num"]):
#             self.Stage3.append(Partial_block(stage["pb_channel"]))
#
#         if cfg["mv2_3"]["use"]:
#             stage = cfg["mv2_3"]
#             self.mv2_3 = mvv2b(stage["in"], stage["out"], drop_path_rate=stage["drop_path_rate"])
#         else:
#             self.mv2_3 = nn.Identity()
#
#         self.Stage4 = nn.Sequential()
#         stage = cfg["stage4"]
#         self.Stage4.append(
#             GhostModule(inp=stage["gm"][0], oup=stage["gm"][1], kernel_size=stage["gm"][2], stride=stage["gm"][3],
#                         relu=stage["gm"][4]))
#         for i in range(stage["pb_num"]):
#             self.Stage4.append(Partial_block(stage["pb_channel"]))
#
#         if cfg["mv2_4"]["use"]:
#             stage = cfg["mv2_4"]
#             self.mv2_4 = mvv2b(stage["in"], stage["out"], drop_path_rate=stage["drop_path_rate"])
#         else:
#             self.mv2_4 = nn.Identity()
#
#         self.Stage5 = nn.Sequential()
#         stage = cfg["stage5"]
#         self.Stage5.append(
#             GhostModule(stage["gm"][0], stage["gm"][1], kernel_size=stage["gm"][2], stride=stage["gm"][3],
#                         relu=stage["gm"][4]))
#         for i in range(stage["pb_num"]):
#             self.Stage5.append(Partial_block(stage["pb_channel"]))
#
#         if cfg["mv2_5"]["use"]:
#             stage = cfg["mv2_5"]
#             # self.mv2_5 = mvv2b(stage["in"], stage["out"], drop_path_rate=stage["drop_path_rate"])
#             # self.mv2_5 = MobileViTBlock(stage[1], stage[2], stage[3], stage[4], (stage[5], stage[5]), stage[6],
#             #                             stage[7])
#             self.mv2_5 = MobileViTBlock(256, 2, 256, 3, (2, 2), 256, 0.2)
#         else:
#             self.mv2_5 = nn.Identity()
#         stage = cfg["classfier"]
#         self.base = stage["base"]
#         if stage["base"]:
#             self.conv2d = conv_1x1_bn(256, 960)
#             self.pool2 = nn.AvgPool2d(14, 1)
#             self.classifier = nn.Sequential(
#                 nn.Linear(960, 1280, bias=False),
#                 nn.BatchNorm1d(1280),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(0.2),
#                 nn.Linear(1280, num_classes)
#             )
#         else:
#             self.conv2 = conv_1x1_bn(stage["conv"][0], stage["conv"][1])
#             self.pool = nn.AvgPool2d(stage["pool"][0], stage["pool"][1])
#             self.fc = nn.Linear(stage["fc"][0], num_classes, bias=stage["fc"][1])
#
#     def forward(self, x, *args, **kwargs):
#         x = self.Stage1(x)
#         x = self.mv2_1(x)
#         x = self.Stage2(x)
#         x = self.mv2_2(x)
#         x = self.Stage3(x)
#         x = self.mv2_3(x)
#         x = self.Stage4(x)
#         x = self.mv2_4(x)
#         x = self.Stage5(x)
#         x = self.mv2_5(x)
#         if self.base:
#             x = self.conv2d(x)
#             x = self.pool2(x).view(x.size(0), -1)
#             # x = torch.flatten(x, start_dim=1, end_dim=3)
#             x = self.classifier(x)
#         else:
#             x = self.conv2(x)
#             x = self.pool(x).view(-1, x.shape[1])
#             x = self.fc(x)
#
#         return x
