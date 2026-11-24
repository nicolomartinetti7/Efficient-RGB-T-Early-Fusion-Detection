import torch
import torch.nn as nn

class EMEClassificationHead(nn.Module):

    def __init__(self, in_channels=128, num_classes=6):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class EMESegmentationHead(nn.Module):

    def __init__(self, in_channels=128, num_classes=6):
        super().__init__()
        self.conv1 = nn.Conv2d(
          in_channels, num_classes,
          kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_classes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
          num_classes, num_classes,
          kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        return x

class EMEWeakSupHeads(nn.Module):

    def __init__(self, num_classes=6):
        super().__init__()
        self.adapters = nn.ModuleList([
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.Conv2d(512, 128, kernel_size=1, bias=False)
        ])
        self.classification_head = EMEClassificationHead(
            in_channels=128,
            num_classes=num_classes
        )

        self.segmentation_head = EMESegmentationHead(
            in_channels=128,
            num_classes=num_classes
        )

    def forward(self, features):

        cls_outputs = []
        seg_outputs = []

        for feat, adapter in zip(features, self.adapters):
            adapted_feat = adapter(feat)

            cls_outputs.append(
                self.classification_head(adapted_feat)
            )

            seg_outputs.append(
                self.segmentation_head(adapted_feat)
            )

        return cls_outputs, seg_outputs