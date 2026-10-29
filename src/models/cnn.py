import torch
from torch import nn


class BaselineCNN(nn.Module):
    """Four-convolution CNN baseline; a new, documented approximation, not original source."""
    def __init__(self, num_classes=10, in_channels=1, channels=(13, 28, 43, 43)):
        super().__init__()
        layers = []
        previous = in_channels
        for i, width in enumerate(channels):
            layers += [nn.Conv2d(previous, width, kernel_size=3, padding=1),
                       nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
            if i in (1, 3):
                layers.append(nn.MaxPool2d(2))
            previous = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(previous, 256),
                                        nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))
