import torch
from torch import nn


class CyclicGCNN(nn.Module):
    """Four-stage ESCNN cyclic model; widths count regular fields, not channels."""
    def __init__(self, num_classes=10, in_channels=1, channels=(10, 20, 40, 40), order=4):
        super().__init__()
        try:
            from escnn import gspaces
            from escnn import nn as enn
        except ImportError as exc:
            raise ImportError('C4GCNN requires escnn. Install requirements.txt and see README.') from exc
        if order not in (4, 8):
            raise ValueError('Supported cyclic orders: 4 and 8')
        self.order = order
        self.enn = enn
        self.gspace = gspaces.rot2dOnR2(N=order)
        in_type = enn.FieldType(self.gspace, in_channels * [self.gspace.trivial_repr])
        self.input_type = in_type
        blocks = []
        for i, width in enumerate(channels):
            out_type = enn.FieldType(self.gspace, width * [self.gspace.regular_repr])
            blocks.extend([enn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
                           enn.InnerBatchNorm(out_type), enn.ReLU(out_type, inplace=False)])
            if i in (1, 3):
                blocks.append(enn.PointwiseMaxPool(out_type, kernel_size=2, stride=2))
            in_type = out_type
        self.features = enn.SequentialModule(*blocks)
        self.invariant = enn.GroupPooling(in_type)
        self.pool = nn.AdaptiveAvgPool2d(1)
        feature_dim = self.invariant.out_type.size
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(feature_dim, 256),
                                        nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes))

    def forward(self, x):
        features = self.features(self.enn.GeometricTensor(x, self.input_type))
        return self.classifier(self.pool(self.invariant(features).tensor))


class C4GCNN(CyclicGCNN):
    """Backward-compatible C4 model name and original default field widths."""
    def __init__(self, num_classes=10, in_channels=1, channels=(10, 20, 40, 40)):
        super().__init__(num_classes, in_channels, channels, order=4)
