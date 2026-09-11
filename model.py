# model.py
# EEGNet-ish CNN for our EEG task
# mostly follows Lawhern et al. 2018 but its a simplified version I wrote
# Elham Latif

# input shape: (batch, 1, n_channels, n_samples)

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    # small conv net, 2 classes

    def __init__(
        self,
        n_channels,
        n_samples,
        n_classes=2,
        F1=8,
        D=2,
        F2=16,
        dropout=0.5,
    ):
        super().__init__()

        # block 1 - temporal then spatial
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                1,
                F1,
                kernel_size=(1, 64),
                padding=(0, 32),
                bias=False,
            ),
            nn.BatchNorm2d(F1),
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                F1,
                F1 * D,
                kernel_size=(n_channels, 1),
                groups=F1,
                bias=False,
            ),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # block 2 - depthwise + pointwise (separable)
        # keeps params low
        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                F1 * D,
                F1 * D,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=F1 * D,
                bias=False,
            ),
            nn.Conv2d(
                F1 * D,
                F2,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # need the flat size for the linear layer
        # running a dummy through is the easiest way, couldnt find a cleaner one
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            feats = self._forward_features(dummy)
            flat_size = feats.view(1, -1).shape[1]

        self.classifier = nn.Linear(flat_size, n_classes)

    def _forward_features(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # flatten
        return self.classifier(x)


# quick test
if __name__ == "__main__":
    model = EEGNet(n_channels=21, n_samples=1024)

    x = torch.randn(4, 1, 21, 1024)
    out = model(x)

    print("Output shape:", out.shape)