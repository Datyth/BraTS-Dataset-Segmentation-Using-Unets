import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        kernel_initializer = nn.init.kaiming_uniform_

        # Contraction path
        self.conv1 = self._double_conv(in_channels, 16, kernel_initializer)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = self._double_conv(16, 32, kernel_initializer)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = self._double_conv(32, 64, kernel_initializer)
        self.pool3 = nn.MaxPool3d(2)

        self.conv4 = self._double_conv(64, 128, kernel_initializer)
        self.pool4 = nn.MaxPool3d(2)

        self.conv5 = self._double_conv(128, 256, kernel_initializer)

        # Expansive path
        self.upconv6 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.conv6 = self._double_conv(256, 128, kernel_initializer)

        self.upconv7 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv7 = self._double_conv(128, 64, kernel_initializer)

        self.upconv8 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.conv8 = self._double_conv(64, 32, kernel_initializer)

        self.upconv9 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.conv9 = self._double_conv(32, 16, kernel_initializer)

        # Final output
        self.final = nn.Conv3d(16, out_channels, kernel_size=1)

    def _double_conv(self, in_channels, out_channels, kernel_initializer):
        layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Apply kernel initializer to all layers
        for layer in layers:
            if isinstance(layer, nn.Conv3d):
                kernel_initializer(layer.weight)
        return layers

    def forward(self, x):
        # Contraction path
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        c5 = self.conv5(p4)

        # Expansive path
        u6 = self.upconv6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.conv6(u6)

        u7 = self.upconv7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.conv7(u7)

        u8 = self.upconv8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.conv8(u8)

        u9 = self.upconv9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.conv9(u9)

        outputs = self.final(c9)
        return outputs


if __name__ == "__main__":
    model = UNet3D(in_channels=3, out_channels=4)
    print(model)

    x = torch.randn(2, 3, 128, 128, 128)  # Batch size 1, 3 channels, 128x128x128
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
