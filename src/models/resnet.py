import torch.nn as nn

# ResNet18 re-implementation, modeled after the ResNet paper (https://arxiv.org/pdf/1512.03385)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResidualBlock, self).__init__()

        self.weightLayer1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,  # downsampling in the first block (see table 1)
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU()
        self.weightLayer2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
        )

        # we need to fix the shape of x when applying stride=2 or when the channels don't match
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        # F(x)
        out = self.weightLayer1(x)
        out = self.relu(out)
        out = self.weightLayer2(out)

        # F(x) + x
        out = out + self.downsample(identity)
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, grayscale=False):
        super(ResNet18, self).__init__()

        if grayscale:
            # for grayscale images, i.e MNIST, we need to change the number of input channels
            in_channels = 1
        else:
            in_channels = 3

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 64, stride=1),
        )
        self.conv3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1),
        )
        self.conv4 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1),
        )
        self.conv5 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=1),
        )
        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.avgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
