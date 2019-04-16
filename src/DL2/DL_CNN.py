class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(  #
                in_channels=3,  # 输入通道数，RGB图像为3，灰度图像为1
                out_channels=8,  # 输出通道数
                kernel_size=3,  # 卷积核的长宽
                stride=1,  # 步长
                padding=1,  # 在图像旁边补充，如果步长为1，padding=（kernel_size-1)/2
            ),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # nn.MaxPool2d(
            #     kernel_size=2
            # ),
            # 224
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 112
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 48, 3, 1, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 56
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(48, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(2),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 28
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.MaxPool2d(2),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 14
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            # 14
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 7
        )
        # self.out = nn.Linear(1024 * 7 * 7, 65)
        self.out = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            # nn.Linear(1024, 512),
            # nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output