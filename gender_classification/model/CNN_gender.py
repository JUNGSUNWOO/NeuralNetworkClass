import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=7, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.avgpool1 = nn.AvgPool2d(2, stride=2, padding=1)
        self.dropout = nn.Dropout(0.5)

        self.conv3 = nn.Conv2d(16, 96, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.avgpool1 = nn.AvgPool2d(2, stride=2, padding=1)
        self.dropout = nn.Dropout(0.5)

        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.avgpool1 = nn.AvgPool2d(2, stride=2, padding=1)
        self.dropout = nn.Dropout(0.5)

        self.conv7 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.avgpool1 = nn.AvgPool2d(2, stride=2, padding=1)
        self.dropout = nn.Dropout(0.5)

        self.conv9 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.avgpool1 = nn.AvgPool2d(2, stride=2, padding=1)

        # self.pool = nn.MaxPool2d(2, stride=2)
        # self.pool2 = nn.AvgPool2d(2, stride=2)

        self.fc1 = nn.Linear(256 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 2)
        # self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        # print(x.shape)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn1(F.relu(self.conv2(x)))
        x = self.avgpool1(x)
        x = self.dropout(x)

        # print(x.shape)
        x = self.bn2(F.relu(self.conv3(x)))
        x = self.bn2(F.relu(self.conv4(x)))
        x = self.avgpool1(x)
        x = self.dropout(x)

        # print(x.shape)
        x = self.bn3(F.relu(self.conv5(x)))
        x = self.bn3(F.relu(self.conv6(x)))
        x = self.avgpool1(x)
        x = self.dropout(x)

        # print(x.shape)
        x = self.bn4(F.relu(self.conv7(x)))
        x = self.bn4(F.relu(self.conv8(x)))
        x = self.avgpool1(x)
        x = self.dropout(x)

        # print(x.shape)
        x = self.bn5(F.relu(self.conv9(x)))
        x = F.relu(self.conv10(x))
        x = self.avgpool1(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
