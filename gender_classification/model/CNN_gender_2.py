import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 96, kernel_size=7,stride=4)
        self.maxPool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm1 = nn.CrossMapLRN2d(5, alpha=0.0001, beta=0.75)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.maxPool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm2 = nn.CrossMapLRN2d(5, alpha=0.0001, beta=0.75)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.maxPool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(384 * 6 * 6 ,512)
        self.dropout1 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.maxPool1(F.relu(self.conv1(x)))
        x = self.norm1(x)

        x = self.maxPool2(F.relu(self.conv2(x)))
        x = self.norm2(x)

        x = self.maxPool3(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
