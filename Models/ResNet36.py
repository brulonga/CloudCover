import torch.nn as nn
import torch.nn.functional as F 
from Utils.models import n_block

class ResNet36(nn.Module):
    def __init__(self, n, num_classes, activation=nn.ReLU()):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.block1 = n_block(n, 16, 32, stride=2, activation=activation)
        self.block2 = n_block(n, 32, 64, stride=2, activation=activation)
        self.block3 = n_block(n, 64, 128, stride=2, activation=activation)
        self.block4 = n_block(n, 128, 256, stride=2, activation=activation)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc3 = nn.Linear(512, num_classes)

        self.bnL1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(p=0.4)

        self.activation = activation

    def forward(self, x):
        y = self.activation(self.bn1(self.conv1(x)))
        y = self.block4(self.block3(self.block2(self.block1(y))))
        y = self.flatten(y)
        y = self.dropout(self.bnL1(self.activation(self.fc1(y))))
        y = self.fc3(y)
        return y
