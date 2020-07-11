import torch
import torch.nn as nn
import torch.nn.functional as F

class deepKNet(nn.Module):
    def __init__(self):
        super(deepKNet, self).__init__()
        # layer 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # fc1
        self.fc1 = nn.Linear(16384, 1)

    def forward(self, image):
        # image size -- (BS, C, H, W)
        net = F.relu(self.bn1(self.conv1(image)))
        net = self.max_pool1(net)

        net = F.relu(self.bn2(self.conv2(net)))
        net = self.max_pool2(net)

        net = net.reshape(net.shape[0], -1)
        
        y_pred = self.fc1(net)
        
        return y_pred


