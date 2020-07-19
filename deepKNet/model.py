import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, classification=True):
        super(LeNet5, self).__init__()
        self.classification = classification
        # conv1
        self.conv1 = nn.Conv2d(21, 64, kernel_size=7,stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # conv2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3)
        self.bn2 = nn.BatchNorm2d(128)
        self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # FC layers
        self.fc1 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        if self.classification:
            self.fc3 = nn.Linear(64, 2)
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.fc3 = nn.Linear(64, 1)

    def forward(self, image):
        net = F.relu(self.bn1(self.conv1(image)))
        net = self.pooling1(net)

        net = F.relu(self.bn2(self.conv2(net)))
        net = self.pooling2(net)
        
        # pooling multi-views
        #net = torch.max(net, dim=1, keepdim=True)[0]
        net = torch.mean(net, dim=1, keepdim=True)

        net = net.view(image.shape[0], -1)
        net = F.relu(self.bn3(self.fc1(net)))
        if self.classification:
            net = self.dropout(net)
        
        net = F.relu(self.bn4(self.fc2(net)))
       
        out = self.fc3(net)
        if self.classification:
            out = self.logsoftmax(out)
        
        return out


#class ResNet(nn.Module):
    
