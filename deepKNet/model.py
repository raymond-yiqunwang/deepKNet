import torch
import torch.nn as nn
import torch.nn.functional as F

# LeNet-based implementation
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # conv1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,stride=1, padding=3)
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
        self.fc3 = nn.Linear(64, 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, image):
        net = F.relu(self.bn1(self.conv1(image)))
        net = self.pooling1(net)

        net = F.relu(self.bn2(self.conv2(net)))
        net = self.pooling2(net)
        
        # pooling multi-views
        net = torch.max(net, dim=1, keepdim=True)[0]

        net = net.view(image.shape[0], -1)
        net = F.relu(self.bn3(self.fc1(net)))
        net = self.dropout(net)
        
        net = F.relu(self.bn4(self.fc2(net)))
        out = self.logsoftmax(self.fc3(net))
        
        return out


# ResNet-based implementation
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, net):
        identity = net
        if self.downsample is not None:
            identity = self.downsample(net)
        out = F.relu(self.bn1(self.conv1(net)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)

        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(mod, nn.BatchNorm2d):
                nn.init.constant_(mod.weight, 1)
                nn.init.constant_(mod.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, image):
        net = F.relu(self.bn1(self.conv1(image)))
        net = self.maxpool(net)

        net = self.layer1(net)
        net = self.layer2(net)
        net = self.layer3(net)
        net = self.layer4(net)

        net = self.avgpool(net)
        net = torch.flatten(net, 1)
#        net = self.dropout(net)
        out = self.logsoftmax(self.fc(net))
        
        return out


