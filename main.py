import argparse
import numpy as np
from deepKNet.dataset import deepKNetDataset
from deepKNet.model import deepKNet
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

def main():
    # load data
    dataset = deepKNetDataset('./data/', 'band_gap')
    indices = list(range(len(dataset)))
    train_sampler = SubsetRandomSampler(indices)
    data_loader = DataLoader(dataset, batch_size=8, sampler=train_sampler, num_workers=4)

    # build model
    model = deepKNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # training
    for epoch in range(2):
        train(data_loader, model, criterion, optimizer, epoch)


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    for idx, (point_cloud, target) in enumerate(train_loader):
        input_var = Variable(point_cloud)
        target_var = Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)
        print(idx, loss)

        # compute grad and optim
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()


