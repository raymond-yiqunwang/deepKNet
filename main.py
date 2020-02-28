import argparse
from deepKNet.data import deepKNetDataset
from deepKNet.model import deepKNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
#from torch.utils.tensorboard import SummaryWriter

def main():
    # load data
    dataset = deepKNetDataset('./data/', 'band_gap')
    # TODO random sampler or shuffle?
#    indices = list(range(len(dataset)))
#    train_sampler = SubsetRandomSampler(indices)
#    data_loader = DataLoader(dataset, batch_size=8, sampler=train_sampler, num_workers=4)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    # build model
    model = deepKNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # TensorBoard writer
#    writer = SummaryWriter('runs/test1')
    # training
    for epoch in range(2):
        train(data_loader, model, criterion, optimizer, epoch)


def train(train_loader, model, criterion, optimizer, epoch):
    # training mode
    model.train()

    running_loss = 0.0
    for idx, data in enumerate(train_loader):
        # features and target
        point_cloud, target = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(point_cloud)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if (idx+1)%1000 == 0:
            # log the running loss
#            writer.add_scalar('training loss',
#                              running_loss/1000,
#                              epoch*len(trainloader)+i)
            # print loss
            print('[{:d}, {:5d}] loss: {:.3f}'\
                .format(epoch+1, idx+1, running_loss/1000))

            # reset running loss
            running_loss = 0.0


if __name__ == "__main__":
    main()


