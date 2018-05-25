"""
Run similar model as in lab 4

modified from https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from util import plot_confusion_matrix


def cifar100(seed):
    np.random.seed(seed)
    resize = transforms.Resize((224, 224))
    transform = transforms.Compose(
        [resize, transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    trainset = torchvision.datasets.ImageFolder(root='./faces/train',
                                             transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.ImageFolder(root='./faces/test',
                                            transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    return trainloader, testloader


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

    def forward(self):
        raise StandardError

    def fit(self, trainloader):
        # switch to train mode
        self.train()

        # define loss function
        criterion = nn.CrossEntropyLoss()

        # setup SGD
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.0)

        for epoch in range(20):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # compute forward pass
                outputs = self.forward(inputs)

                # get loss function
                loss = criterion(outputs, labels)

                # do backward pass
                loss.backward()

                # do one gradient step
                optimizer.step()

                # print statistics
                running_loss += loss.data[0]

            print('[Epoch: %d] loss: %.3f' %
                  (epoch + 1, running_loss / (i+1)))
            running_loss = 0.0

        print('Finished Training')

    def predict(self, testloader):
        # switch to evaluate mode
        self.eval()

        correct = 0
        total = 0
        all_predicted = []
        for images, labels in testloader:
            outputs = self.forward(Variable(images))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            all_predicted += predicted.numpy().tolist()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

        return all_predicted


class ConvNet(BaseNet):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(1152, 5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 1152)
        x = self.fc1(x)
        return x


class FullNet(BaseNet):
    def __init__(self):
        super(FullNet, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 500)
        self.fc2 = nn.Linear(500, 5)

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


def main():
    # get data
    trainloader, testloader = cifar100(1337)

    # full net
    print "Fully connected network"
    net1 = FullNet()
    net1.fit(trainloader)
    pred_labels = net1.predict(testloader)
    plt.figure(1)
    test_labels = testloader.dataset.test_labels
    plot_confusion_matrix(pred_labels, test_labels, "FullNet")

    # conv net
    print "Convolutional network"
    net2 = ConvNet()
    net2.fit(trainloader)
    pred_labels = net2.predict(testloader)
    plt.figure(2)
    plot_confusion_matrix(pred_labels, test_labels, "ConvNet")

    plt.show()

main()
