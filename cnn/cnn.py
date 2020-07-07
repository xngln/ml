import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim 
import torchvision
import math

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.cnn_layers = nn.Sequential(
            # cnn layer 1
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #cnn layer 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # cnn layer 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # cnn layer 4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # cnn layer 5
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # cnn layer 6
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # cnn layer 7 
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # cnn layer 8
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            # fc layer 1
            nn.Linear(512, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            # fc layer 2
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            # fc layer 3
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    

train_set = torchvision.datasets.MNIST("./", train=True, transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)), torchvision.transforms.ToTensor()]), download=True)
test_set = torchvision.datasets.MNIST("./", train=False, transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)), torchvision.transforms.ToTensor()]), download=True)
train_set_subset = torch.utils.data.Subset(train_set, list(range(6000)))
train_loader = torch.utils.data.DataLoader(train_set_subset, batch_size=100, shuffle=False)
test_set_subset = torch.utils.data.Subset(test_set, list(range(1000)))
test_loader = torch.utils.data.DataLoader(test_set_subset, batch_size=1000, shuffle=False)

net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(1, 3):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        train_x, train_y = autograd.Variable(inputs), autograd.Variable(labels)

        net.train() # put net in training mode

        optimizer.zero_grad()

        train_pred = net(train_x) 

        train_loss = criterion(train_pred, train_y)
        
        train_loss.backward()
        optimizer.step()
        
        if i % 24 == 0: 
            train_losses.append(train_loss.item())
            _, train_predictions = torch.max(train_pred.data, 1)
            train_accuracy = 0.0
            for i in range(100):
                if train_predictions[i] == train_y[i]:
                    train_accuracy += 1
            train_accuracy = float(train_accuracy / 100)
            train_accuracies.append(train_accuracy)
            print(epoch, " train ", train_accuracy)

            for j, test_data in enumerate(test_loader, 0):
                test_inputs, test_labels = test_data
                test_x, test_y = autograd.Variable(test_inputs), autograd.Variable(test_labels)
                test_pred = net(test_x)
                test_loss = criterion(test_pred, test_y)
                test_losses.append(test_loss.item())
                _, predictions = torch.max(test_pred.data, 1)

                test_accuracy = 0.0
                for i in range(1000):
                    if predictions[i] == test_y[i]:
                        test_accuracy += 1
                test_accuracy = float(test_accuracy / 1000)
                test_accuracies.append(test_accuracy)
                print(epoch, " test ", test_accuracy)

plt.plot(list(range(len(train_losses))), train_losses)
plt.savefig('q2_train_loss.png')
plt.close()
plt.plot(list(range(len(train_accuracies))), train_accuracies)
plt.savefig('q2_train_accuracy.png')
plt.close()
plt.plot(list(range(len(test_losses))), test_losses)
plt.savefig('q2_test_loss.png')
plt.close()
plt.plot(list(range(len(test_accuracies))), test_accuracies)
plt.savefig('q2_test_accuracy.png')
plt.close()


### Question 3 ###
test_horiz = torchvision.datasets.MNIST("./", train=False, transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)), torchvision.transforms.RandomHorizontalFlip(p=1),torchvision.transforms.ToTensor()]), download=True)
test_horiz_loader = torch.utils.data.DataLoader(test_horiz, batch_size=10000, shuffle=False)
for j, data in enumerate(test_horiz_loader, 0):
    test_inputs, test_labels = data
    test_x, test_y = autograd.Variable(test_inputs), autograd.Variable(test_labels)
    test_pred = net(test_x)
    test_loss = criterion(test_pred, test_y)
    test_losses.append(test_loss.item())
    _, predictions = torch.max(test_pred.data, 1)

    test_accuracy = 0.0
    for i in range(10000):
        if predictions[i] == test_y[i]:
            test_accuracy += 1
    test_accuracy = float(test_accuracy / 10000)
    print("horizontal accuracy ", test_accuracy)


test_vert = torchvision.datasets.MNIST("./", train=False, transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)), torchvision.transforms.RandomVerticalFlip(p=1), torchvision.transforms.ToTensor()]), download=True)
test_vert_loader = torch.utils.data.DataLoader(test_vert, batch_size=10000, shuffle=False)
for j, data in enumerate(test_vert_loader, 0):
    test_inputs, test_labels = data
    test_x, test_y = autograd.Variable(test_inputs), autograd.Variable(test_labels)
    test_pred = net(test_x)
    test_loss = criterion(test_pred, test_y)
    test_losses.append(test_loss.item())
    _, predictions = torch.max(test_pred.data, 1)

    test_accuracy = 0.0
    for i in range(10000):
        if predictions[i] == test_y[i]:
            test_accuracy += 1
    test_accuracy = float(test_accuracy / 10000)
    print("vertical accuracy ", test_accuracy)

test_noise_1 = torchvision.datasets.MNIST("./", train=False, transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)), torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambda x : x + 1*torch.randn_like(x))]), download=True)
test_noise_1_loader = torch.utils.data.DataLoader(test_noise_1, batch_size=10000, shuffle=False)
for j, data in enumerate(test_noise_1_loader, 0):
    test_inputs, test_labels = data
    test_x, test_y = autograd.Variable(test_inputs), autograd.Variable(test_labels)
    test_pred = net(test_x)
    test_loss = criterion(test_pred, test_y)
    test_losses.append(test_loss.item())
    _, predictions = torch.max(test_pred.data, 1)

    test_accuracy = 0.0
    for i in range(10000):
        if predictions[i] == test_y[i]:
            test_accuracy += 1
    test_accuracy = float(test_accuracy / 10000)
    print("noise 1 accuracy ", test_accuracy)

test_noise_01 = torchvision.datasets.MNIST("./", train=False, transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)), torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambda x : x + math.sqrt(0.1)*torch.randn_like(x))]), download=True)
test_noise_01_loader = torch.utils.data.DataLoader(test_noise_01, batch_size=10000, shuffle=False)
for j, data in enumerate(test_noise_01_loader, 0):
    test_inputs, test_labels = data
    test_x, test_y = autograd.Variable(test_inputs), autograd.Variable(test_labels)
    test_pred = net(test_x)
    test_loss = criterion(test_pred, test_y)
    test_losses.append(test_loss.item())
    _, predictions = torch.max(test_pred.data, 1)

    test_accuracy = 0.0
    for i in range(10000):
        if predictions[i] == test_y[i]:
            test_accuracy += 1
    test_accuracy = float(test_accuracy / 10000)
    print("noise 01 accuracy ", test_accuracy)

test_noise_001 = torchvision.datasets.MNIST("./", train=False, transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)), torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambda x : x + 0.1*torch.randn_like(x))]), download=True)
test_noise_001_loader = torch.utils.data.DataLoader(test_noise_001, batch_size=10000, shuffle=False)
for j, data in enumerate(test_noise_001_loader, 0):
    test_inputs, test_labels = data
    test_x, test_y = autograd.Variable(test_inputs), autograd.Variable(test_labels)
    test_pred = net(test_x)
    test_loss = criterion(test_pred, test_y)
    test_losses.append(test_loss.item())
    _, predictions = torch.max(test_pred.data, 1)

    test_accuracy = 0.0
    for i in range(10000):
        if predictions[i] == test_y[i]:
            test_accuracy += 1
    test_accuracy = float(test_accuracy / 10000)
    print("noise 001 accuracy ", test_accuracy)
