import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from agc_optims.optim import SGD_AGC, Adam_AGC, AdamW_AGC, RMSprop_AGC
from torch.optim import SGD, Adam, AdamW, RMSprop
from tqdm import tqdm
import time
import json

batch_size = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
testset = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_optim(optim_name, parameters):
    if optim_name == 'adam_agc':
        return Adam_AGC(parameters, lr=0.001, clipping=0.16)
    elif  optim_name == 'adam':
        return Adam(parameters, lr=0.001)
    elif  optim_name == 'sgd_agc':
        return SGD_AGC(parameters, lr=0.01, clipping=0.16)
    elif  optim_name == 'sgd':
        return SGD(parameters, lr=0.01)
    elif  optim_name == 'rmsprop_agc':
        return RMSprop_AGC(parameters, lr=0.001, clipping=0.16)
    elif  optim_name == 'rmsprop':
        return RMSprop(parameters, lr=0.001)

optims = {
    'adam_agc' : {},
    'adam' : {},
    'sgd_agc' : {},
    'sgd' : {},
    'rmsprop_agc' : {},
    'rmsprop' : {} 
    }

for optim in optims:
    print("Current optimizer: " + optim)
    for run in tqdm(range(10)):

        net = Net().to(device=device)
        criterion = nn.CrossEntropyLoss()

        optimizer = get_optim(optim, net.parameters())

        optims[optim][run+1] = []

        for epoch in range(10): 
            t0 = time.time()

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                
                inputs, labels = data
                inputs = inputs.to(device=device)
                labels = labels.to(device=device)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss
            epoch_train_time = time.time() - t0
            epoch_loss = running_loss / len(trainloader)

            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images = images.to(device=device)
                    labels = labels.to(device=device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            epoch_accuracy = correct/total * 100

            optims[optim][run+1].append((float(epoch_loss.detach().cpu().numpy()), 
                    float(epoch_accuracy), epoch_train_time))


with open('performance.json', 'w') as fp:
    json.dump(optims, fp)