import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=1)

class Chaoyang(nn.Module):
    def __init__(self):
        super(Chaoyang, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )


    def forward(self, x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
chaoyang = Chaoyang()
optim = torch.optim.SGD(tudui.parameters(),lr=0.01)

for epoch in range(5):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = chaoyang(imgs)
        result_loss = loss(outputs, targets)
        # result_loss.backward()
        optim.zero_grad() # zero
        result_loss.backward() # get each point's gradient
        optim.step() # optimize every parameters
        running_loss = running_loss + result_loss
    print(running_loss)

