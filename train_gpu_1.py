import torch.optim
import torchvision
from torch.utils.tensorboard import SummaryWriter

# from model import *
from torch import nn
from torch.utils.data import DataLoader

# prepare dataset
train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("The length of train data is:{}".format(train_data_size))
print("The length of test data is:{}".format(test_data_size))

# use DataLoader to load dataset
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# create network model
class Chaoyang(nn.Module):
    def __init__(self):
        super(Chaoyang, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x
chaoyang = Chaoyang()
chaoyang = chaoyang.cuda()

# create loss function
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

# optimizer
# learning_rate = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(chaoyang.parameters(), lr=learning_rate)

# setup parameters
# record train times
total_train_step = 0
# record test times
total_test_step = 0
# epoch
epoch = 10

# add tensorboard
writer = SummaryWriter("logs_train")


for i in range(epoch):
    print("-----------the {}th training starts-----------".format(i+1))

    # start training
    chaoyang.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = chaoyang(imgs)
        loss = loss_fn(outputs, targets)

        # optimize model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        total_train_step += 1
        if total_train_step % 100 == 0:
            print("training times: {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # test step starts
    chaoyang.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = chaoyang(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("total test data loss: {}".format(total_test_loss))
    print("totoal test data accuracy: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(chaoyang, "chaoyang_{}.pth".format(i))
    # torch.save(chaoyang.state_dict(), "chaoyang_{}.pth".format(i))  # officially recommend
    print("model saved")

writer.close()









