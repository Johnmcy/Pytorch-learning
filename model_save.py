import torch
import  torchvision
from torch import nn
from torch.nn import Conv2d

vgg16 = torchvision.models.vgg16(pretrained=False)

# save method 1  structure and parameter
torch.save(vgg16, "vgg16_method1.pth")

# save method 2  parameter(officially recommend)
torch.save(vgg16.state_dict(), "vgg16_method2.pth")

# trap
class Chaoyang(nn.Module):
    def __init__(self):
        super(Chaoyang, self).__init__()
        self.conv1 = Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x

chaoyang = Chaoyang()
torch.save(chaoyang, "chaoyang_method1.pth")
