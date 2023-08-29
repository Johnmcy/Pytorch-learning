from torch import nn
import torch


class Chaoyang(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


chaoyang = Chaoyang()
x = torch.tensor(1.0)
output = chaoyang(x)
print(output)
