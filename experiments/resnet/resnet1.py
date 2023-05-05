import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class BasicDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(BasicDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size[0], stride[0], padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size[0], stride[1], padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels,
                               kernel_size[1], stride[0],bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.bn1(self.conv1(x))
        output = self.bn2(self.conv2(output))
        output1 = self.bn3(self.conv3(x))
        return F.relu(output1+output)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = BasicDownBlock(64, 128, [3, 1], [2, 1])
x = torch.ones(128, 64, 3, 3)
x = x.to(device)
model.eval()
model = model.to(device)

print("Input shape: {}".format(x.shape))
conv1_output = model.conv1(x)
print("conv1_output shape: {}".format(conv1_output.shape))
bn1_output = model.bn1(conv1_output)
print("bn1_output shape: {}".format(bn1_output.shape))
conv2_output = model.conv2(bn1_output)
print("conv2_output shape: {}".format(conv2_output.shape))
bn2_output = model.bn2(conv2_output)
print("bn2_output shape: {}".format(bn2_output.shape))
conv3_output = model.conv3(x)
print("conv3_output shape: {}".format(conv3_output.shape))
bn3_output = model.bn3(conv3_output)
print("bn3_output shape: {}".format(bn3_output.shape))

res = model.forward(x)
res = res.to('cpu')
out = res.detach().numpy()
out = out.astype(np.float32)
out.tofile("out_ref.dat")
