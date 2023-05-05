import torch
from torchvision import models
import numpy as np
import time
torch.manual_seed(1337)
np.random.seed(1337)

torch.torch.backends.cudnn.deterministic = True
torch.torch.backends.cudnn.benchmark = False
# probably non-deterministic came from here, MIOpen cannot set benchmark to False
# which uses non-deterministic conv kernels, and contained atomic-add kernels(AdaptiveAvgPool2d)
# according to doc, using below
# torch.use_deterministic_algorithms(True) should throw error when call AdaptiveAvgPool2d
# which implicit amd-rocm pytorch lacks support of such deterministic
torch.use_deterministic_algorithms(True)

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
model = models.resnet18(pretrained=True, progress=True)
model.eval()
model = model.to(device)
x = torch.ones(1, 3, 224, 224)
sum = 0
for it in range(0,100):
    x = x.to(device)
    res = model.forward(x)
min =10000000
max =0.0
avg =0.0
#long running
for it in range(0,1000):
    start = time.perf_counter_ns()
    x = x.to(device)
    res = model.forward(x)
    res = res.to('cpu')
    end = time.perf_counter_ns()
    cur = (end-start)/1000
    if max < cur:
        max = cur
    if min > cur:
        min = cur
    avg += cur
avg/=1000
print("[min, max, avg] = [{},{},{}] us".format(min,max,avg))

