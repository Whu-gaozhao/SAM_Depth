import torch
import time
import torchvision
from thop import profile
from torchstat import stat
from model.UNetFormer import UNetFormer as UNetFormer

print('==> building model..')
net = UNetFormer(num_classes=6).cuda()
input = torch.randn(10,4,256,256).cuda()
# input = input.to(device)
flops,params = profile(net,(input,))
print('flops:%.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))


torch.cuda.synchronize()
start = time.time()
result = net(input)
torch.cuda.synchronize()
end = time.time()
print('infer_time',end-start)


