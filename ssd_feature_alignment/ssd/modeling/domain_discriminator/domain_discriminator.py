import torch
import torch.nn as nn
from torch.autograd import Function

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class DomainDiscriminator(nn.Module):
    def __init__(self):
        super(DomainDiscriminator, self).__init__()

        self.layer1 = self._make_layer(512, 256, kernel_size=3, stride=3, padding=1)#[256,13,13]
        self.layer2 = self._make_layer(256, 128, kernel_size=3, stride=3, padding=1)#[128,5,5]
        self.layer3 = nn.Linear(3200, 2048)
        self.layer4 = nn.Linear(2048, 1024)
        self.layer5 = nn.Linear(1024, 512)
        self.layer6 = nn.Linear(512, 256)
        self.layer7 = nn.Linear(256, 128)
        self.layer8 = nn.Linear(128, 64)
        self.layer9 = nn.Linear(64, 32)
        self.layer10 = nn.Linear(32, 16)
        self.layer11 = nn.Linear(16, 8)
        self.layer12 = nn.Linear(8, 4)
        self.layer13 = nn.Linear(4, 2)
        self.layer14 = nn.Linear(2, 1)
        self.layer15 = nn.Sigmoid()

    def _make_layer(self, in_nc, out_nc, kernel_size, stride, padding):
        block = [nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding),
                 nn.BatchNorm2d(out_nc)]
        return nn.Sequential(*block)

    def forward(self, x):
        #x = ReverseLayerF.apply()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(torch.reshape(x, [x.size(0), 3200]))
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        return x
