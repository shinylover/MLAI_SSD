
import torch
import torch.nn as nn
from torch.autograd import Variable

creterion = nn.BCELoss()

def domain_loss(input, real=True):
    real_label = Variable(torch.ones(input.size(0))).cuda()
    fake_label = Variable(torch.zeros(input.size(0))).cuda()
    if real:
        loss = creterion(input, real_label)
    else:
        loss = creterion(input, fake_label)
    return loss