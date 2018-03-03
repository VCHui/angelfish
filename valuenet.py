#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sunfish import MATE_LOWER, MATE_UPPER
from sunfish import tools
from engines import Superposition
from posvector import brepr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class ValueNet(nn.Module):
    """

    >>> FEN_MATE5 = '1k6/8/2K5/3R4/8/8/8/8 w - - 0 1' # 5 plies to checkmate
    >>> q5 = Superposition(*tools.parseFEN(FEN_MATE5))
    >>> net = ValueNet()
    >>> net
    ValueNet (
      (fc0): Linear (64 -> 256)
      (fc1): Linear (256 -> 64)
      (fc2): Linear (64 -> 1)
      (delta): MSELoss (
      )
    )

    >>> [ par.data.shape for par in net.parameters() ] == [
    ... torch.Size([256, 64]),
    ... torch.Size([256]),
    ... torch.Size([64, 256]),
    ... torch.Size([64]),
    ... torch.Size([1, 64]),
    ... torch.Size([1]),
    ... ]
    True

    >>> net.zero_all_parameters()
    >>> torch.equal(net(Variable(torch.ones(64))).data,torch.zeros(1))
    True

    >>> net.eval(q5)
    0.0

    """

    LEARNING_RATE = 0.5
    lower = MATE_LOWER
    upper = MATE_UPPER

    def __init__(self,lr=LEARNING_RATE):
        super(ValueNet,self).__init__()
        self.fc0 = nn.Linear(64,256) # 4*64
        self.fc1 = nn.Linear(256,64)
        self.fc2 = nn.Linear(64,1)
        self.optim = torch.optim.Adam(self.parameters(),lr)
        self.delta = nn.MSELoss()
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.cuda()

    def zero_all_parameters(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                m.weight.data.zero_()
                m.bias.data.zero_()

    def randomize_all_parameters(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                m.weight.data.uniform_(-1,1)
                m.bias.data.uniform_(-1,1)

    def forward(self,x):
        y = F.relu(self.fc0(x))
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        return MATE_UPPER * F.tanh(y)

    def psi(self,pos):
        """return the feature vector of `pos`"""
        x = torch.FloatTensor(brepr(pos)/8.0)
        return Variable(x,requires_grad=True)

    def eval(self,pos):
        """return the score for `pos`"""
        out = self.__call__(self.psi(pos)) # torch.autograd.variable.Variable
        out = out.data.numpy()
        return out[0]

    def update(self,pos,target):
        self.optim.zero_grad()
        target = Variable(torch.FloatTensor([float(target)]))
        value = self.__call__(self.psi(pos))
        delta = self.delta(value,target)
        delta.backward()
        self.optim.step()
        return delta.data.numpy()[0]


if __name__ == '__main__':

    import doctest
    print(doctest.testmod(optionflags=doctest.REPORT_ONLY_FIRST_FAILURE))
