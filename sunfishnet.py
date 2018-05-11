#!/usr/bin/env python

"""

sunfish.pst implemented as torch.nn.module


"""

from sunfish import pst, MATE_UPPER
from sunfish import tools
from torch import nn, FloatTensor
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np


PIECETYPES = 'KQRBNP'


def brepr(pos):
    """return the concatenated binary position representation of
    the pieces ``KQRBNP`` on the serialized chess board;

    """
    sq = np.array(list("".join(pos.board.split())))
    b = [ (sq == piece) for piece in PIECETYPES ]
    return np.concatenate(b).astype(float)


def abrepr(pos):
    """return the anti-symmetric `bvec` of ``pos``;

    >>> p = tools.parseFEN(tools.FEN_INITIAL)

    >>> b = brepr(p)
    >>> b.shape == (6*64,) == (384,)
    True

    >>> all(np.packbits(b.astype(int)) == np.array([
    ...   0,   0,   0,   0,   0,   0,   0,   8, # K
    ...   0,   0,   0,   0,   0,   0,   0,  16, # Q
    ...   0,   0,   0,   0,   0,   0,   0, 129, # R 1 + 128
    ...   0,   0,   0,   0,   0,   0,   0,  36, # B 4 +  32
    ...   0,   0,   0,   0,   0,   0,   0,  66, # N 2 +  64
    ...   0,   0,   0,   0,   0,   0, 255,   0, # P
    ... ]))
    True

    >>> c = brepr(p.rotate())
    >>> all(np.packbits(c.astype(int)) == np.array([
    ...   0,   0,   0,   0,   0,   0,   0,  16, # K
    ...   0,   0,   0,   0,   0,   0,   0,   8, # Q
    ...   0,   0,   0,   0,   0,   0,   0, 129, # R 1 + 128
    ...   0,   0,   0,   0,   0,   0,   0,  36, # B 4 +  32
    ...   0,   0,   0,   0,   0,   0,   0,  66, # N 2 +  64
    ...   0,   0,   0,   0,   0,   0, 255,   0, # P
    ... ]))
    True

    >>> a = abrepr(p)
    >>> all(a.astype(int) == b.astype(int) - c.astype(int))
    True

    """
    return brepr(pos) - brepr(pos.rotate())


class SunfishNET(nn.Module):
    """

    * sunfish.pst implemented as torch.nn.module;
    * input is an anti-symmetric bitboard representation of a board;
    * has member functions to accept input of :obj:`pos` and `fen`;


    """

    N = len(PIECETYPES)*64

    def __init__(self):
        super(SunfishNET,self).__init__()
        self.pst = nn.Linear(self.N,1,bias=False)
        self.init()

    def forward(self,x):
        y = self.pst(x)
        return F.hardtanh(y,-MATE_UPPER,MATE_UPPER)

    def init(self):
        """assign sunfish ``pst`` to ``self.pst``:"""
        for i,piecetype in enumerate(PIECETYPES):
            n = i*64
            table = np.array(pst[piecetype],dtype=float)
            table.shape = (12,10)
            table = table[2:-2,1:-1] # remove sunfish board paddings
            self.pst.weight.data[0,n:n+64] = FloatTensor(table.flatten())

    def aeval(self,a):
        """return the evaluation of `a` of `abrepr(pos)`:"""
        a.shape = (1,self.N)
        a = Variable(FloatTensor(a))
        y = self.forward(a)
        return y.data[0][0]

    def peval(self,pos):
        """return the evaluation of `pos`:"""
        return self.aeval(abrepr(pos))

    def feval(self,fen):
        """return the evaluation of `fen`:"""
        return self.peval(tools.parseFEN(fen))

    def verify(self,fen):
        """print a verification of the forward output against ``pos.score``:"""
        pos = tools.parseFEN(fen)
        score = int(self.peval(pos))
        assertion = score == pos.score
        print('{:>7} {} {!r}'.format(score,assertion,fen))

    def printpst(self,indent=""):
        """print the pst as csv:"""
        csv = (indent + '{:7},'*8).format
        label = '   #+{}{}'.format
        bigtable = self.pst.weight.data.numpy().astype(int)
        bigtable.shape = (len(PIECETYPES),8,8)
        for i in range(len(PIECETYPES)):
            for r in range(8):
                print(csv(*tuple(bigtable[i,r]))
                          + label(r+1,PIECETYPES[i]))

    def __repr__(self):
        lines = super(SunfishNET,self).__repr__()
        return "# " + "# ".join(lines.splitlines(True))



if __name__ == '__main__':

    import sys, os, doctest

    sunfishnet = SunfishNET()

    if sys.argv[0] == "": # if the python session is inside an emacs buffer
        print(doctest.testmod(optionflags=doctest.REPORT_ONLY_FIRST_FAILURE))
        print(sunfishnet)
        with open('sunfish/tests/mate1.fen',"r") as f:
            for fen in f:
                  sunfishnet.verify(fen.strip())
    else:
        if len(sys.argv) == 1:
            print('usage:',sys.argv[0],'[fen]')
        else:
            sunfishnet.verify(sys.argv[1])
