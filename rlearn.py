#!/usr/bin/env pypy3
# -*- coding: utf-8 -*-

"""

reinforcement learning


"""

from sunfish import tools
from engines import Superposition, AlphaBeta
from valuenet import ValueNet
import sys, os


class TreeStrapNegamax(object):
    """An implementation of TreeStrap(Minimax)

    .. _[1]: "Bootstrapping from Game Tree Search", Veness J, (2009)

    """

    def __init__(self):
        self.heuristics = ValueNet()
        self.heuristics.randomize_all_parameters()
        self.negamax = AlphaBeta(maxdepth=3,policy=self.heuristics)
        self.negamax.pruning = False
        assert self.heuristics == self.negamax.policy
        assert self.heuristics.eval == self.negamax.policy.eval
        assert self.heuristics.upper == self.negamax.upper
        self.heuristics.randomize_all_parameters()

    def backup(self,pos,depth):
        entry = self.negamax.tt.get(pos)
        if entry is None:
            return
        target = entry.score
        nodes = []
        for move in self.negamax.getpv(pos):
            nodes.append(pos)
            if move is None: # gameover
                target = -self.heuristics.upper
                break
            pos = pos.move(move)
        for pos in nodes:
            self.deltasum += self.heuristics.update(pos,target)

    def walktree(self,pos,depth):
        self.backup(pos,depth)
        if depth > 0:
            for move in pos.gen_moves():
                nextpos = pos.move(move)
                self.walktree(nextpos,depth-1)

    def selflearn(self,pos):
        self.deltasum = 0
        while not pos.gameover():
            move,score = self.negamax.search(pos,secs=3600)
            self.walktree(pos,depth=self.negamax.maxdepth)
            if move is None:
                break
            pos = pos.move(move)
        print('delta',self.deltasum)


if __name__ == '__main__':

    import doctest
    print(doctest.testmod(optionflags=doctest.REPORT_ONLY_FIRST_FAILURE))

    FEN_MATE5 = '1k6/8/2K5/3R4/8/8/8/8 w - - 0 1' # 5 plies to checkmate
    q5 = Superposition(*tools.parseFEN(FEN_MATE5))
    t = TreeStrapNegamax()
    a = AlphaBeta()
