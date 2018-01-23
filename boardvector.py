#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sunfish import tools
import numpy as np

p0 = tools.parseFEN(tools.FEN_INITIAL)

piececounts = zip('PNBRQKpnbrqk',[8,2,2,2,9,1]*2)
indices = np.array([[i,j] for j in range(8) for i in range(8)])/8.0
null = [[-1.0,-1.0]]

def rfvrepr(board):
    """rank and file array of board in order and dimension as piececounts"""
    squares = np.array(list("".join(board.split()))) # strip spaces
    return np.concatenate([
        indices[squares == piece].tolist()
        + null*(counts - sum(squares == piece))
        for piece,counts in piececounts]).flatten()
