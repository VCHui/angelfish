#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sunfish import tools
from engines import legal
import numpy as np


piecetypes = 'PNBRQKpnbrqk'

def vrepr(pos):
    """
    encode pos into an array of ``12*64 = 768`` elements

      - square has a value 2 if occupied
      - square has a value 1 if possibly occupied
      - king square is 3 or 5 for possible left or right castling

    * Game opening

    >>> p0 = tools.parseFEN(tools.FEN_INITIAL)
    >>> v0 = vrepr(p0)

    * To input to a neural network (without normalization)
    >>> v = np.array(v0).flatten().astype(float)
    >>> len(v)
    768

    >>> v0[piecetypes.find("P")].reshape(8,8)
    array([[0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 0]])
    >>> v0[piecetypes.find("p")].reshape(8,8)
    array([[0, 0, 0, 0, 0, 0, 0, 0],
           [2, 2, 2, 2, 2, 2, 2, 2],
           [1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0]])

    * Black's turn, white checkmate!

    >>> p1 = tools.parseFEN('7k/6Q1/5K2/8/8/8/8/8 b - - 0 1')
    >>> "".join(p1.rotate().board.split()) == (
    ... '.......k'
    ... '......Q.'
    ... '.....K..'
    ... '........'
    ... '........'
    ... '........'
    ... '........'
    ... '........' )
    True
    >>> v1 = vrepr(p1)
    >>> v1[piecetypes.find("k")].reshape(8,8)
    array([[0, 0, 0, 0, 0, 0, 0, 2],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0]])
    >>> v1[piecetypes.find("K")].reshape(8,8)
    array([[0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 2, 1, 0],
           [0, 0, 0, 0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0]])
    >>> v1[piecetypes.find("Q")].reshape(8,8)
    array([[0, 0, 0, 0, 0, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 2, 1],
           [0, 0, 0, 0, 0, 0, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 1, 0]])

    """

    if pos.board.startswith("\n"): # is black
        pos = pos.rotate() # un-rotate to the  conventional orientation
    squares = np.array(tuple("".join(pos.board.split()))) # strip
    v = [(squares == piece)*2 for piece in piecetypes]
    v[piecetypes.find("K")] = (squares == "K")*(2**pos.wc[0] + 4**pos.wc[1])
    v[piecetypes.find("k")] = (squares == "k")*(2**pos.bc[0] + 4**pos.bc[1])
    moves = tuple(move for move in pos.gen_moves() if legal(pos,move))
    for i,j in moves:
        ipiece = piecetypes.find(pos.board[i])
        jsq = ((j//10) - 2)*8 + ((j%10) - 1)
        v[ipiece][jsq] = 1
    pos = pos.rotate()
    moves = tuple(move for move in pos.gen_moves() if legal(pos,move))
    for i,j in moves:
        ipiece = 6 + piecetypes.find(pos.board[i]) # lower case piece
        jsq = ((j//10) - 2)*8 + ((j%10) - 1) + 1
        v[ipiece][-jsq] = 1 # un-rotating jsq
    return v

if __name__ == '__main__':

    import doctest
    print(doctest.testmod(optionflags=doctest.REPORT_ONLY_FIRST_FAILURE))
    scripts = doctest.script_from_examples(vrepr.__doc__) # exec(scripts)
