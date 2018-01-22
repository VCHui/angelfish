#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sunfish import Searcher, print_pos
from sunfish import MATE_LOWER, MATE_UPPER
from sunfish import tools
import random


def legal(pos,move):
    """test legality for move on pos;"""
    nextpos = pos.move(move)
    nextscores = map(nextpos.value,nextpos.gen_moves())
    return all(score < MATE_LOWER for score in nextscores)

def game(white,black,plies=200,secs=1,fen=tools.FEN_INITIAL):
    """return a generator of moves of a game;"""
    pos = tools.parseFEN(fen)
    engines = [white,black]
    for ply in range(plies):
        engine = engines[ply%2]
        move,score = engine.search(pos,secs)
        if move is None:
            break
        if not legal(pos,move):
            print('{} {} not illegal!'.format(
                engine.name,tools.mrender(pos,move)))
            break
        yield ply,pos,move
        pos = pos.move(move)

def play(white,black,plies=200,secs=1,fen=tools.FEN_INITIAL):
    """play the :func:`game` with outputs;"""
    print('*: {}; {};'.format(white.name,black.name))
    for ply,pos,move in game(white,black,plies,secs,fen):
        if ply%2 == 0:
            print()
            print_pos(pos)
            print(tools.renderFEN(pos))
            print('{}-{}'.format(ply+1,ply+2),end=": ")
        print(tools.mrender(pos,move),end="; ")
    print('\n{} resigned!'.format([white,black][(ply+1)%2].name))
    return pos


class Engine(object):
    """

    Chess engine base class


    """

    @property
    def name(self):
        return self.__class__.__name__


class Fool(Engine):
    """

    An engine with no strategy:- make random legal moves


    """

    def __init__(self):
        super(Fool,self).__init__()

    def search(self,pos,secs=NotImplemented):
        moves = [move for move in pos.gen_moves() if legal(pos,move)]
        if len(moves) == 0:
            return None,-MATE_UPPER # lost
        move = random.choice(moves)
        return move,pos.value(move)


class Sunfish(Searcher,Engine):
    """

    Sunfish MTDf-bi mix-in with :class:`Engine`


    """

    def __init__(self):
        super(Sunfish,self).__init__()


if __name__ == '__main__':

    import sys

    if sys.argv[0] != "":
        p = play(Fool(),Sunfish())
