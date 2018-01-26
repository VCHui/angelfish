#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sunfish import Position, print_pos
from sunfish import Searcher, MATE_LOWER, MATE_UPPER
from sunfish import tools
import time
import random


class Superposition(Position):
    """Extended Position class:

    * meth:`rotate`, meth:`nullmove`, and
      meth:`move` return Superposition instances;

    >>> p0 = tools.parseFEN(tools.FEN_INITIAL)
    >>> q0 = Superposition(*p0)
    >>> assert issubclass(Superposition,Position)
    >>> assert isinstance(q0,Superposition)
    >>> assert tuple(q0) == tuple(p0)
    >>> assert isinstance(q0.rotate(),Superposition)
    >>> assert tuple(q0.rotate()) == tuple(p0.rotate())
    >>> assert isinstance(q0.nullmove(),Superposition)
    >>> assert tuple(q0.nullmove()) == tuple(p0.nullmove())
    >>> assert tuple(q0.gen_moves()) == tuple(p0.gen_moves())
    >>> assert all(map(q0.legal,q0.gen_moves()))
    >>> q1 = q0.move(q0.gen_moves().send(None))
    >>> assert isinstance(q1,Superposition)

    """

    def rotate(self):
        return Superposition(*super().rotate())

    def nullmove(self):
        return Superposition(*super().nullmove())

    def move(self,move):
        pos = Superposition(*super().move(move))
        pos.move_from = move
        return pos

    def legal(self,move):
        """is `move` not ignore check?"""
        nextpos = self.move(move)
        nextscores = map(nextpos.value,nextpos.gen_moves())
        return all(score < MATE_LOWER for score in nextscores)

    def print(self):
        print_pos(self)


def game(white,black,plies=200,secs=1,fen=tools.FEN_INITIAL):
    """return a generator of moves of a game;"""
    pos = Superposition(*tools.parseFEN(fen))
    engines = [white,black]
    for ply in range(plies):
        engine = engines[ply%2]
        move,score = engine.search(pos,secs)
        if move is None:
            break
        if not pos.legal(move):
            print('{} {} not illegal!'.format(
                engine,tools.mrender(pos,move)))
            break
        yield ply,pos,move
        pos = pos.move(move)

def play(white,black,plies=200,secs=1,fen=tools.FEN_INITIAL):
    """play the :func:`game` with outputs;"""
    print('*: {}; {};'.format(white,black))
    for ply,pos,move in game(white,black,plies,secs,fen):
        if ply%2 == 0:
            print()
            pos.print()
            print(tools.renderFEN(pos))
            print('{}-{}'.format(ply+1,ply+2),end=": ")
        print(tools.mrender(pos,move),end="; ")
    print('\n{} resigned!'.format([white,black][(ply+1)%2]))
    return pos


class Engine(object):
    """

    Chess engine base class


    """

    MAXDEPTH = 4

    def __repr__(self):
        name = self.__class__.__name__
        scorefn = self.scorefn.__name__ if hasattr(self,'scorefn') else ""
        maxdepth = str(self.maxdepth) if hasattr(self,'maxdepth') else ""
        return ".".join([name,scorefn,maxdepth])


    @staticmethod
    def sunfishscorefn(pos):
        """just return the sunfish pos.score:

        .. code-block:: python

           pos_next.score == -(pos.score + pos.value(move))


        * examples of sunfish scoring method

          >>> # black's turn, white checkmate!
          >>> p1 = tools.parseFEN('7k/6Q1/5K2/8/8/8/8/8 b - - 0 1')
          >>> assert p1.board.split() == [ # sunfish rotated board for black
          ... '........',
          ... '........',
          ... '........',
          ... '........',
          ... '........',
          ... '..k.....',
          ... '.q......',
          ... 'K.......',
          ... ]
          >>> p1.board.find("K"), p1.board.find("q")
          (91, 82)
          >>> tuple(p1.gen_moves())
          ((91, 81), (91, 92), (91, 82))
          >>> p1.value((91,82)) > p1.value((91,92))
          True
          >>> p2 = p1.move((91,82)) # black captures white "q"
          >>> assert p2.board.split() == [
          ... '........',
          ... '......k.',
          ... '.....K..',
          ... '........',
          ... '........',
          ... '........',
          ... '........',
          ... '........',
          ... ]
          >>> p2.score == -(p1.score + p1.value((91,82)))
          True
          >>> p2.board.find("K"), p2.board.find("k")
          (46, 37)
          >>> MATE_UPPER > p2.value((46,37)) > MATE_LOWER
          True
          >>> len(list(p2.gen_moves())) == 8
          True
          >>> max(map(p2.value,p2.gen_moves())) == p2.value((46,37))
          True

        """
        return pos.score

    @staticmethod
    def randomsunfishscore(pos):
        return random.randint(-MATE_LOWER,MATE_LOWER)

    @staticmethod
    def randomscore(pos):
        return random.random()


class Sunfish(Searcher,Engine):
    """

    Sunfish MTDf-bi mix-in with :class:`Engine`


    """

    def __init__(self):
        super(Sunfish,self).__init__()


class Fool(Engine):
    """

    An engine with no strategy:- make random legal moves


    """

    def __init__(self):
        super(Fool,self).__init__()

    def search(self,pos,secs=NotImplemented):
        moves = [move for move in pos.gen_moves() if pos.legal(move)]
        if len(moves) == 0:
            return None,-MATE_UPPER # lost
        move = random.choice(moves)
        return move,pos.value(move)


class Negamax(Engine):
    """

    Negamax with prunning


    """

    def __init__(self,maxdepth=Engine.MAXDEPTH,scorefn=None):
        super(Negamax,self).__init__()
        self.maxdepth = maxdepth
        if scorefn is None:
            scorefn = Engine.sunfishscorefn
        self.scorefn = scorefn

    def recursion(self,pos,depth,alpha,beta,child=-1):
        """recursion method for :class:`Negamax`:"""
        return self.negamax(pos,depth,alpha,beta) # dropped child index

    def negamax(self,pos,depth,alpha=-MATE_UPPER,beta=MATE_UPPER):
        if depth == 0 or abs(pos.score) >= MATE_LOWER:
            return self.scorefn(pos)
        bestscore = -MATE_UPPER
        for child,move in enumerate(pos.gen_moves()):
            nextpos = pos.move(move)
            score = -self.recursion(nextpos,depth-1,-beta,-alpha,child)
            if score >= bestscore:
                bestscore = score
                if depth == self.depth: # at root
                    self.bestmove = move
                alpha = max(alpha,score)
        return bestscore

    def search(self,pos,secs=1):
        """a naive iterative deepening search;"""
        self.depth = self.maxdepth
        self.bestmove = None
        score = -MATE_UPPER
        timestart = time.time()
        score = self.negamax(pos,self.depth)
        timespent = time.time() - timestart
        if self.bestmove and pos.legal(self.bestmove):
            return self.bestmove,score
        return None,score


if __name__ == '__main__':

    import sys
    import doctest
    print(doctest.testmod(optionflags=doctest.REPORT_ONLY_FIRST_FAILURE))
    scripts = doctest.script_from_examples(
        Superposition.__doc__ + Engine.sunfishscorefn.__doc__)

    if sys.argv[0] != "":
        p = play(
            Negamax(scorefn=Engine.randomsunfishscore,maxdepth=1),
            Negamax())
