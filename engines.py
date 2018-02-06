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

    * check move is legal independent of :attr:`score`;

    * test compatibility with :class:`sunfish.Position`:

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

    * legal move tests

      >>> p2 = tools.parseFEN('7k/6Q1/5K2/8/8/8/8/8 b - - 0 1')
      >>> q2 = Superposition(*p2)
      >>> assert q2.board.split() == [ # *black plays "K"*
      ... '........',
      ... '........',
      ... '........',
      ... '........',
      ... '........',
      ... '..k.....',
      ... '.q......',
      ... 'K.......',
      ... ]
      >>> True not in map(q2.legal,q2.gen_moves()) # checkmate!
      True

    """

    def rotate(self):
        return Superposition(*super().rotate())

    def nullmove(self):
        return Superposition(*super().nullmove())

    def move(self,move):
        pos = Superposition(*super().move(move))
        pos.move_from = move
        return pos

    def gameover(self):
        return "K" not in self.board or "k" not in self.board

    def legal(self,move):
        """is `move` not ignore check?"""
        nextpos = self.move(move)
        captures = (nextpos.board[move[1]] for move in nextpos.gen_moves())
        return "k" not in captures

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
    print("{} v {}".format(white,black))
    for ply,pos,move in game(white,black,plies,secs,fen):
        if ply%2 == 0:
            print()
            pos.print()
            print(tools.renderFEN(pos))
            print("\a") # bell
        print('{}: {}'.format(ply+1,tools.mrender(pos,move)))
    print('{} resigned!'.format([white,black][(ply+1)%2]))
    return pos


class SunfishScore(object):
    """Encapsulation of `pos.score` and `pos.value` of :mod:`sunfish`

    * the example in :class:`Superposition`

      >>> p2 = tools.parseFEN('7k/6Q1/5K2/8/8/8/8/8 b - - 0 1')

    * **board score is anti-symmetric**

      >>> assert p2.score == -p2.rotate().score

    * moves and :meth:`Position.value`

      >>> p2.board.find("K"), p2.board.find("q")
      (91, 82)
      >>> tuple(p2.gen_moves())
      ((91, 81), (91, 92), (91, 82))
      >>> p2.value((91,82)) > p2.value((91,92)) > p2.value((91,81))
      True
      >>> p3 = p2.move((91,82)) # black captures white "q"

    * **score of a new position**

      >>> assert p3.score == -(p2.score + p2.value((91,82)))

    * checkmate

      >>> p3.board.find("K"), p3.board.find("k")
      (46, 37)
      >>> MATE_UPPER > p3.value((46,37)) > MATE_LOWER
      True
      >>> len(list(p3.gen_moves())) == 8
      True
      >>> max(map(p3.value,p3.gen_moves())) == p3.value((46,37))
      True

    """
    def __init__(self):
        self.lower = MATE_LOWER
        self.upper = MATE_UPPER
    def __call__(self,pos):
        return pos.score


class Engine(object):
    """Chess engine base class

    :meth:`score`:
        A function to evaluate a score for `pos`.

    :attr:`maxdepth`:
        Maximum search depth.

    """

    def __repr__(self):
        name = self.__class__.__name__
        if hasattr(self,'score'):
            name += "." + self.score.__class__.__name__
        if hasattr(self,'maxdepth'):
            name += ".maxdepth" + str(self.maxdepth)
        return name


class Sunfish(Searcher,Engine):
    """Sunfish MTDf-bi mix-in with :class:`Engine`

    """

    def __init__(self):
        super(Sunfish,self).__init__()


class Fool(Engine):
    """An engine with no strategy

    * make random legal moves;
    * return :mod:`sunfish` score of the corresponding random move;

    """

    def __init__(self):
        super(Fool,self).__init__()

    def search(self,pos,secs=NotImplemented):
        moves = [move for move in pos.gen_moves() if pos.legal(move)]
        if len(moves) == 0:
            return None,-MATE_UPPER # lost
        move = random.choice(moves)
        return move,pos.move(move).score


class Negamax(Engine):
    """Negamax

    * example, the same one as in :class:`SunfishScore`:

      >>> p2 = tools.parseFEN('7k/6Q1/5K2/8/8/8/8/8 b - - 0 1')
      >>> q2 = Superposition(*p2)
      >>> g = Negamax()
      >>> s = g.ordermoves(q2)
      >>> s,m = zip(*s)
      >>> m
      ((91, 81), (91, 92), (91, 82))
      >>> s[0] > s[1] > s[2]
      True

    """

    MAXDEPTH = 4

    def __init__(
            self,
            maxdepth = MAXDEPTH,
            score = SunfishScore()):
        super(Negamax,self).__init__()
        self.maxdepth = maxdepth
        self.score = score
        self.upper = self.score.upper
        self.lower = self.score.lower

    def ordermoves(self,pos):
        """return ``((score,move),...)`` in descending order of the scores"""
        moves = list(pos.gen_moves())
        scores = list(self.score(pos.move(move)) for move in moves)
        return sorted(zip(scores,moves),reverse=True)

    def negamax(self,pos,depth,alpha,beta):
        if depth == 0 or pos.gameover():
            return self.score(pos)
        maxscore = -self.upper
        for i,move in enumerate(pos.gen_moves()):
            nextpos = pos.move(move)
            score = -self.negamax(nextpos,depth-1,-beta,-alpha)
            if score >= maxscore:
                maxscore = score
                alpha = max(alpha,score)
                if depth == self.depth:
                    self.bestmove = move
        return maxscore

    def search(self,pos,secs=NotImplemented):
        self.bestmove = None
        timestart = time.time()
        self.depth = self.maxdepth
        score = self.negamax(pos,self.depth,-self.upper,self.upper)
        timespent = time.time() - timestart
        if self.bestmove and pos.legal(self.bestmove):
            return self.bestmove,score
        return None,score


if __name__ == '__main__':

    import sys
    import doctest
    print(doctest.testmod(optionflags=doctest.REPORT_ONLY_FIRST_FAILURE))
    scripts = doctest.script_from_examples(
        Superposition.__doc__ +
        SunfishScore.__doc__ +
        Negamax.__doc__
        )

    if sys.argv[0] != "":
        p = play(Fool(),Negamax())
