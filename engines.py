#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

* Negamax and Minimax are equivalent

  >>> m = Minimax(maxdepth=5)
  >>> m
  Minimax.SunfishPolicy.maxdepth5
  >>> n = Negamax(maxdepth=5)
  >>> n
  Negamax.SunfishPolicy.maxdepth5

  >>> FEN_MATE5 = '1k6/8/2K5/3R4/8/8/8/8 w - - 0 1' # 5 plies to checkmate
  >>> q5 = Superposition(*tools.parseFEN(FEN_MATE5))
  >>> assert q5.board.split() == [
  ... '.k......',
  ... '........',
  ... '..K.....',
  ... '...R....',
  ... '........',
  ... '........',
  ... '........',
  ... '........',
  ... ]

  >>> mbest = m.maximize(q5,m.maxdepth)
  >>> nbest = n.negamax(q5,n.maxdepth,-n.upper,n.upper)
  >>> nbest == mbest
  True
  >>> bestbranch,bestscore = nbest
  >>> bestscore > MATE_LOWER
  True
  >>> len(bestbranch) == 5
  True
  >>> q = q5
  >>> for move in bestbranch:
  ...    q = q.move(move)
  >>> 'K' in q.board
  False

"""

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

      >>> FEN_MATE0 = '7k/6Q1/5K2/8/8/8/8/8 b - - 0 1'
      >>> q2 = Superposition(*tools.parseFEN(FEN_MATE0))
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
      >>>
      >>> not any(map(q2.legal,q2.gen_moves())) # checkmate!
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
        """return `move` if not ignore check else None;"""
        nextpos = self.move(move)
        captures = (nextpos.board[move[1]] for move in nextpos.gen_moves())
        if "k" not in captures:
            return move
        return None

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
        if pos.legal(move) is None:
            print('{} {} not illegal!'.format(
                engine,tools.mrender(pos,move)))
            break
        yield ply,pos,move,score
        pos = pos.move(move)


def play(white,black,plies=200,secs=1,fen=tools.FEN_INITIAL):
    """play the :func:`game` with outputs;"""
    print("{} v {}".format(white,black))
    for ply,pos,move,score in game(white,black,plies,secs,fen):
        if ply%2 == 0:
            print()
            pos.print()
            print(tools.renderFEN(pos))
            print("\a") # bell
            print(ply+1,end=":")
        print("",tools.mrender(pos,move),end="")
    print("#")
    return pos


class SunfishPolicy(object):
    """Encapsulation of `pos.score` and `pos.value` of :mod:`sunfish`

    * use ``FEN_MATE0`` as in :class:`Superposition`

      >>> FEN_MATE0 = '7k/6Q1/5K2/8/8/8/8/8 b - - 0 1'
      >>> q2 = Superposition(*tools.parseFEN(FEN_MATE0))

    * **board score is anti-symmetric**

      >>> assert q2.score == -q2.rotate().score

    * moves and :meth:`Position.value`

      >>> q2.board.find("K"), q2.board.find("q")
      (91, 82)
      >>> tuple(q2.gen_moves())
      ((91, 81), (91, 92), (91, 82))
      >>> q2.value((91,82)) > q2.value((91,92)) > q2.value((91,81))
      True
      >>> q3 = q2.move((91,82)) # black captures white "q"

    * **score of a new position**

      >>> assert q3.score == -(q2.score + q2.value((91,82)))

    * checkmate

      >>> q3.board.find("K"), q3.board.find("k")
      (46, 37)
      >>> MATE_UPPER > q3.value((46,37)) > MATE_LOWER
      True
      >>> len(list(q3.gen_moves())) == 8
      True
      >>> max(map(q3.value,q3.gen_moves())) == q3.value((46,37))
      True

    """
    def __init__(self):
        self.upper = MATE_UPPER
    def eval(self,pos):
        return pos.score


class Engine(object):
    """Chess engine base class

    :attr:`policy`:
        An instance of a policy to evaluate a score for `pos`.

    :attr:`maxdepth`:
        Maximum search depth.

    """

    def __repr__(self):
        name = self.__class__.__name__
        if hasattr(self,'policy'):
            name += "." + self.policy.__class__.__name__
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
        moves = [ move for move in pos.gen_moves() if pos.legal(move) ]
        if len(moves) == 0:
            return None,-MATE_UPPER # lost
        move = random.choice(moves)
        return move,pos.move(move).score


class Negamax(Engine):
    """Negamax

    """

    MAXDEPTH = 3

    def __init__(
            self,
            maxdepth = MAXDEPTH,
            policy = SunfishPolicy()):
        super(Negamax,self).__init__()
        self.maxdepth = maxdepth
        self.policy = policy
        self.upper = self.policy.upper

    def negamax(self,pos,depth,alpha,beta):
        if depth == 0 or pos.gameover():
            return [],self.policy.eval(pos)
        bestbranch,maxscore = [],-self.upper
        for i,move in enumerate(pos.gen_moves()):
            nextpos = pos.move(move)
            branch,score = self.negamax(nextpos,depth-1,-beta,-alpha)
            score = -score # nega
            if score >= maxscore:
                bestbranch,maxscore = [move,]+branch,score
                alpha = max(alpha,score)
        return bestbranch,maxscore

    def search(self,pos,secs=NotImplemented):
        timestart = time.time()
        bestbranch,bestscore = self.negamax(
            pos,self.maxdepth,-self.upper,self.upper)
        timespent = time.time() - timestart
        return pos.legal(bestbranch[0]),bestscore


class Minimax(Engine):
    """Minimax

    """

    MAXDEPTH = 3

    def __init__(
            self,
            maxdepth = MAXDEPTH,
            policy = SunfishPolicy()):
        super(Minimax,self).__init__()
        self.maxdepth = maxdepth
        self.policy = policy
        self.upper = self.policy.upper

    def maximize(self,pos,depth):
        if depth == 0 or pos.gameover():
            return [],self.policy.eval(pos)
        bestbranch,maxscore = [],-self.upper
        for move in pos.gen_moves():
            nextpos = pos.move(move)
            branch,score = self.minimize(nextpos,depth-1)
            if score >= maxscore:
                bestbranch,maxscore = [move,]+branch,score
        return bestbranch,maxscore

    def minimize(self,pos,depth):
        # sunfish convention puts the curent player white
        # black requires score sign reversion
        if depth == 0 or pos.gameover():
            return [],-self.policy.eval(pos)
        bestbranch,minscore = [],self.upper
        for move in pos.gen_moves():
            nextpos = pos.move(move)
            branch,score = self.maximize(nextpos,depth-1)
            if score <= minscore:
                bestbranch,minscore = [move,]+branch,score
        return bestbranch,minscore

    def search(self,pos,secs=NotImplemented):
        timestart = time.time()
        bestbranch,bestscore = self.maximize(pos,self.maxdepth)
        timespent = time.time() - timestart
        return pos.legal(bestbranch[0]),bestscore


enginedict = dict(
    Sunfish=Sunfish,
    Fool=Fool,
    Minimax=Minimax,
    Negamax=Negamax,
    )


if __name__ == '__main__':

    import sys
    import doctest
    print(doctest.testmod(optionflags=doctest.REPORT_ONLY_FIRST_FAILURE))
    scripts = doctest.script_from_examples(__doc__)

    if sys.argv[0] != "":
        if len(sys.argv) == 3:
            white = enginedict[sys.argv[1]]()
            black = enginedict[sys.argv[2]]()
            p = play(white,black)
