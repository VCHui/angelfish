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

  >>> m.nodes,m.showsearch = 0,0
  >>> n.nodes,n.showsearch = 0,0
  >>> mbest = m.maximize(q5,m.maxdepth)
  >>> mbest == n.negamax(q5,n.maxdepth,-n.upper,n.upper)
  True
  >>> bestbranch,bestscore = mbest
  >>> bestscore > MATE_LOWER
  True
  >>> len(bestbranch) == 5+1 and bestbranch[-1] == None
  True
  >>> q = q5
  >>> for move in bestbranch[:-1]:
  ...    q = q.move(move)
  >>> 'K' in q.board
  False
  >>> bestmove = bestbranch[0]
  >>> assert q5.move(bestmove).board.split() == [
  ... '........',
  ... '........',
  ... '........',
  ... '........',
  ... '.......r',
  ... '.....k..',
  ... '........',
  ... '......K.',
  ... ]

* AlphaBeta and Negamax should produce the same outcomes for `q5`

  >>> a = AlphaBeta()
  >>> a.showsearch = 0
  >>> a
  AlphaBeta.SunfishPolicy.maxdepth15
  >>> assert a.search(q5) == (bestmove,bestscore)
  >>> abest = a.tt.get(q5)
  >>> assert abest.score == bestscore
  >>> assert abest.branch == bestbranch

"""

from sunfish import Position, print_pos
from sunfish import Searcher, MATE_LOWER, MATE_UPPER
from sunfish import LRUCache, TABLE_SIZE
from sunfish import tools
from collections import OrderedDict, namedtuple
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
        # return "K" not in self.board or "k" not in self.board
        return "K" not in self.board # "k" was captured on last ply;

    def legal(self,move):
        """return `move` if not ignore check else None;"""
        nextpos = self.move(move)
        captures = (nextpos.board[move[1]] for move in nextpos.gen_moves())
        if "k" not in captures:
            return move
        return None

    def print(self,variation=[]):
        """pretty print the chessboard or chessboards for `variation`;"""
        pos = self
        print_pos(pos)
        for move in variation:
            if move is None:
                break
            pos = pos.move(move)
            print_pos(pos)

    def mrender(self,move):
        """return the algebraic notation for `move` or `variation`;"""
        if isinstance(move,tuple):
            return tools.mrender(self,move)
        variation = move
        pos,moves = self,[]
        for move in variation:
            if move is None:
                moves.append(None)
                break
            moves.append(tools.mrender(pos,move))
            pos = pos.move(move)
        return moves

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
                engine,pos.mrender(move)))
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
            print(ply//2+1,end=":")
        print("",pos.mrender(move),end="")
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

    def __init__(self,showsearch=1):
        super(Sunfish,self).__init__()
        self.showsearch = showsearch

    def search(self,pos,secs=1):
        """Override :meth:`Searcher.search` to perform `go` as in `xboard`"""
        timestart = time.time()
        for _ in self._search(pos):
            timespent = time.time() - timestart
            if self.showsearch:
                self.show(pos,self.depth,timespent)
            if timespent > secs:
                break
        entry = self.tp_score.get((pos,self.depth,True))
        return self.tp_move.get(pos),entry.lower

    def show(self,pos,depth,timespent):
        entry = self.tp_score.get((pos,depth,True))
        pv = ",".join(pos.mrender(self.getpv(pos)))
        if depth == 1:
            print()
        print('${} {}ms {}/{} {}: {}'.format(
            depth,int(timespent*1000),
            self.nodes,len(self.tp_move.od),
            entry.lower,pv))

    def getpv(self,pos):
        """return `pv` from `pos` using :attr:`tp_move`

        - a simplified re-implementation of :func:`tools.pv`

        """
        pv = OrderedDict()
        while True:
            move = self.tp_move.get(pos)
            if move is None:
                break
            pos = pos.move(move)
            if pos in pv:
                raise Warning((move,'loop'))
                break
            pv[pos] = move
        return list(pv.values())


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


class Minimax(Engine):
    """Minimax

    """

    MAXDEPTH = 4

    def __init__(
            self,
            maxdepth = MAXDEPTH,
            policy = SunfishPolicy(),
            showsearch = 1):
        super(Minimax,self).__init__()
        self.maxdepth = maxdepth
        self.policy = policy
        self.upper = self.policy.upper
        self.showsearch = showsearch

    def maximize(self,pos,depth):
        self.nodes += 1
        if pos.gameover():
            return [None,],self.policy.eval(pos)
        if depth == 0:
            return [],self.policy.eval(pos)
        bestbranch,maxscore = [],-self.upper
        for move in pos.gen_moves():
            nextpos = pos.move(move)
            branch,score = self.minimize(nextpos,depth-1)
            if score >= maxscore:
                bestbranch,maxscore = [move,]+branch,score
        return bestbranch,maxscore

    def minimize(self,pos,depth):
        # sunfish convention puts the current player white
        # black requires score sign reversion
        self.nodes += 1
        if pos.gameover():
            return [None,],-self.policy.eval(pos)
        if depth == 0:
            return [],-self.policy.eval(pos)
        bestbranch,minscore = [],self.upper
        for move in pos.gen_moves():
            nextpos = pos.move(move)
            branch,score = self.maximize(nextpos,depth-1)
            if score <= minscore:
                bestbranch,minscore = [move,]+branch,score
        return bestbranch,minscore

    def search(self,pos,secs=NotImplemented):
        self.nodes = 0
        timestart = time.time()
        bestbranch,bestscore = self.maximize(pos,self.maxdepth)
        timespent = time.time() - timestart
        if self.showsearch:
            print()
            print('${} {}ms {} {}: {}'.format(
                self.maxdepth,int(timespent*1000),
                self.nodes,bestscore,pos.mrender(bestbranch)))
        return pos.legal(bestbranch[0]),bestscore


class Negamax(Engine):
    """Negamax

    """

    MAXDEPTH = 4

    def __init__(
            self,
            maxdepth = MAXDEPTH,
            policy = SunfishPolicy(),
            showsearch = 1):
        super(Negamax,self).__init__()
        self.maxdepth = maxdepth
        self.policy = policy
        self.upper = self.policy.upper
        self.showsearch = showsearch

    def negamax(self,pos,depth,alpha,beta):
        self.nodes += 1
        if pos.gameover():
            return [None,],self.policy.eval(pos)
        if depth == 0:
            return [],self.policy.eval(pos)
        bestbranch,maxscore = [],-self.upper
        for move in pos.gen_moves():
            nextpos = pos.move(move)
            branch,score = self.negamax(nextpos,depth-1,-beta,-alpha)
            score = -score # nega
            if score >= maxscore:
                bestbranch,maxscore = [move,]+branch,score
                alpha = max(alpha,score)
        return bestbranch,maxscore

    def search(self,pos,secs=NotImplemented):
        self.nodes = 0
        timestart = time.time()
        bestbranch,bestscore = self.negamax(
            pos,self.maxdepth,-self.upper,self.upper)
        timespent = time.time() - timestart
        if self.showsearch:
            print()
            print('${} {}ms {} {}: {}'.format(
                self.maxdepth,int(timespent*1000),
                self.nodes,bestscore,pos.mrender(bestbranch)))
        return pos.legal(bestbranch[0]),bestscore


class Entry(namedtuple('Entry',['depth','score','branch','bound'])):
    """create :obj:`Entry` for the transposition table:

    >>> alpha_o,beta_o = -1,1
    >>> bound = lambda score:(
    ...     (score >= beta_o) # BOUND_LOWER
    ...     -(score <= alpha_o) # BOUND_UPPER
    ...     # BOUND_EXACT
    ... )
    >>> assert bound(+1) == Entry.BOUND_LOWER
    >>> assert bound(-1) == Entry.BOUND_UPPER
    >>> assert bound( 0) == Entry.BOUND_EXACT

    """
    BOUND_LOWER,BOUND_EXACT,BOUND_UPPER = 1,0,-1

    def narrowing(self,alpha,beta):
        if self.bound == self.BOUND_LOWER: # was entry.score >= beta
            alpha = max(alpha,self.score)
        elif self.bound == self.BOUND_UPPER: # was entry.score <= alpha
            beta = min(beta,self.score)
        return alpha,beta

    def isexact(self):
        return self.bound == self.BOUND_EXACT


class AlphaBeta(Engine):
    """alpha-beta is negamax with pruning and transposition table

    * has moves prioritization - :meth:`AlphaBeta.sorted_gen_moves`
    * has iterative deepening - :meth:`AlphaBeta.search`
    * has show analysis - :meth:`AlphaBeta.show`

    """

    MAXDEPTH = 15

    def __init__(
            self,
            maxdepth = MAXDEPTH,
            policy = SunfishPolicy(),
            showsearch = 1):
        super(AlphaBeta,self).__init__()
        self.maxdepth = maxdepth
        self.policy = policy
        self.upper = self.policy.upper
        self.showsearch = showsearch

    def sorted_gen_moves(self,pos):
        """return `moves` in ascending order of the next move scores

        * ascending scores for the next player is equivalent to
          descending value for moves of the current player;

        >>> FEN_MATE0 = '7k/6Q1/5K2/8/8/8/8/8 b - - 0 1'
        >>> q2 = Superposition(*tools.parseFEN(FEN_MATE0))
        >>> q2.board.find("K"), q2.board.find("q")
        (91, 82)
        >>> a = AlphaBeta()
        >>> moves = a.sorted_gen_moves(q2)

        >>> list(moves)
        [(91, 82), (91, 92), (91, 81)]
        >>> scores = list(q2.move(move).score for move in moves)
        >>> scores[0] < scores[1] < scores[2]
        True

        """
        scoremoves = sorted(
            (self.policy.eval(pos.move(move)),move)
            for move in pos.gen_moves())
        return OrderedDict(scoremoves).values()

    def negamax(self,pos,depth,alpha,beta):
        self.nodes += 1
        alpha_o = alpha
        entry = self.tt.get(pos) # default=None
        if entry is not None and entry.depth >= depth:
            self.hits += 1
            alpha,beta = entry.narrowing(alpha,beta)
            if alpha >= beta or entry.isexact():
                return entry.branch,entry.score
        if pos.gameover():
            return [None,],self.policy.eval(pos)
        if depth == 0:
            return [],self.policy.eval(pos)
        bestbranch,maxscore = [],-self.upper
        for move in self.sorted_gen_moves(pos):
            nextpos = pos.move(move)
            branch,score = self.negamax(nextpos,depth-1,-beta,-alpha)
            score = -score # nega
            if score >= maxscore:
                bestbranch,maxscore = [move,]+branch,score
                alpha = max(alpha,score)
                if alpha >= beta: break # pruning
        self.tt[pos] = Entry(
            depth,maxscore,bestbranch,
            (maxscore >= beta)-(maxscore <= alpha_o))
        return bestbranch,maxscore

    def search(self,pos,secs=60):
        timestart = time.time()
        self.nodes = 0
        for depth in range(1,self.maxdepth+1):
            self.tt,self.hits = LRUCache(TABLE_SIZE),0 # transposition table
            bestbranch,bestscore = self.negamax(
                pos,depth,-self.upper,self.upper)
            timespent = time.time() - timestart
            if self.showsearch:
                self.show(pos,timespent,depth)
            if (timespent > secs) or (bestbranch[-1]) is None:
                break
        return pos.legal(bestbranch[0]),bestscore

    def show(self,pos,timespent,depth):
        best = self.tt.get(pos)
        if best is None:
            return
        pv = ",".join(pos.mrender(best.branch))
        if depth == 1:
            print()
        print('${} {}ms {}/{}/{} {}: {}'.format(
            depth,int(timespent*1000),
            self.hits,len(self.tt.od),self.nodes,
            best.score,pv))


enginedict = dict(
    Sunfish=Sunfish,
    Fool=Fool,
    Minimax=Minimax,
    Negamax=Negamax,
    AlphaBeta=AlphaBeta,
    )


if __name__ == '__main__':

    import sys, os
    import doctest

    docscript = lambda obj=None: doctest.script_from_examples(
        __doc__ if obj is None else getattr(obj,'__doc__'))

    if sys.argv[0] == "":
        print(doctest.testmod(optionflags=doctest.REPORT_ONLY_FIRST_FAILURE))
    else:
        if len(sys.argv) == 3:
            white = enginedict[sys.argv[1]]()
            black = enginedict[sys.argv[2]]()
            p = play(white,black)
