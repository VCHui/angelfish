# angelfish
Some exercise implementations of chess engines using the very nice [sunfish](https://github.com/thomasahle/sunfish) framework and tools

## Engines implemented
In order of increasing performance:
* Minimax
* Negamax
* AlphaBeta - with pruning, transposition table and iterative deepening
* Sunfish - refitting as a mixin

All engines have a method `search(pos,secs)` as in class `sunfish.Search`

## Framework
* Decouple `sunfish.Position` into move handlings and score evaluations separately
  - Subclass `sunfish.Position` as `Superposition` to handle moves,
    * moves generation
    * game over test (test `"k" in board` instead of `score > sunfish.MATE_LOWER`)
    * legal test
    * retain `sunfish` convention of `black` player moves from a rotated position
  - A new `SunfishPolicy` class to handle heuristic evaluation of `position` in engines
    * abstraction of `sunfish.Position.score` and `sunfish.Position.value`
    * the base class adopted `sunfish.pst` scores

## A screenshot
![screenshot](https://github.com/VC-H/angelfish/blob/master/screenshot.png?raw=true)

## `python3` only, or `pypy3`

## `git` cloning with a submodule
* `sunfish` is imported into `angelfish.engines` as a `git` submodule
* My forked [`sunfish`](https://github.com/VC-H/sunfish/tree/submodule) in a separate branch has added `__init__.py` to make `sunfish` importable 
```shell
git clone --recursive https://github.com/VC-H/angelfish.git
```

## angelfish
* Distant relatives of sunfish
* Refer to a class of fresh water aquarium fish for appreciation
