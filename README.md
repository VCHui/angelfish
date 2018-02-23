# angelfish
Some exercise implementations of chess engines using the very neat [sunfish](https://github.com/thomasahle/sunfish) as a framework

## Engines implemented
* Minimax
* Negamax
* AlphaBeta - with pruning and transposition table
* Sunfish - refitting as a mixin

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

## `python3` only, or `pypy3`

## `git` cloning with a submodule
* `sunfish` is imported into `angelfish.engines` as a `git` submodule
* My forked [`sunfish`](https://github.com/VC-H/sunfish/tree/submodule) in a separate branch has added `__init__.py` to make `sunfish` importable 
```shell
git clone --recursive https://github.com/VC-H/angelfish.git
```

## anglefish
* Distant relatives of sunfish
* Refer to a class of fresh water aquarium fish for appreciation
