#!/usr/bin/env python
# -*- coding: utf-8 -*-

from engines import Superposition,enginedict

def matetests(engineclass,fensfilepath,secs):
    """a re-implementation of :func:`sunfish.test.allmate`"""
    with open(fensfilepath,"r") as f:
        for line in f:
            line = line.strip()
            pos = Superposition.init(line)
            engine = engineclass(showsearch=1)
            move,score = engine.search(pos,secs)
            print("!",line,"=",pos.mrender(move))

if __name__ == '__main__':

    import sys, os

    if len(sys.argv) < 3:
        this = os.path.basename(sys.argv[0])
        print()
        print('usage: {} [engine] [fensfilepath] [secs]'.format(this))
        print('  engines available:',", ".join(enginedict.keys()))
        print('  secs: 2 (default)')
        print()
        sys.exit(1)
    engineclass = enginedict[sys.argv[1]]
    fensfilepath = sys.argv[2]
    secs = 2
    if len(sys.argv) > 3:
        secs = int(sys.argv[3])
    matetests(engineclass,fensfilepath,secs)
