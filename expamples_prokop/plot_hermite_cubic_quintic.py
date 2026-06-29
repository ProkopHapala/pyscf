#!/usr/bin/env python3
'''Thin wrapper → expamples_prokop/hermite_radial_study.py'''
import sys
from hermite_radial_study import main

if __name__ == '__main__':
    argv = sys.argv[1:]
    if not argv or argv[0] not in ('carbon', 'grid', 'compare', 'report', 'matrix', 'f32'):
        sys.argv = [sys.argv[0], 'carbon'] + argv
    else:
        sys.argv = [sys.argv[0]] + argv
    main()
