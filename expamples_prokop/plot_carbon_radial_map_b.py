#!/usr/bin/env python3
'''Thin wrapper → expamples_prokop/hermite_radial_study.py carbon'''
import sys
from hermite_radial_study import main

if __name__ == '__main__':
    sys.argv = [sys.argv[0], 'carbon'] + sys.argv[1:]
    main()
