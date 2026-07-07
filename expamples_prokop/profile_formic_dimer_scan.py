#!/usr/bin/env python3
'''Formic dimer scan — thin wrapper around profile_dimer_scan.py (n0=5).'''
import os
import subprocess
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_SCRIPT = os.path.join(os.path.dirname(__file__), 'profile_dimer_scan.py')
_GEOM = os.path.join(_REPO, 'data', 'xyz', 'formic_dimer.xyz')
_DIST = '/home/prokophapala/git/CompChemUtils/tmp/H2O_dimer_scan_dftb/distances.dat'
_REF = '/home/prokophapala/git/CompChemUtils/tmp/H2O_dimer_scan_dftb/scan.dat'
_OUT = os.path.join(_REPO, 'debug', 'profile_formic_dimer_scan')

if __name__ == '__main__':
    cmd = [sys.executable, '-u', _SCRIPT, '--geom', _GEOM, '--n0', '5', '--distances-file', _DIST, '--ref-scan', _REF, '--title', 'formic dimer', '--z-label', 'O···O distance (Å)', '--outdir', _OUT, *sys.argv[1:]]
    env = os.environ.copy()
    env.setdefault('PYTHONPATH', _REPO)
    raise SystemExit(subprocess.call(cmd, env=env))
