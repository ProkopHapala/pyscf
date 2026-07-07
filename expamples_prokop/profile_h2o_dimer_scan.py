#!/usr/bin/env python3
'''H2O dimer scan — thin wrapper around profile_dimer_scan.py (n0=3).'''
import os
import subprocess
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_SCRIPT = os.path.join(os.path.dirname(__file__), 'profile_dimer_scan.py')
_SCAN = '/home/prokophapala/git/CompChemUtils/tmp/H2O_dimer_scan_dftb/scan_out.xyz'
_DIST = '/home/prokophapala/git/CompChemUtils/tmp/H2O_dimer_scan_dftb/distances.dat'
_REF = '/home/prokophapala/git/CompChemUtils/tmp/H2O_dimer_scan_dftb/scan.dat'
_OUT = os.path.join(_REPO, 'debug', 'profile_h2o_dimer_scan')

if __name__ == '__main__':
    cmd = [sys.executable, '-u', _SCRIPT, '--scan-xyz', _SCAN, '--distances-file', _DIST, '--ref-scan', _REF, '--n0', '3', '--title', 'H₂O dimer', '--z-label', 'O···O distance (Å)', '--outdir', _OUT, *sys.argv[1:]]
    env = os.environ.copy()
    env.setdefault('PYTHONPATH', _REPO)
    raise SystemExit(subprocess.call(cmd, env=env))
