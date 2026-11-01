"""Separate fixed-GL24 cost namespace; frozen old sources are read-only.

The old cost functions already obtain the integration order from their bound
independent numerical report. Reuse exactly those loops and synthetic inputs;
add only this version's source/protocol provenance and enforce order24.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

from experiments import round15_quartic_preflight as P

ROOT=P.ROOT
EXTRA_SOURCES=('ROUND15_NUMERICAL24_PROTOCOL.md',
               'experiments/round15_quartic_preflight24.py',
               'tests/test_round15_quartic_preflight24.py')


def require_order24(value,key):
    if value[key]!=24:
        raise ValueError('this separate numerical-repair namespace requires fixed GL24')


def freeze(out,report):
    require_order24(P.read(report/'verification.json'),'quadrature_order')
    # Scoped provenance override only. P.freeze's complete checks and exact
    # source/input snapshot mechanism are unchanged; no on-disk old file edits.
    with patch.object(P,'SOURCES',P.SOURCES+EXTRA_SOURCES):
        P.freeze(out,report)


def run(out,device):
    lock=P.verify(out)
    require_order24(lock,'order')
    if not set(EXTRA_SOURCES).issubset(lock['source_hashes']):
        raise ValueError('missing numerical-repair protocol/source provenance')
    P.run(out,device)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'))
    parser.add_argument('--out',type=Path,default=ROOT/'results/round15_quartic_preflight24')
    parser.add_argument('--report',type=Path);parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    if args.stage=='freeze':
        if args.report is None:parser.error('freeze requires passed GL24 --report')
        freeze(args.out,args.report)
    else:run(args.out,args.device)


if __name__=='__main__':main()
