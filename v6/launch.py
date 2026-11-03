"""Explicit selection; neural dispatches to the unchanged v5 application."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def command(controller='neural', seconds=30, hand='right'):
    if controller == 'neural':
        return [sys.executable, str(ROOT / 'neural_live.py'), '--seconds', str(seconds), '--hand', hand]
    if controller not in ('optimizer', 'hybrid'):
        raise ValueError('Unknown controller')
    return [sys.executable, '-m', 'v6.live', '--controller', controller,
            '--seconds', str(seconds), '--label', 'selected_' + controller]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller', choices=['neural', 'optimizer', 'hybrid'], default='neural')
    p.add_argument('--seconds', type=float, default=30)
    p.add_argument('--hand', choices=['right', 'left', 'any'], default='right', help='Existing v5 selector; ignored by experimental modes')
    a = p.parse_args()
    raise SystemExit(subprocess.call(command(a.controller, a.seconds, a.hand), cwd=ROOT))

if __name__ == '__main__':
    main()
