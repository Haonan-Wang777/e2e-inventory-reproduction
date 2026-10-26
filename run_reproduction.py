"""One entry point for the released-data reproduction and transparent demo."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-demo', action='store_true', help='Skip the illustrative neural-network training')
    parser.add_argument('--assemble-only', action='store_true', help='Assemble report from previously verified outputs')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    os.chdir(ROOT)
    (ROOT / 'tmp').mkdir(exist_ok=True)
    (ROOT / 'results' / 'logs').mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING='utf-8', TEMP=str(ROOT / 'tmp'), TMP=str(ROOT / 'tmp'),
               MPLCONFIGDIR=str(ROOT / 'tmp' / 'matplotlib'))
    tasks = [] if args.assemble_only else [
        ('offline', ['scripts/reproduce_offline.py']),
        ('field', ['scripts/reproduce_field.py']),
    ]
    if not args.assemble_only and not args.skip_demo:
        tasks.append(('model_demo', ['scripts/train_demo.py', '--epochs', str(args.epochs)]))
    tasks.extend([
        ('source_equivalence', ['scripts/verify_source_equivalence.py']),
        ('provenance', ['scripts/provenance.py']),
        ('report', ['scripts/build_report.py']),
    ])
    records = []
    for name, command in tasks:
        started = time.perf_counter()
        print(f'Running {name} ...', flush=True)
        with (ROOT / 'results' / 'logs' / f'{name}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, *command], cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        records.append(dict(stage=name, exit_code=result.returncode, seconds=time.perf_counter()-started))
        (ROOT / 'results' / 'run_summary.json').write_text(json.dumps(dict(
            mode='assemble_existing_outputs' if args.assemble_only else 'recompute',
            demo_requested=not args.skip_demo, stages=records), indent=2), encoding='utf-8')
        if result.returncode:
            print((ROOT / 'results' / 'logs' / f'{name}.log').read_text(encoding='utf-8'))
            raise SystemExit(result.returncode)
        print(f'Completed {name} ({records[-1]["seconds"]:.1f}s)', flush=True)
    print('Finished. Read results/reproduction_report.md', flush=True)


if __name__ == '__main__':
    main()
