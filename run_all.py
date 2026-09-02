#!/usr/bin/env python3
"""
Runs the full reproduction pipeline end to end: every numbered script under
scripts/, in order, each as its own subprocess (matching how you'd run them
by hand). Every script checkpoints its outputs under checkpoints/, so a
second run of this after the first completes is fast -- every heavy step
just hits its cache.

Usage:
    python run_all.py                  # run every script, 02 through 27
    python run_all.py --from 15        # resume starting at script 15
    python run_all.py --only 21 24 25  # run just these sections
    python run_all.py --list           # print the section order and exit

See the top-level README for the data you need to supply under data/
before running this, and for what each script depends on / produces.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"


def discover_scripts():
    return sorted(SCRIPTS_DIR.glob("[0-9][0-9]_*.py"), key=lambda p: p.name)


def section_number(path):
    return path.name.split("_", 1)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="from_section", metavar="NN", help="start at this section number (e.g. 15)")
    parser.add_argument("--to", dest="to_section", metavar="NN", help="stop after this section number (e.g. 20)")
    parser.add_argument("--only", nargs="+", metavar="NN", help="run only these section numbers")
    parser.add_argument("--list", action="store_true", help="print the section order and exit")
    args = parser.parse_args()

    scripts = discover_scripts()
    if not scripts:
        print(f"No scripts found under {SCRIPTS_DIR}", file=sys.stderr)
        sys.exit(1)

    if args.list:
        for p in scripts:
            print(p.name)
        return

    if args.only:
        wanted = set(args.only)
        scripts = [p for p in scripts if section_number(p) in wanted]
    else:
        if args.from_section:
            scripts = [p for p in scripts if section_number(p) >= args.from_section]
        if args.to_section:
            scripts = [p for p in scripts if section_number(p) <= args.to_section]

    print(f"Running {len(scripts)} script(s):")
    for p in scripts:
        print(f"  {p.name}")
    print()

    for p in scripts:
        print(f"{'='*70}\n=== {p.name} ===\n{'='*70}", flush=True)
        t0 = time.time()
        result = subprocess.run([sys.executable, str(p)])
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"\n!!! {p.name} failed (exit code {result.returncode}) after {elapsed:.1f}s. Stopping.",
                  file=sys.stderr)
            sys.exit(result.returncode)
        print(f"--- {p.name} done in {elapsed:.1f}s ---\n", flush=True)

    print("All scripts completed successfully.")


if __name__ == "__main__":
    main()
