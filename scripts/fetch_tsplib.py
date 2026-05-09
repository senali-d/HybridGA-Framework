"""
Download the five baseline TSPLIB instances used by the TSP suite.

Files are written to `data/tsplib/` (or `$TSPLIB_DIR` if set) under their
canonical TSPLIB names: eil51.tsp, berlin52.tsp, kroA100.tsp, ch150.tsp,
kroA200.tsp.

Source: the official TSPLIB-95 mirror at the University of Heidelberg
(http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/). Files are
public-domain academic test data.

Re-running is safe — instances that already exist are skipped unless
`--force` is passed.
"""

import argparse
import gzip
import os
import sys
import urllib.request
from pathlib import Path

INSTANCES = ["eil51", "berlin52", "kroA100", "ch150", "kroA200"]
# Stable GitHub mirror of the TSPLIB-95 TSP archive (uncompressed .tsp files).
# Falls back to the Heidelberg origin if the mirror is unavailable.
MIRRORS = [
    ("https://raw.githubusercontent.com/mastqe/tsplib/master", ".tsp", False),
    ("http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp", ".tsp.gz", True),
]


def target_dir():
    env = os.environ.get("TSPLIB_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "data" / "tsplib"


def fetch(name: str, out_dir: Path, force: bool = False) -> Path:
    out = out_dir / f"{name}.tsp"
    if out.exists() and not force:
        print(f"  [skip] {out.name} already present")
        return out
    last_err = None
    for base, suffix, gzipped in MIRRORS:
        url = f"{base}/{name}{suffix}"
        try:
            print(f"  [get]  {url}")
            with urllib.request.urlopen(url, timeout=20) as resp:
                payload = resp.read()
            text = (gzip.decompress(payload).decode("ascii") if gzipped
                    else payload.decode("ascii"))
            out.write_text(text)
            print(f"  [ok]   {out} ({len(text)} bytes)")
            return out
        except Exception as e:  # noqa: BLE001 — try next mirror
            print(f"  [warn] mirror failed: {e}")
            last_err = e
    raise RuntimeError(f"all mirrors failed for {name}: {last_err}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if the file already exists.")
    args = ap.parse_args()

    out_dir = target_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {out_dir}")
    failed = []
    for name in INSTANCES:
        try:
            fetch(name, out_dir, force=args.force)
        except Exception as e:  # noqa: BLE001 — we want to keep going
            print(f"  [fail] {name}: {e}")
            failed.append(name)
    if failed:
        print(f"\n{len(failed)} instance(s) failed: {failed}", file=sys.stderr)
        sys.exit(1)
    print("\nAll instances ready.")


if __name__ == "__main__":
    main()
