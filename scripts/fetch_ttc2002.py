"""
Download the TTC 2002 (International Timetabling Competition) instances
used by the UCTP suite. 20 instances named comp01..comp20 are written
to `data/ttc2002/` (or `$TTC2002_DIR` if set), in the same .tim file
format as the Socha set.

The TTC 2002 instances are referenced in Yang & Jat 2011 (Table VII)
and have been mirrored at the IDSIA website. We try a small list of
known mirrors in turn; if all fail, the script prints the URLs so the
user can fetch them manually.

Re-running is safe — instances that already exist are skipped unless
`--force` is passed.
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path


INSTANCES = [f"comp{i:02d}" for i in range(1, 21)]

# Known mirror locations for the TTC 2002 instance set. The script
# tries each base URL in turn; the first one that returns a valid file
# wins. Add additional mirrors here as they appear.
BASE_URLS = (
    "http://www.idsia.ch/Files/ttcomp2002",
    "https://www.idsia.ch/Files/ttcomp2002",
)


def target_dir():
    env = os.environ.get("TTC2002_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "data" / "ttc2002"


def fetch(name: str, out_dir: Path, force: bool = False) -> Path:
    out = out_dir / f"{name}.tim"
    if out.exists() and not force:
        print(f"  [skip] {out.name} already present")
        return out
    last_err = None
    for base in BASE_URLS:
        url = f"{base}/{name}.tim"
        try:
            print(f"  [get]  {url}")
            with urllib.request.urlopen(url, timeout=20) as resp:
                text = resp.read().decode("ascii")
            out.write_text(text)
            print(f"  [ok]   {out} ({len(text)} bytes)")
            return out
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  [warn] mirror failed ({e}); trying next...")
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
        except Exception as e:  # noqa: BLE001
            print(f"  [fail] {name}: {e}")
            failed.append(name)
    if failed:
        print(
            f"\n{len(failed)} instance(s) failed: {failed}\n"
            f"Manual fallback: download each `<name>.tim` file from any "
            f"TTC 2002 mirror (e.g. http://www.idsia.ch/Files/ttcomp2002/) "
            f"into {out_dir}.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("\nAll TTC 2002 instances ready.")


if __name__ == "__main__":
    main()
