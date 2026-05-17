"""
Download the Socha 2002 university course timetabling instances used by
the UCTP suite.

Files are written to `data/socha/` (or `$SOCHA_DIR` if set) under their
canonical names: easy01..easy05.tim, medium01..medium05.tim,
hard01..hard05.tim — the 15 instances from the Metaheuristics Network
problem set originally hosted by the IRIDIA group at ULB.

Source: https://iridia.ulb.ac.be/supp/IridiaSupp2002-001/

Re-running is safe — instances that already exist are skipped unless
`--force` is passed.
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path


PREFIXES = ("easy", "medium", "hard")
INSTANCES = [f"{p}{i:02d}" for p in PREFIXES for i in range(1, 6)]
BASE_URL = "https://iridia.ulb.ac.be/supp/IridiaSupp2002-001"


def target_dir():
    env = os.environ.get("SOCHA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "data" / "socha"


def fetch(name: str, out_dir: Path, force: bool = False) -> Path:
    out = out_dir / f"{name}.tim"
    if out.exists() and not force:
        print(f"  [skip] {out.name} already present")
        return out
    url = f"{BASE_URL}/{name}.tim"
    print(f"  [get]  {url}")
    with urllib.request.urlopen(url, timeout=20) as resp:
        text = resp.read().decode("ascii")
    out.write_text(text)
    print(f"  [ok]   {out} ({len(text)} bytes)")
    return out


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
        print(f"\n{len(failed)} instance(s) failed: {failed}", file=sys.stderr)
        sys.exit(1)
    print("\nAll instances ready.")


if __name__ == "__main__":
    main()
