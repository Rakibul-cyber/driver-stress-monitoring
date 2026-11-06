"""Download the WESAD HRV feature CSV from Hugging Face into data/raw/.

Uses `requests` (bundled CA certificates via certifi) rather than urllib,
which fails SSL verification on python.org macOS builds without extra setup.
"""

from __future__ import annotations

import requests

from src.preprocessing import DATASET_URL, RAW_CSV


def main() -> None:
    if RAW_CSV.exists():
        print(f"Already present: {RAW_CSV}")
        return
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    tmp = RAW_CSV.with_suffix(".part")
    print(f"Downloading {DATASET_URL} (~225 MB) ...")
    with requests.get(DATASET_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.rename(RAW_CSV)
    print(f"Saved to {RAW_CSV} ({RAW_CSV.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
