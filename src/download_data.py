"""Download the WESAD HRV feature CSV from Hugging Face into data/raw/."""

from __future__ import annotations

import urllib.request

from src.preprocessing import DATASET_URL, RAW_CSV


def main() -> None:
    if RAW_CSV.exists():
        print(f"Already present: {RAW_CSV}")
        return
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    tmp = RAW_CSV.with_suffix(".part")
    print(f"Downloading {DATASET_URL} (~225 MB) ...")
    urllib.request.urlretrieve(DATASET_URL, tmp)
    tmp.rename(RAW_CSV)
    print(f"Saved to {RAW_CSV}")


if __name__ == "__main__":
    main()
