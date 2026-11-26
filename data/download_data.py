import csv
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"


def download_all():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "sources.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        path = RAW_DIR / row["file"]
        if path.exists():
            print("exists:", row["file"])
            continue
        response = requests.get(row["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
        print("downloaded:", row["file"], len(response.content) // 1024, "KB")


if __name__ == "__main__":
    download_all()
