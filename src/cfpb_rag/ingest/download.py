"""Download the public CFPB Consumer Complaint Database CSV."""
from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

from ..config import CFPB_URL, RAW_DIR


def download(dest_dir: Path = RAW_DIR, url: str = CFPB_URL, force: bool = False) -> Path:
    """Download and unzip the complaint CSV. Returns the path to the CSV."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "complaints.csv.zip"
    csv_path = dest_dir / "complaints.csv"
    if csv_path.exists() and not force:
        print(f"Using existing {csv_path}")
        return csv_path

    print(f"Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "cfpb-rag/0.1"})
    with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
    print()

    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(name) as src, open(csv_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    zip_path.unlink()
    print(f"Wrote {csv_path}")
    return csv_path
