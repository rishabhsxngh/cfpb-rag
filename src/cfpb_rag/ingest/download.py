"""Download the public CFPB Consumer Complaint Database CSV.

CFPB's own file server refuses scripted downloads (HTTP 403); `download_hf` is the default route. To use the CFPB file instead, download the
"All complaint data" zip in a browser and save it as data/raw/complaints.csv.zip. The ingest
step reads the zip directly, so it does not need to be unzipped.
"""
from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

from ..config import CFPB_URL, HF_REPO, RAW_DIR

MANUAL_STEPS = """
The CFPB server refused the scripted download. Download it in your browser instead:
  1. Open https://www.consumerfinance.gov/data-research/consumer-complaints/ and go to the search tool.
  2. Click "Export data", choose "All complaint data" (CSV ZIP), and download.
  3. Move the file to: {dest}
     (keep it zipped; the ingest step reads the zip directly)
Then run: cfpb-rag ingest --max-rows 60000
"""


def download_hf(dest_dir: Path = RAW_DIR, repo: str = HF_REPO) -> Path:
    """Fetch the Parquet shards of the Hugging Face mirror (about 1 GB). Resumes if interrupted."""
    from huggingface_hub import snapshot_download

    dest = dest_dir / "hf"
    print(f"Downloading Parquet shards from Hugging Face dataset {repo} to {dest} ...")
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(dest),
                      allow_patterns=["data/*.parquet", "metadata/*", "README.md"])
    n = len(list(dest.rglob("*.parquet")))
    if n == 0:
        raise SystemExit(f"No Parquet files were downloaded from {repo}.")
    print(f"Done: {n} files in {dest}")
    return dest


def download(dest_dir: Path = RAW_DIR, url: str = CFPB_URL, force: bool = False) -> Path:
    """Download the complaint zip. Returns the path to the saved file."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "complaints.csv.zip"
    csv_path = dest_dir / "complaints.csv"
    for existing in (zip_path, csv_path):
        if existing.exists() and not force:
            print(f"Using existing {existing}")
            return existing

    print(f"Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "cfpb-rag/0.1"})
    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} from the CFPB server." + MANUAL_STEPS.format(dest=zip_path))
    with resp, open(zip_path, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
    print(f"\nWrote {zip_path}")
    return zip_path
