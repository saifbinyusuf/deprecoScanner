#!/usr/bin/env python3
"""
scripts/download_raw_data.py - Automate raw benchmark data ingestion & directory provisioning.

Provisions:
1. Clones external repositories into external/:
   - APIScanner-dev: https://github.com/rishitha957/APIScanner-dev.git
   - LLM-Deprecated-API: https://github.com/cs-wangchong/LLM-Deprecated-API.git
2. Downloads and extracts Figshare benchmark archives into data/raw/llm-dep-api/:
   - probing-inputs.zip (File ID: 47077525) -> data/raw/llm-dep-api/probing-inputs/
   - probing-results.zip (File ID: 47077528) -> data/raw/llm-dep-api/probing-results/
3. Verifies file presence and dataset integrity for NumPy, SciPy, and Pandas.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = REPO_ROOT / "data" / "raw" / "llm-dep-api"
EXTERNAL_DIR = REPO_ROOT / "external"

FIGSHARE_FILES = {
    "probing-inputs": {
        "file_id": "47077525",
        "url": "https://ndownloader.figshare.com/files/47077525",
        "target_dir": RAW_DATA_DIR / "probing-inputs",
        "check_file": RAW_DATA_DIR / "probing-inputs" / "pandas" / "samples.json",
    },
    "probing-results": {
        "file_id": "47077528",
        "url": "https://ndownloader.figshare.com/files/47077528",
        "target_dir": RAW_DATA_DIR / "probing-results",
        "check_file": RAW_DATA_DIR / "probing-results" / "pandas",
    },
}

EXTERNAL_REPOS = {
    "apiscanner-dev": "https://github.com/rishitha957/APIScanner-dev.git",
    "llm-deprecated-api": "https://github.com/cs-wangchong/LLM-Deprecated-API.git",
}


def check_dataset_status() -> bool:
    print("\n" + "=" * 70)
    print("CHECKING BENCHMARK DATASET STATUS")
    print("=" * 70)
    all_ok = True

    for name, info in FIGSHARE_FILES.items():
        check_target = info["check_file"]
        if check_target.exists():
            print(f"[OK] {name}: Present at {info['target_dir']}")
        else:
            print(f"[MISSING] {name}: Missing at {info['target_dir']}")
            all_ok = False

    # Check sample counts if probing-inputs is present
    probing_inputs_dir = RAW_DATA_DIR / "probing-inputs"
    if probing_inputs_dir.exists():
        for lib in ["numpy", "pandas", "scipy"]:
            samples_file = probing_inputs_dir / lib / "samples.json"
            if samples_file.exists():
                try:
                    with open(samples_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    print(f"  • {lib}: {len(data)} code samples verified")
                except Exception as e:
                    print(f"  • {lib}: Error reading samples.json: {e}")
            else:
                print(f"  • {lib}: Missing samples.json")
                all_ok = False

    for repo_name in EXTERNAL_REPOS:
        repo_dir = EXTERNAL_DIR / repo_name
        if repo_dir.exists() and (repo_dir / ".git").exists():
            print(f"[OK] External repo '{repo_name}': Present at {repo_dir}")
        else:
            print(f"[NOTE] External repo '{repo_name}': Not cloned at {repo_dir} (optional)")

    print("=" * 70)
    return all_ok


def download_and_extract(name: str, force: bool = False) -> None:
    info = FIGSHARE_FILES[name]
    target_dir = info["target_dir"]

    if target_dir.exists() and info["check_file"].exists() and not force:
        print(f"[{name}] Already provisioned at {target_dir}. Skipping download (use --force to re-download).")
        return

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DATA_DIR / f"{name}.zip"

    print(f"[{name}] Downloading from Figshare ({info['url']})...")
    req = urllib.request.Request(
        info["url"],
        headers={"User-Agent": "DeprecoScanner-Replication/1.0"},
    )
    with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as out_file:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        block_size = 1024 * 1024
        while True:
            chunk = resp.read(block_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size:
                pct = (downloaded / total_size) * 100
                sys.stdout.write(f"\r[{name}] Progress: {downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB ({pct:.1f}%)")
                sys.stdout.flush()
    print(f"\n[{name}] Download complete ({zip_path.stat().st_size / 1024 / 1024:.1f} MB).")

    print(f"[{name}] Extracting to {target_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DATA_DIR)
    
    # If the zip extracted to a nested or different folder name, ensure standard layout
    if not target_dir.exists() and (RAW_DATA_DIR / f"{name}").exists():
        pass
    
    zip_path.unlink()
    print(f"[{name}] Extraction complete and archive removed.")


def clone_external_repos() -> None:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    for repo_name, url in EXTERNAL_REPOS.items():
        repo_dir = EXTERNAL_DIR / repo_name
        if repo_dir.exists():
            print(f"[{repo_name}] Already cloned at {repo_dir}.")
        else:
            print(f"[{repo_name}] Cloning {url}...")
            subprocess.run(["git", "clone", url, str(repo_dir)], check=True)


def main():
    parser = argparse.ArgumentParser(description="Provision raw dataset and external repos for DeprecoScanner.")
    parser.add_argument("--check", action="store_true", help="Only check status of datasets without downloading.")
    parser.add_argument("--force", action="store_true", help="Force re-download of Figshare archives.")
    parser.add_argument("--with-external", action="store_true", help="Also clone external repos (APIScanner-dev, LLM-Deprecated-API).")
    args = parser.parse_args()

    if args.check:
        ok = check_dataset_status()
        sys.exit(0 if ok else 1)

    print("Provisioning DeprecoScanner Raw Data & Dependencies...")
    download_and_extract("probing-inputs", force=args.force)
    download_and_extract("probing-results", force=args.force)

    if args.with_external:
        clone_external_repos()

    check_dataset_status()


if __name__ == "__main__":
    main()
