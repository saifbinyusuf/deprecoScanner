#!/usr/bin/env python3
"""
scripts/setup_benchmark_libs.py - Download and extract minimal covering historical source trees.

Fetches official distribution packages (wheels or sdists) from PyPI for:
- numpy: 1.26.4
- pandas: 0.22.0, 1.5.3, 2.2.3
- scipy: 0.19.1, 1.2.3, 1.7.3, 1.12.0

Extracts only the Python package directory into data/benchmark_libs/<lib>-<version>/<lib>/.
Also establishes the lightweight stubs in data/benchmark_libs/stubs/ for C-extension re-exports.
"""

from __future__ import annotations

import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple


TARGET_SNAPSHOTS: List[Tuple[str, str]] = [
    ("numpy", "1.26.4"),
    ("pandas", "0.22.0"),
    ("pandas", "1.5.3"),
    ("pandas", "2.2.3"),
    ("scipy", "0.19.1"),
    ("scipy", "1.2.3"),
    ("scipy", "1.7.3"),
    ("scipy", "1.12.0"),
]


def fetch_pypi_urls(pkg: str, version: str) -> List[Dict]:
    url = f"https://pypi.org/pypi/{pkg}/{version}/json"
    req = urllib.request.Request(url, headers={"User-Agent": "deprecoScanner-pilot"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    return data.get("urls", [])


def extract_wheel(whl_path: Path, out_dir: Path, pkg: str) -> None:
    with zipfile.ZipFile(whl_path, "r") as zf:
        members = [m for m in zf.namelist() if m.startswith(f"{pkg}/") and not m.endswith(".so")]
        for m in members:
            zf.extract(m, out_dir)


def extract_sdist(tar_path: Path, out_dir: Path, pkg: str, version: str) -> None:
    prefix = f"{pkg}-{version}/{pkg}/"
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf.getmembers():
            if m.name.startswith(prefix):
                rel_path = m.name[len(f"{pkg}-{version}/"):]
                m.name = rel_path
                tf.extract(m, out_dir)


def setup_snapshot(pkg: str, version: str, base_dir: Path) -> Path:
    target_dir = base_dir / f"{pkg}-{version}"
    pkg_sub = target_dir / pkg
    if pkg_sub.exists() and any(pkg_sub.glob("*.py")):
        print(f"[{pkg} {version}] Already present at {target_dir}")
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    urls = fetch_pypi_urls(pkg, version)
    if not urls:
        raise RuntimeError(f"No PyPI URLs found for {pkg}=={version}")

    wheels = [u for u in urls if u["packagetype"] == "bdist_wheel"]
    best_wheel = None
    if wheels:
        best_wheel = wheels[0]
        for w in wheels:
            fn = w["filename"]
            if "py3" in fn or "none-any" in fn or "cp310" in fn or "cp39" in fn:
                best_wheel = w
                break

    if best_wheel:
        fn = best_wheel["filename"]
        dl_path = target_dir / fn
        print(f"[{pkg} {version}] Downloading wheel {fn} ({best_wheel['size'] / 1024 / 1024:.1f} MB)...")
        urllib.request.urlretrieve(best_wheel["url"], dl_path)
        print(f"[{pkg} {version}] Extracting {pkg}/ from wheel...")
        extract_wheel(dl_path, target_dir, pkg)
        dl_path.unlink()
    else:
        sdists = [u for u in urls if u["packagetype"] == "sdist"]
        if not sdists:
            raise RuntimeError(f"No sdist or wheel found for {pkg}=={version}")
        sdist = sdists[0]
        fn = sdist["filename"]
        dl_path = target_dir / fn
        print(f"[{pkg} {version}] Downloading sdist {fn} ({sdist['size'] / 1024 / 1024:.1f} MB)...")
        urllib.request.urlretrieve(sdist["url"], dl_path)
        print(f"[{pkg} {version}] Extracting {pkg}/ from sdist...")
        extract_sdist(dl_path, target_dir, pkg, version)
        dl_path.unlink()

    print(f"[{pkg} {version}] Extracted successfully.")
    return target_dir


def setup_stubs(base_dir: Path) -> None:
    stubs_dir = base_dir / "stubs"
    numpy_stubs = stubs_dir / "numpy"
    numpy_stubs.mkdir(parents=True, exist_ok=True)

    stub_content = '''"""
Type stub file for numpy C-extension and dynamic top-level re-exports.
Allows Jedi to resolve top-level module symbols without compiling native extensions.
"""
from numpy.core.fromnumeric import (
    alltrue as alltrue,
    product as product,
    cumproduct as cumproduct,
)
'''
    (numpy_stubs / "__init__.pyi").write_text(stub_content, encoding="utf-8")

    scipy_misc_stubs = stubs_dir / "scipy" / "misc"
    scipy_misc_stubs.mkdir(parents=True, exist_ok=True)
    misc_stub = '''"""
Type stub file for scipy.misc re-exports.
"""
from scipy.special import (
    comb as comb,
    factorial as factorial,
    factorial2 as factorial2,
    logsumexp as logsumexp,
)
'''
    # Also inject directly into numpy-1.26.4 __init__.pyi so it resolves natively
    numpy_tree_init_pyi = base_dir / "numpy-1.26.4" / "numpy" / "__init__.pyi"
    if numpy_tree_init_pyi.exists():
        pyi_text = numpy_tree_init_pyi.read_text(encoding="utf-8")
        if "from numpy.core.fromnumeric import alltrue" not in pyi_text:
            with open(numpy_tree_init_pyi, "a", encoding="utf-8") as f:
                f.write("\nfrom numpy.core.fromnumeric import alltrue as alltrue, product as product, cumproduct as cumproduct\n")

    print(f"Established Jedi stubs at {stubs_dir}")


def main() -> None:
    base_dir = Path("data/benchmark_libs").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    print("======================================================================")
    print("SETTING UP HISTORICAL BENCHMARK SOURCE TREES")
    print("======================================================================")

    for pkg, ver in TARGET_SNAPSHOTS:
        try:
            setup_snapshot(pkg, ver, base_dir)
        except Exception as e:
            print(f"Error setting up {pkg}=={ver}: {e}")

    setup_stubs(base_dir)
    print("\nAll historical source trees and stubs are ready in data/benchmark_libs/.")


if __name__ == "__main__":
    main()
