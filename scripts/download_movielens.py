"""Download and unzip a MovieLens dataset from GroupLens.

    python scripts/download_movielens.py                 # ml-latest-small (~1 MB)
    python scripts/download_movielens.py --name ml-32m   # bigger; check grouplens.org for names

Data: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
History and Context. ACM TiiS 5(4). Read the README in the zip for license
terms (non-commercial; cite the paper).
"""
import argparse
import io
import pathlib
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="ml-latest-small")
ap.add_argument("--dest", default=str(ROOT / "data"))
a = ap.parse_args()
url = f"https://files.grouplens.org/datasets/movielens/{a.name}.zip"
print("downloading", url)
with urllib.request.urlopen(url, timeout=120) as r:
    zipfile.ZipFile(io.BytesIO(r.read())).extractall(a.dest)
print("extracted to", pathlib.Path(a.dest) / a.name)
