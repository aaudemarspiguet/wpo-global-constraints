"""Load ContentWise Impressions, check the schema, and print the category mix by row position.

    python scripts/contentwise_calibrate.py --data-dir data/contentwise --sample 2000000

Get the data link from github.com/ContentWise/contentwise-impressions (the
README links a short, optional form). License: CC BY-NC-SA 4.0 for the data.
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wpo.data.contentwise import load_tables, check_schema, calibration_report  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=str(ROOT / "data" / "contentwise"))
ap.add_argument("--sample", type=int, default=2_000_000, help="subsample rows per table")
a = ap.parse_args()

tables = load_tables(a.data_dir, nrows_sample=a.sample)
if not check_schema(tables):
    sys.exit("Fix the column constants in wpo/data/contentwise.py, then re-run.")
mix = calibration_report(tables)
out = ROOT / "results" / "contentwise_row_mix.csv"
out.parent.mkdir(exist_ok=True)
mix.to_csv(out)
print(mix.head(10).round(3))
print("saved", out)
