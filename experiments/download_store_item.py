import argparse
import os
import shutil
import subprocess
import sys
import zipfile


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COMPETITION = "demand-forecasting-kernels-only"
EXPECTED_FILES = ["train.csv", "test.csv", "sample_submission.csv"]


def has_expected_files(raw_dir):
    return all(os.path.exists(os.path.join(raw_dir, name)) for name in EXPECTED_FILES)


def unzip_archives(raw_dir):
    for name in os.listdir(raw_dir):
        if not name.endswith(".zip"):
            continue
        path = os.path.join(raw_dir, name)
        with zipfile.ZipFile(path) as zf:
            zf.extractall(raw_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=os.path.join(ROOT, "data", "raw", "store_item"))
    parser.add_argument("--force", action="store_true", help="download even when the CSV files already exist")
    args = parser.parse_args()

    raw_dir = os.path.abspath(args.raw_dir)
    os.makedirs(raw_dir, exist_ok=True)

    if has_expected_files(raw_dir) and not args.force:
        print("Store Item raw files already exist in %s" % raw_dir)
        return 0

    if shutil.which("kaggle") is None:
        print("Kaggle CLI is not installed.")
        print("Install it with: pip install -r requirements-data.txt")
        print("Then configure Kaggle credentials: https://github.com/Kaggle/kaggle-api#api-credentials")
        return 1

    cmd = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        COMPETITION,
        "-p",
        raw_dir,
        "--force",
    ]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print("\nKaggle download failed.")
        print("Make sure the Kaggle account has accepted the competition rules:")
        print("https://www.kaggle.com/competitions/%s/rules" % COMPETITION)
        return proc.returncode

    unzip_archives(raw_dir)
    if not has_expected_files(raw_dir):
        print("Downloaded files, but expected CSV files were not found in %s" % raw_dir)
        print("Expected: %s" % ", ".join(EXPECTED_FILES))
        return 1

    print("Store Item raw files are ready in %s" % raw_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
