import os
import subprocess


M5_FILES = ["calendar.csv", "sales_train_validation.csv", "sell_prices.csv"]


def download_file(url, output_path):
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path
    cmd = ["curl", "-L", "--fail", "--retry", "3", "-o", output_path, url]
    subprocess.check_call(cmd)
    return output_path


def download_m5(raw_dir, url_base):
    os.makedirs(raw_dir, exist_ok=True)
    paths = []
    for name in M5_FILES:
        paths.append(download_file("%s/%s" % (url_base.rstrip("/"), name), os.path.join(raw_dir, name)))
    return paths
