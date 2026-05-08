import os
import subprocess


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run(cmd):
    print(">>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def main():
    run(["python", "experiments/run_layer1_synthetic.py", "--config", "configs/default.yaml"])
    run(["python", "experiments/download_m5.py"])
    run(["python", "experiments/run_m5_full.py", "--config", "configs/default.yaml", "--n-series", "1000"])
    run(["python", "experiments/run_ablation.py", "--dataset", "m5", "--n-series", "1000"])
    run(["python", "experiments/run_sensitivity.py", "--dataset", "m5", "--n-series", "1000"])


if __name__ == "__main__":
    main()
