import json
import subprocess
from datetime import datetime

EXPERIMENTS_PART1_FILE = "experiments/experiments_part1.json"
EXPERIMENTS_PART2_FILE = "experiments/experiments_part2.json"

def run_experiment(exp):
    cmd = [
        "python", "src/train.py",
        "--model_name", exp["model_name"],
        "--epochs", str(exp["epochs"]),
        "--batch_size", str(exp["batch_size"]),
        "--learning_rate", str(exp["learning_rate"]),
        "--weight_decay", str(exp["weight_decay"]),
        "--results_file", f"results_{datetime.now().strftime('%Y%m%d')}.xlsx"
    ]

    # flags opcionais
    if exp.get("use_dropout", False):
        cmd += ["--use_dropout", "True", "--dropout_rate", str(exp.get("dropout_rate", 0.5))]

    if exp.get("use_batch_norm", False):
        cmd += ["--use_batch_norm", "True"]

    if exp.get("use_data_augmentation", False):
        cmd += ["--use_data_augmentation", "True"]

    if exp.get("freeze_features", False):
        cmd += ["--freeze_features", "True"]
        cmd += ["--unfreeze_last_n_layers", str(exp.get("unfreeze_last_n_layers", 0))]

    print("\n▶ Executando experimento:", exp["name"])
    subprocess.run(cmd, check=True)


def main():
    with open(EXPERIMENTS_PART1_FILE) as f:
        experiments = json.load(f)
    
    with open(EXPERIMENTS_PART2_FILE) as f:
        experiments += json.load(f)

    for exp in experiments:
        run_experiment(exp)


if __name__ == "__main__":
    main()
