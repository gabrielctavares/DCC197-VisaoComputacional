import json
import logging
import os
import subprocess
from datetime import datetime

EXPERIMENTS_FOLDER = "./experiments/"

def run_experiment(exp):
    cmd = [
        "python", "src/train.py",
        "--id", exp["id"],
        "--model_name", exp["model_name"],
        "--epochs", str(exp["epochs"]),
        "--batch_size", str(exp["batch_size"]),        
        "--img_size", str(exp["img_size"]),
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

    if exp.get("freeze_backbone", False):
        cmd += ["--freeze_backbone", "True"]
        cmd += ["--unfreeze_last_n_params", str(exp.get("unfreeze_last_n_params", 0))]

    logging.info("\n▶ Executando experimento: %s", exp["id"])
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logging.error("Erro ao executar experimento %s: %s", exp["id"], e)


def main():
    run_output = "./run_outputs"
    
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s', handlers=[
        logging.FileHandler(os.path.join(run_output, f"experiment{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ])
      
    if not os.path.exists(EXPERIMENTS_FOLDER):
        logging.error(f"Pasta de experimentos '{EXPERIMENTS_FOLDER}' não encontrada.")
        return
    experiments_files = [f for f in os.listdir(EXPERIMENTS_FOLDER) if f.endswith('.json')]
    
    if not experiments_files:
        logging.error(f"Nenhum arquivo de experimento encontrado na pasta '{EXPERIMENTS_FOLDER}'.")
        return
    
    for exp_file in experiments_files:
        exp_path = os.path.join(EXPERIMENTS_FOLDER, exp_file)
        with open(exp_path, 'r') as f:
            exp = json.load(f)
            run_experiment(exp)


if __name__ == "__main__":
    main()
