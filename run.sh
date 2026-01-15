#!/bin/bash

EXPERIMENTS_FOLDER="./experiments"
RUN_OUTPUT="./run_outputs"
DATE=$(date +%Y%m%d)
LOG_FILE="$RUN_OUTPUT/experiment_$DATE.log"
RESULTS_FILE="results_$DATE.xlsx"

ID_FILTER=""

# -----------------------------
# Parse argumentos
# -----------------------------
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --id) ID_FILTER="$2"; shift ;;
    *) echo "Argumento desconhecido: $1"; exit 1 ;;
  esac
  shift
done

# -----------------------------
# Setup inicial
# -----------------------------
mkdir -p "$RUN_OUTPUT"

echo "[INFO] $(date) - Iniciando execução" | tee -a "$LOG_FILE"

if [[ ! -d "$EXPERIMENTS_FOLDER" ]]; then
  echo "[ERROR] Pasta '$EXPERIMENTS_FOLDER' não encontrada." | tee -a "$LOG_FILE"
  exit 1
fi

EXPERIMENT_FILES=$(ls "$EXPERIMENTS_FOLDER"/*.json 2>/dev/null)

if [[ -z "$EXPERIMENT_FILES" ]]; then
  echo "[ERROR] Nenhum arquivo de experimento encontrado." | tee -a "$LOG_FILE"
  exit 1
fi

# -----------------------------
# Loop de experimentos
# -----------------------------
for FILE in $EXPERIMENT_FILES; do
  echo "[INFO] $(date) - Lendo arquivo $FILE" | tee -a "$LOG_FILE"

  COUNT=$(jq length "$FILE")

  for ((i=0; i<COUNT; i++)); do
    EXP_ID=$(jq -r ".[$i].id" "$FILE")

    if [[ -n "$ID_FILTER" && "$ID_FILTER" != "$EXP_ID" ]]; then
      continue
    fi

    MODEL_NAME=$(jq -r ".[$i].model_name" "$FILE")
    EPOCHS=$(jq -r ".[$i].epochs" "$FILE")
    BATCH_SIZE=$(jq -r ".[$i].batch_size" "$FILE")
    IMG_SIZE=$(jq -r ".[$i].img_size" "$FILE")
    LR=$(jq -r ".[$i].learning_rate" "$FILE")
    WD=$(jq -r ".[$i].weight_decay" "$FILE")

    USE_DROPOUT=$(jq -r ".[$i].use_dropout // false" "$FILE")
    DROPOUT_RATE=$(jq -r ".[$i].dropout_rate // 0.5" "$FILE")

    USE_BN=$(jq -r ".[$i].use_batch_norm // false" "$FILE")
    USE_AUG=$(jq -r ".[$i].use_data_augmentation // false" "$FILE")

    FREEZE=$(jq -r ".[$i].freeze_backbone // false" "$FILE")
    UNFREEZE_N=$(jq -r ".[$i].unfreeze_last_n_params // 0" "$FILE")

    CMD=(
      python src/train.py
      --id "$EXP_ID"
      --model_name "$MODEL_NAME"
      --epochs "$EPOCHS"
      --batch_size "$BATCH_SIZE"
      --img_size "$IMG_SIZE"
      --learning_rate "$LR"
      --weight_decay "$WD"
      --results_file "$RESULTS_FILE"
    )

    [[ "$USE_DROPOUT" == "true" ]] && CMD+=(--use_dropout True --dropout_rate "$DROPOUT_RATE")
    [[ "$USE_BN" == "true" ]] && CMD+=(--use_batch_norm True)
    [[ "$USE_AUG" == "true" ]] && CMD+=(--use_data_augmentation True)
    [[ "$FREEZE" == "true" ]] && CMD+=(--freeze_backbone True --unfreeze_last_n_params "$UNFREEZE_N")

    echo "[INFO] $(date) - Executando experimento $EXP_ID" | tee -a "$LOG_FILE"
    echo "[CMD] ${CMD[*]}" | tee -a "$LOG_FILE"

    "${CMD[@]}" >> "$LOG_FILE" 2>&1
    STATUS=$?

    if [[ $STATUS -ne 0 ]]; then
      echo "[ERROR] $(date) - Experimento $EXP_ID falhou (exit code $STATUS)" | tee -a "$LOG_FILE"
      exit 1
    fi

    echo "[INFO] $(date) - Experimento $EXP_ID finalizado com sucesso" | tee -a "$LOG_FILE"
  done
done

echo "[INFO] $(date) - Todos os experimentos concluídos" | tee -a "$LOG_FILE"
