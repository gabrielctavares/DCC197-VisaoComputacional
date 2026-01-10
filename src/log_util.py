import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import Counter
from sklearn.metrics import confusion_matrix


def save_results_to_excel(file_path, row_data):
    import pandas as pd
    try:
        df = pd.read_excel(file_path, sheet_name="resultados")
    except FileNotFoundError:
        df = pd.DataFrame(columns=row_data.keys())

    df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
    df.to_excel(file_path, sheet_name="resultados", index=False)
    logging.info(f"✅ Resultados salvos em {file_path}")


def display_class_distribution(type, dataset, emotion_table):
    labels = np.array(dataset.targets)
    class_counts = np.bincount(labels, minlength=len(emotion_table))
    logging.info(f"{type} class distribution:")
    total = len(dataset)
    for idx, count in enumerate(class_counts):
        cname = emotion_table[idx]
        perc = (count / total) * 100 if total > 0 else 0
        logging.info(f"  {cname:10s}: {count} ({perc:.2f}%)")

    logging.info(f"{type} dataset size: {total}\n")


def plot_confusion_matrix(cm, class_names, title="Matriz de Confusão", save_path=None):
    """Gera e retorna um gráfico Matplotlib a partir de uma matriz de confusão já calculada."""

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    tick_marks = np.arange(len(class_names))

    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    thresh = cm_norm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        value = cm_norm[i, j] * 100
        text = f"{value:.2f}".replace(".", ",") + "%"

        ax.text(
            j, i, text,
            ha="center",
            va="center",
            fontsize=8,
            color="white" if cm_norm[i, j] > thresh else "black"
        )

    ax.set_ylabel("Real")
    ax.set_xlabel("Predito")

    fig.tight_layout()        
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        logging.info(f"✅ Matriz de confusão salva em: {save_path}")

    return fig
