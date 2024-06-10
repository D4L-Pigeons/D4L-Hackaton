from argparse import Namespace
from itertools import cycle

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.preprocessing import label_binarize

from models.building_blocks import Block


def evaluate_classification(y_true, y_pred, y_proba=None):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    class_report = classification_report(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "class_report": class_report,
        "conf_matrix": conf_matrix,
        "mse": mse,
        "mae": mae,
        "roc_auc": roc_auc,
    }

    print(metrics)
    return metrics


def plot_roc_auc(y_true, y_pred, num_classes):  # what to do with this?
    y_true_bin = label_binarize(y_true, classes=[i for i in range(num_classes)])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(num_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label="ROC curve of class {0} (area = {1:0.2f})" "".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()


def plot_clustering(data, y_pred):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=y_pred, palette="viridis")
    plt.title("Clustering Results")
    plt.show()


def evaluate_clustering(y_true, y_pred, data):
    silhouette = silhouette_score(data, y_pred)  # figure out what data is
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    metrics = {
        "silhouette_score": silhouette,
        "adjusted_rand_index": ari,
        "normalized_mutual_info": nmi,
    }
    print(metrics)
    plot_clustering(data, y_pred)
    return metrics


def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    data = []

    with torch.no_grad():
        for batch in test_loader:
            x_fst, x_snd, labels = batch
            data.append(
                torch.cat([x_fst, x_snd], dim=1)
            )  # check this and if matches labels
            logits = model.predict(x_fst, x_snd)
            predictions = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    classification_metrics = evaluate_classification(y_true, y_pred)
    clustering_metrics = evaluate_clustering(y_true, y_pred, data)
    # plot_roc_auc(y_true, y_proba)


def apply_classification_head(model, cfg: Namespace, data):
    if not cfg.classification_head:
        classification_head = Block(
            input_size=cfg.latent_dim,
            output_size=cfg.num_classes,
            hidden_size=cfg.num_classes * 2,
            batch_norm=cfg.batch_norm,
        )
        logits = classification_head(data)
    else:
        logits = model._predict(data)
    pass
