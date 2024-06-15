from argparse import Namespace
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import sklearn
import torch
from pytorch_lightning import LightningModule
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    auc,
    normalized_mutual_info_score,
    roc_curve,
    silhouette_score,
)
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.nn import functional as F

from models.building_blocks import Block
from utils.paths import CONFIG_PATH, RESULTS_PATH
import os


class ClassificationModel(pl.LightningModule):
    def __init__(self, latent_dim, num_classes, do_batch_norm=False):
        super().__init__()
        self.classification_head = Block(
            input_size=latent_dim,
            output_size=num_classes,
            hidden_size=num_classes * 2,
            batch_norm=do_batch_norm,
        )
        self.loss = nn.CrossEntropyLoss()  # Adjust

    def forward(self, z):
        return self.classification_head(z)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        # Calculate metrics here
        # preds = F.softmax(y_hat, dim=1)  # Adjust for other activation functions
        # self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict(self, z):
        y_hat = self(z)
        return y_hat

    def configure_optimizers(self):
        # Define optimizer here
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.001
        )  # Adjust learning rate
        return optimizer


from torch.utils.data import DataLoader, TensorDataset


def train_classification_head(model, data):
    print("train_classification_head...")
    latent_representation = model.predict(data)
    print(latent_representation.shape)
    latent_dim = latent_representation.shape[1]
    num_classes = len(data.obs["cell_type"].cat.categories)
    print("Number of classes is:", num_classes)
    print("Init classification head...")
    classification_head = ClassificationModel(latent_dim, num_classes)
    print("Train classification head...")
    # Create trainer here
    trainer = pl.Trainer(max_epochs=1)  # Adjust epochs as needed
    print(torch.tensor(data.obs["cell_type"].cat.codes.values, dtype=torch.long))
    curr_dataset = TensorDataset(
        latent_representation,
        torch.tensor(data.obs["cell_type"].cat.codes.values, dtype=torch.long),
    )
    dataloader = DataLoader(curr_dataset, batch_size=128, shuffle=False)
    trainer.fit(classification_head, dataloader)  # Pass data as a PyTorch DataLoader
    return classification_head


def get_predictions_gt_classes(model, classification_head, data):
    print("Predict latent")
    test_latent_representation = model.predict(data)
    print("Predict classification head")
    prediction = classification_head.predict(test_latent_representation)
    print("Finished prediction")
    prediction_probability = torch.softmax(prediction, dim=1)
    ground_truth = data.obs["cell_type"].cat.codes.values
    classes = data.obs["cell_type"].cat.categories
    return (
        prediction.detach().numpy(),
        prediction_probability,
        ground_truth,
        classes,
        test_latent_representation.detach().numpy(),
    )


def evaluate_clustering(y_true, y_pred, data):
    #     silhouette = silhouette_score(data, y_pred)  # figure out what data is
    #     print("Silhouette finished")
    #     # ari = adjusted_rand_score(y_true, y_pred)
    #     # print("Ari finished")
    #     # nmi = normalized_mutual_info_score(y_true, y_pred)
    #     # print("Normalized MI finished")
    #     metrics = {
    #         "silhouette_score": silhouette,
    #         # "adjusted_rand_index": ari,
    #         # "normalized_mutual_info": nmi,
    #     }
    #     print(metrics)
    plot_clustering(data, y_pred)
    # return metrics


def get_metrics(model, classification_head, test_data):
    (
        prediction,
        prediction_probability,
        ground_truth,
        classes,
        test_latent_representation,
    ) = get_predictions_gt_classes(model, classification_head, test_data)
    metrics = evaluate_clustering(ground_truth, prediction, test_data)

    metrics["f1_score_per_cell_type"] = sklearn.metrics.f1_score(
        ground_truth, prediction, labels=classes, average=None
    )
    metrics["f1_score"] = sklearn.metrics.f1_score(
        ground_truth, prediction, labels=classes, average="macro"
    )
    metrics["accuracy"] = sklearn.metrics.accuracy_score(ground_truth, prediction)
    metrics[
        "average_precision_per_cell_type"
    ] = sklearn.metrics.average_precision_score(
        ground_truth, prediction_probability, average=None
    )
    metrics["roc_auc_per_cell_type"] = sklearn.metrics.roc_auc_score(
        ground_truth,
        prediction_probability,
        multi_class="ovr",
        average=None,
        labels=classes,
    )
    plot_roc_auc(ground_truth, prediction_probability, len(classes))
    metrics["confusion_matrix"] = sklearn.metrics.confusion_matrix(
        ground_truth, prediction, labels=classes
    )
    return metrics


def calculate_metrics(model, train_data, test_data):
    classification_head = train_classification_head(model, train_data)
    print("Finished training classification head.")
    metrics = get_metrics(model, classification_head, test_data)
    return metrics


def latent_metrics(model, test_data):
    latent_representation = model.predict(test_data)
    ground_truth = test_data.obs["cell_type"]
    # todo: umap + clustering


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
    import scanpy as sc
    from sklearn.decomposition import PCA

    # Extract the latent embedding
    latent_embedding = GEX_anndata.obsm["latent_embedding"]

    # Perform PCA on the latent embedding
    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latent_embedding)

    # Store the PCA results back in the AnnData object
    GEX_anndata.obsm["X_pca_latent"] = latent_pca
    GEX_anndata.uns["pca_latent_variance_ratio"] = pca.explained_variance_ratio_

    # Create a new AnnData object for visualization purposes
    pca_gex_anndata = sc.AnnData(X=latent_pca)
    pca_gex_anndata.obs = GEX_anndata.obs
    pca_gex_anndata.var_names = [f"PC{i+1}" for i in range(latent_pca.shape[1])]

    # Visualize the PCA results
    sc.pl.scatter(
        pca_gex_anndata, x="PC1", y="PC2", color="level1", title="GEX latent by PCA"
    )

    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=y_pred, palette="viridis")
    # plt.title("Clustering Results")
    # plt.show()


import argparse
from utils.data_utils import load_anndata
from train import load_config, create_model
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Validate model")
    parser.add_argument(
        "--path",
        help="Path to load config from.",
    )
    parser.add_argument(
        "--method",
        choices=["omivae", "babel", "advae", "vae"],
        help="Name of the method to use.",
    )
    parser.add_argument(
        "--config",
        default="standard",
        help="Name of a configuration in src/config/{method}.yaml.",
    )
    args = parser.parse_args()

    config = load_config(args)
    model = create_model(args, config)

    model.load(args.path)

    # train_data = load_anndata(
    #     mode="train",
    #     normalize=config.normalize,
    #     remove_batch_effect=config.remove_batch_effect,
    #     target_hierarchy_level=config.target_hierarchy_level,
    #     preload_subsample_frac=None,
    # )

    test_data = load_anndata(
        mode="test",
        normalize=config.normalize,
        remove_batch_effect=config.remove_batch_effect,
        target_hierarchy_level=config.target_hierarchy_level,
        preload_subsample_frac=None,
    )

    latent_test = model.predict(test_data)

    results_path = RESULTS_PATH / args.method
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    results_path = RESULTS_PATH / args.method / f"{args.config}"
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    torch.save(latent_test, str(results_path / "latent_test"))

    # metrics_dict = calculate_metrics(model, train_data, test_data)
    # pd.DataFrame.from_dict(metrics_dict).to_csv(args.path + "metrics.csv")


if __name__ == "__main__":
    main()
