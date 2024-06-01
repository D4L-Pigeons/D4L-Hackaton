from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from models.building_blocks import Block


def silhouette(X, labels):
    silhouette_avg = silhouette_score(X, labels)
    print(f"Silhouette Score: {silhouette_avg}")


def visualise_clustering(X):
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis")
    plt.title("PCA of Clustered Data")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar()
    plt.show()


def compare_results():
    pass


def apply_classification_head(model, cfg: Namespace, data):
    if not cfg.classification_head:
        classification_head = Block(
            input_size=cfg.latent_dim,
            output_size=cfg.num_classes,
            hidden_size=cfg.num_classes * 2,
            batch_norm=cfg.batch_norm,
        )
        logits = classification_head(mu)
    else:
        logits = model._predict(data)
    pass
