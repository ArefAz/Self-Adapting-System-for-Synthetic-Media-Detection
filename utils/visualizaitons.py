import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def plot_embeddings_tsne(X, y, log_dir, title="Embeddings", filename=None):

    X_embedded = TSNE(n_components=2).fit_transform(X)
    sns.scatterplot(
        x=X_embedded[:, 0],
        y=X_embedded[:, 1],
        hue=y,
        palette="tab10",
        s=25,
    )
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    filename = filename if filename else title
    plt.savefig(os.path.join(log_dir, "visuals", f"{filename}.png"))
    plt.show()
    plt.close()


def plot_confusion_matrix(
    cm, log_dir, labels=None, title="Confusion Matrix", filename=None
):

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    filename = filename if filename else title
    plt.savefig(os.path.join(log_dir, "visuals", f"{filename}.png"))
    plt.show()
    plt.close()


def plot_clustering_results(
    predicted_ood_data,
    predicted_ood_true_labels,
    kmean_preds,
    selected_cluster,
    emerging_source,
    log_dir,
):

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    X_pca = PCA(n_components=2).fit_transform(predicted_ood_data)
    ax[0].set_xlabel("PCA component 1")
    ax[0].set_ylabel("PCA component 2")
    ax[0].set_title(
        f"Selected cluster: {selected_cluster} among OOD samples for emerging source {emerging_source}"
    )
    ax[1].set_xlabel("PCA component 1")
    ax[1].set_ylabel("PCA component 2")
    ax[1].set_title(
        f"True labels among OOD samples for emerging source {emerging_source}"
    )
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1], hue=kmean_preds, palette="tab10", s=25, ax=ax[0]
    )
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=predicted_ood_true_labels,
        palette="tab10",
        s=25,
        ax=ax[1],
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "visuals", f"{emerging_source}_clustering.png"))
    plt.show()
    plt.close()
