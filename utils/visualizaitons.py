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
        xticklabels=labels if labels is not None else "auto",
        yticklabels=labels if labels is not None else "auto",
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
    emerging_label,
    log_dir,
):

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
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
    plt.savefig(os.path.join(log_dir, "visuals", f"{emerging_label}_{emerging_source}_clustering.png"))
    plt.show()
    plt.close()


def plot_cm_cluster_before_after(
    cm_before,
    cm_labels_before,
    cm_after,
    cm_labels_after,
    emerging_source,
    emerging_label,
    log_dir,
    filename=None,
):

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(
        f"Confusion matrices before and after update for emerging source {emerging_source} with label: {cm_labels_before[-1]}"
    )
    sns.heatmap(
        cm_before,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax[0],
        cbar=False,
        xticklabels=cm_labels_before,
        yticklabels=cm_labels_before,
    )
    sns.heatmap(
        cm_after,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax[1],
        cbar=False,
        xticklabels=cm_labels_after,
        yticklabels=cm_labels_after,
    )

    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("True")
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("True")

    ax[0].set_title(f"Confusion Matrix before update for {emerging_source}")
    ax[1].set_title(f"Confusion Matrix after update for {emerging_source}")

    filename = filename if filename else f"{emerging_label}_{emerging_source}_cm_before_after"
    plt.savefig(os.path.join(log_dir, "visuals", f"{filename}.png"))
    plt.show()
    plt.close()


def plot_tsne_before_after_ae(
    X_train,
    y_train,
    X_test,
    y_test,
    autoencoder,
    emerging_source,
    emerging_label,
    log_dir,    
):
    
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f"Data transformation for emerging source {emerging_source}")
    for aa in ax:
        for a in aa:
            a.set_xlabel("PCA component 1")
            a.set_ylabel("PCA component 2")
    ax[0][0].set_title(f"Autoencoder training data")
    ax[0][1].set_title(f"Transformed Autoencoder training data")
    ax[1][0].set_title(f"Test data")
    ax[1][1].set_title(f"Transformed Test data")

    X_train_tsne = TSNE(n_components=2).fit_transform(X_train)
    X_train_transformed = autoencoder.embed(X_train)
    X_train_transformed_tsne = TSNE(n_components=2).fit_transform(X_train_transformed)

    X_test_tsne = TSNE(n_components=2).fit_transform(X_test)
    X_test_transformed = autoencoder.embed(X_test)
    X_test_transformed_tsne = TSNE(n_components=2).fit_transform(X_test_transformed)

    sns.scatterplot(
        x=X_train_tsne[:, 0],
        y=X_train_tsne[:, 1],
        hue=y_train,
        palette="tab10",
        s=25,
        ax=ax[0][0],
    )
    sns.scatterplot(
        x=X_train_transformed_tsne[:, 0],
        y=X_train_transformed_tsne[:, 1],
        hue=y_train,
        palette="tab10",
        s=25,
        ax=ax[0][1],
    )
    sns.scatterplot(
        x=X_test_tsne[:, 0],
        y=X_test_tsne[:, 1],
        hue=y_test,
        palette="tab10",
        s=25,
        ax=ax[1][0],
    )
    sns.scatterplot(
        x=X_test_transformed_tsne[:, 0],
        y=X_test_transformed_tsne[:, 1],
        hue=y_test,
        palette="tab10",
        s=25,
        ax=ax[1][1],
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "visuals", f"{emerging_label}_{emerging_source}_tsne_before_after_ae.png"))
    plt.show()
    plt.close()