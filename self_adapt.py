import os
import torch
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import time
import pandas as pd

from copy import deepcopy
from utils.eval_att import *
from itertools import chain
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    normalized_mutual_info_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.mixture import GaussianMixture
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils import *
from config import *

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

transformation = Autoencoder(640, 640, 640)
transformation.load_state_dict(torch.load("autoencoder.pth"))
transformation.eval()

X_train, X_test, y_train, y_test = get_datasets(data_sources, prefix=prefix)
if FixedTransform:
    with torch.no_grad():
        X_train = transformation(torch.tensor(X_train)).numpy()
        X_test = transformation(torch.tensor(X_test)).numpy()

print(
    f"labels_count = {np.unique(y_train, return_counts=True), np.unique(y_test, return_counts=True)}"
)


print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print()

models = {}
unique_labels = np.arange(initial_n_known)
scores = np.zeros((X_test.shape[0], len(unique_labels)))
print(f"unique_labels: {unique_labels}")
print()

for label in unique_labels:
    X_train_label = X_train[y_train == label][:250]
    print(f"X_train_label: {X_train_label.shape}, {label}, {data_sources[label]}")
    gmm_model = GaussianMixture(n_components=5, covariance_type="full")
    gmm_model.fit(X_train_label)
    models[label] = gmm_model
    scores[:, label] = gmm_model.score_samples(X_test)

max_known_label = initial_n_known - 1

results, lists = get_metrics(scores, y_test, max_known_label, initial_n_known)
tpr, fpr, thresholds = lists["tpr"], lists["fpr"], lists["thresholds"]
diff = tpr - fpr
best_idx = np.argmax(diff)
best_threshold = thresholds[best_idx]
print(f"Best threshold: {best_threshold:.4f}")

outlier_scores = np.max(scores, axis=1)  # Use max decision score for each sample
negated_outlier_scores = -outlier_scores
ood_labels = (y_test > max_known_label).astype(int)
ood_decisions = (negated_outlier_scores >= best_threshold).astype(int)

cm = confusion_matrix(ood_labels, ood_decisions)
tn, fp, fn, tp = cm.ravel()
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
acc = (tp + tn) / (tp + tn + fp + fn)
balanced_acc = 0.5 * (tp / (tp + fn) + tn / (tn + fp))
print(f"Accuracy of OOD detection: {acc:.4f}")
print(f"TPR: {tpr:.4f}")
print(f"FPR: {fpr:.4f}")
results["ood_b_acc"] = round(balanced_acc, 4)
results["ood_acc"] = round(acc, 4)
results["ood_tpr"] = round(tpr, 4)
results["ood_fpr"] = round(fpr, 4)
results_dict = {"initial": results}
print(f"initial: {results_dict['initial']}")

for k in range(initial_n_known, len(data_sources) - 1):
    emerging_source = data_sources[k]
    print()
    print("=====================================================================")
    print(f"Emerging Source: {emerging_source}")
    print("=====================================================================")
    max_known_label = k - 1
    print(f"max_known_label: {max_known_label}")
    X_emerging, y_emerging = (
        X_train[y_train == k][:RANDOM_SAMPLES_SIZE],
        y_train[y_train == k][:RANDOM_SAMPLES_SIZE],
    )

    print(f"X_emerging: {X_emerging.shape}, y_emerging: {y_emerging.shape}")

    X_test_known = X_test[y_test <= max_known_label]
    y_test_known = y_test[y_test <= max_known_label]
    samples_size = (
        RANDOM_SAMPLES_SIZE
        if len(X_test_known) > RANDOM_SAMPLES_SIZE
        else len(X_test_known)
    )
    random_indices = np.random.choice(len(X_test_known), samples_size, replace=False)
    X_emerging = np.concatenate([X_emerging, X_test_known[random_indices]])
    y_emerging = np.concatenate([y_emerging, y_test_known[random_indices]])
    ood_labels = (y_emerging > max_known_label).astype(int)

    print(
        f"X_emerging: {X_emerging.shape}, y_emerging: {y_emerging.shape}, ood_labels: {ood_labels.shape}"
    )

    scores_emerging = np.zeros((X_emerging.shape[0], len(models)))
    for i, key in enumerate(models):
        model = models[key]
        scores_emerging[:, i] = model.score_samples(X_emerging)
    outlier_scores = np.max(scores_emerging, axis=1)
    negated_outlier_scores = -outlier_scores
    ood_decisions = (negated_outlier_scores >= best_threshold).astype(int)

    cm = confusion_matrix(ood_labels, ood_decisions)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    acc = (tp + tn) / (tp + tn + fp + fn)
    balanced_acc = 0.5 * (tp / (tp + fn) + tn / (tn + fp))
    print(f"Accuracy of OOD detection on the emerging dataset: {acc:.4f}")
    print(f"TPR: {tpr:.4f}")
    print(f"FPR: {fpr:.4f}")

    print()
    predicted_odd_data = X_emerging[ood_decisions == 1]
    ood_data_true_labels = y_emerging[ood_decisions == 1]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(predicted_odd_data)

    selected_cluster = None

    all_distances = euclidean_distances(predicted_odd_data)
    all_var = all_distances.var()

    S = 1
    for j in range(S, N_TRIALS + S):
        kmeans = KMeans(n_clusters=j, n_init=10)
        kmeans.fit(predicted_odd_data)
        kmean_preds = kmeans.predict(predicted_odd_data)
        # check if exists a cluster that is sufficiently tight by calculating the variance of distances to the cluster center
        cluster_centers = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_
        cluster_sizes = [np.sum(cluster_labels == i) for i in range(j)]
        cluster_variances = []

        all_distances = []

        for cluster in range(j):
            cluster_mask = cluster_labels == cluster
            cluster_center = cluster_centers[cluster]
            cluster_points = predicted_odd_data[cluster_mask]
            distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
            all_distances.extend(distances)
            variance = np.var(distances)
            cluster_variances.append(variance)

        cluster_variances = [v / np.var(all_distances) for v in cluster_variances]

        print(f"Trial {j - S}")
        print(f"Cluster variances: {cluster_variances}")
        print(f"Cluster sizes: {cluster_sizes}")

        # print(
        #     f"Silhouette score: {silhouette_score(predicted_odd_data, cluster_labels)}"
        # )
        print(f"NMI: {normalized_mutual_info_score(ood_data_true_labels, kmean_preds)}")

        selected_cluster = None
        adapted_v_threshold = VARIANCE_THRESHOLD  # + (j - S) * VARIANCE_ADAPTIVE_COEFF
        adapted_size_threshold = SIZE_THRESHOLD / ((j + 0) / SIZE_ADAPTIVE_COEFF)
        print(
            f"Adapted var. threshold: {adapted_v_threshold}, Adapted Size threshold: {adapted_size_threshold}"
        )
        cluster_variances_below_threshold = [
            (i, v) for i, v in enumerate(cluster_variances) if v <= adapted_v_threshold
        ]
        cluster_variances_below_threshold = sorted(
            cluster_variances_below_threshold, key=lambda x: x[1]
        )
        print(f"Cluster variances below threshold: {cluster_variances_below_threshold}")
        # find the cluster with the smallest variance
        if len(cluster_variances_below_threshold) > 0:
            # iteratively find the next smallest variance cluster and check if it is above the size threshold
            for i, v in cluster_variances_below_threshold:
                if cluster_sizes[i] > adapted_size_threshold:
                    selected_cluster = i
                    break
            if selected_cluster is not None:
                break
        else:
            print(f"No clusters have variance below threshold")
        print()

    if selected_cluster is None:
        print(
            f"No new source was identified among OOD samples for emerging source {emerging_source}"
        )
        print()
        new_label = max(unique_labels) + 1
        unique_labels = np.append(unique_labels, new_label)
        models[new_label] = DummyModel()
        max_known_label += 1
        scores = np.zeros((X_test.shape[0], len(models)))
        for i, key in enumerate(models):
            model = models[key]
            scores[:, i] = model.score_samples(X_test)

        results, lists = get_metrics(scores, y_test, max_known_label, initial_n_known)
        results["ood_b_acc"] = round(balanced_acc, 4)
        results["ood_acc"] = round(acc, 4)
        results["ood_tpr"] = round(tpr, 4)
        results["ood_fpr"] = round(fpr, 4)
        new_key = k
        results_dict[data_sources[k] + "_undetected"] = results
        for key in results_dict:
            print(f"{key}: {results_dict[key]}")
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].set_xlabel("PCA component 1")
        ax[0].set_ylabel("PCA component 2")
        ax[0].set_title(f"Selected cluster: {selected_cluster}")
        ax[1].set_xlabel("PCA component 1")
        ax[1].set_ylabel("PCA component 2")
        ax[1].set_title(f"True labels for emerging source {emerging_source}")
        sns.scatterplot(
            x=X_pca[:, 0], y=X_pca[:, 1], hue=kmean_preds, palette="tab10", s=25, ax=ax[0]
        )
        sns.scatterplot(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            hue=ood_data_true_labels,
            palette="tab10",
            s=25,
            ax=ax[1],
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"figures/{k}_{data_sources[k]}_undetected_selected_cluster_true.png")
        continue
    print(
        f"Selected cluster: {selected_cluster} with variance: {cluster_variances[selected_cluster]} and size: {cluster_sizes[selected_cluster]}"
    )
    # tpr = len(kmean_preds[kmean_preds == selected_cluster]) / len(
    #     ood_data_true_labels[ood_data_true_labels == max_known_label + 1]
    # )
    # print(f"TPR of the selected cluster: {tpr:.4f}")
    # print()

    selected_cluster_mask = cluster_labels == selected_cluster
    selected_cluster_points = predicted_odd_data[selected_cluster_mask]
    selected_cluster_labels = ood_data_true_labels[selected_cluster_mask]
    selected_cluster_size = len(selected_cluster_points)

    # Plot the selected cluster
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].set_xlabel("PCA component 1")
    ax[0].set_ylabel("PCA component 2")
    ax[0].set_title(f"Selected cluster: {selected_cluster}")
    ax[1].set_xlabel("PCA component 1")
    ax[1].set_ylabel("PCA component 2")
    ax[1].set_title(f"True labels for emerging source {emerging_source}")
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1], hue=kmean_preds, palette="tab10", s=25, ax=ax[0]
    )
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=ood_data_true_labels,
        palette="tab10",
        s=25,
        ax=ax[1],
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{k}_{data_sources[k]}_selected_cluster_true.png")

    new_label = max(unique_labels) + 1
    unique_labels = np.append(unique_labels, new_label)
    models[new_label] = GaussianMixture(n_components=5, covariance_type="full")
    models[new_label].fit(selected_cluster_points)
    max_known_label += 1

    scores = np.zeros((X_test.shape[0], len(models)))
    for i, key in enumerate(models):
        model = models[key]
        scores[:, i] = model.score_samples(X_test)

    results, lists = get_metrics(scores, y_test, max_known_label, initial_n_known)
    results["ood_b_acc"] = round(balanced_acc, 4)
    results["ood_acc"] = round(acc, 4)
    results["ood_tpr"] = round(tpr, 4)
    results["ood_fpr"] = round(fpr, 4)
    new_key = k
    results_dict[data_sources[k]] = results


    # Convert results_dict to a DataFrame and save it to a CSV file
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv("results_dict.csv", index_label="Metric")

    for key in results_dict:
        print(f"{key}: {results_dict[key]}")

    print(f"max_known_label: {max_known_label}")
    print()
