import os
import torch
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import time

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
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils import *
from configs import *
import pandas as pd
import warnings

warnings.simplefilter("ignore")  # Ignore all warnings

np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)


def refine_cluster(selected_cluster_points, labels, gmms, threshold, factor=1.2):
    scores_emerging = np.zeros((selected_cluster_points.shape[0], initial_n_known))
    for i, key in enumerate(gmms):
        model = gmms[key]
        scores_emerging[:, i] = model.score_samples(selected_cluster_points)
    outlier_scores = np.max(scores_emerging, axis=1)
    negated_outlier_scores = -outlier_scores
    ood_decisions = (negated_outlier_scores >= threshold * factor).astype(int)
    prev_size = selected_cluster_points.shape[0]
    refined_cluster_points = selected_cluster_points[ood_decisions == 1]
    refined_cluster_labels = labels[ood_decisions == 1]
    new_size = refined_cluster_points.shape[0]
    print(f"Refined cluster size: {prev_size} -> {new_size}")
    return refined_cluster_points, refined_cluster_labels


X_train, X_test, y_train, y_test = get_datasets(data_sources, prefix=prefix)
X_train_ae = X_train[y_train < initial_n_known]
y_train_ae = y_train[y_train < initial_n_known]
X_train_ae, X_val_ae, y_train_ae, y_val_ae = train_test_split(
    X_train_ae, y_train_ae, test_size=0.2, random_state=42
)
saved_autoencoder = "models/autoencoder.pth"
if use_autoencoder:
    if os.path.exists(saved_autoencoder) and not do_retrain:
        init_auto_encoder = torch.load(saved_autoencoder)
        print("Loaded pretrained autoencoder")
    else:
        print("Training initial autoencoder")
        init_auto_encoder = train_autoencoder(
            X_train_ae,
            y_train_ae,
            X_val_ae,
            y_val_ae,
            source_name="initial",
            initial_n_known=initial_n_known,
            max_known_label=initial_n_known,
            kwargs=training_kwargs,
        )
        os.makedirs("models", exist_ok=True)
        torch.save(init_auto_encoder, saved_autoencoder)
else:
    init_auto_encoder = torch.nn.Identity()

current_auto_encoder = init_auto_encoder
# X_train, y_train, X_val, y_val = X_train_ae, X_val_ae, y_train_ae, y_val_ae

with torch.no_grad():
    X_train = init_auto_encoder(torch.tensor(X_train)).numpy()
    X_test = init_auto_encoder(torch.tensor(X_test)).numpy()

print(f"labels_count = {np.unique(y_train, return_counts=True)}", end=" ")
print(np.unique(y_test, return_counts=True))

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print()

models = {}
unique_labels = np.arange(initial_n_known)
scores = np.zeros((X_test.shape[0], len(unique_labels)))
print(f"unique_labels: {unique_labels}")
print()

for label in unique_labels:
    # random_indices = np.random.choice(len(X_train[y_train == label]), 250, replace=False)
    X_train_label = X_train[y_train == label]
    print(f"X_train_label: {X_train_label.shape}, {label}, {data_sources[label]}")
    gmm_model = BayesianGaussianMixture(
        n_components=training_kwargs["n_components"],
        covariance_type=training_kwargs["cov_type"],
    )
    gmm_model.fit(X_train_label)
    models[label] = gmm_model
    scores[:, label] = gmm_model.score_samples(X_test)

initial_models = deepcopy(models)

max_known_label = initial_n_known - 1

results, lists = get_metrics(
    scores, y_test, max_known_label, initial_n_known, is_rotation=False
)
tprs, fprs, thresholds = lists["tpr"], lists["fpr"], lists["thresholds"]

acceptable_thresholds = []
for fpr, tpr, threshold in zip(fprs, tprs, thresholds):
    if tpr >= min_ood_tpr:
        acceptable_thresholds.append((fpr, tpr, threshold))

# Step 2: Select the threshold with the lowest FPR
if acceptable_thresholds:
    best_threshold = min(acceptable_thresholds, key=lambda x: x[0])  # x[0] is the FPR
    selected_fpr, selected_tpr, selected_threshold = best_threshold
    print(f"Selected Threshold: {selected_threshold}, FPR: {selected_fpr}, TPR: {selected_tpr}")
else:
    print("No acceptable threshold found.")

diff = tprs - fprs
best_idx = np.argmax(diff)
best_threshold = selected_threshold
# best_threshold = thresholds[best_idx] * ood_threshold_factor
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
# results["ood_b_acc"] = round(balanced_acc, 4)
results["ood_acc"] = round(acc, 4)
results["ood_tpr"] = round(tpr, 4)
results["ood_fpr"] = round(fpr, 4)
results_dict = {"initial": results}
print(f"initial: {results_dict['initial']}")
# plot the initial data with the true labels
tsne = TSNE(n_components=2)
pca = PCA(n_components=2)
X_test_init = X_test  # [y_test < initial_n_known]
y_test_init = y_test  # [y_test < initial_n_known]
X_tsne_init = tsne.fit_transform(X_test_init)
fig, ax = plt.subplots(1, figsize=(6, 6))
ax.set_xlabel("TSNE component 1")
ax.set_ylabel("TSNE component 2")
ax.set_title("Initial data")
sns.scatterplot(
    x=X_tsne_init[:, 0],
    y=X_tsne_init[:, 1],
    hue=y_test_init,
    palette="tab10",
    s=25,
    ax=ax,
)
plt.legend()
plt.tight_layout()
plt.savefig("figures/0_initial_data.png")

# creat a subdirectory in results/ with a unique name
current_time = time.strftime("%Y%m%d-%H%M%S")
os.makedirs(f"results/'continual'-{current_time}", exist_ok=True)

for k in range(initial_n_known, len(data_sources) - 1):
    current_models = deepcopy(initial_models)
    emerging_source = data_sources[k]
    print()
    print("=====================================================================")
    print(f"Emerging Source: {emerging_source}")
    print("=====================================================================")
    # input(f"Press Enter to continue...")
    max_known_label = k - 1
    print(f"max_known_label: {max_known_label}")
    X_emerging, y_emerging = (
        X_train[y_train == k][:RANDOM_SAMPLES_SIZE],
        y_train[y_train == k][:RANDOM_SAMPLES_SIZE],
    )

    X_test_known = X_train[y_test < initial_n_known]
    y_test_known = y_train[y_test < initial_n_known]
    samples_size = (
        RANDOM_SAMPLES_SIZE
        if len(X_test_known) > RANDOM_SAMPLES_SIZE
        else len(X_test_known)
    )
    random_indices = np.random.choice(len(X_test_known), samples_size, replace=False)
    X_emerging = np.concatenate([X_emerging, X_test_known[random_indices]])
    y_emerging = np.concatenate([y_emerging, y_test_known[random_indices]])
    print(f"X_emerging: {X_emerging.shape}, y_emerging: {y_emerging.shape}")

    # ood_labels = (y_emerging > max_known_label).astype(int)
    ood_labels = (
        (y_emerging >= initial_n_known) & (y_emerging != max_known_label)
    ).astype(int)

    print(
        f"X_emerging: {X_emerging.shape}, y_emerging: {y_emerging.shape}, ood_labels: {ood_labels.shape}"
    )

    scores_emerging = np.zeros((X_emerging.shape[0], len(current_models)))
    for i, key in enumerate(current_models):
        model = current_models[key]
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
    tsne = TSNE(n_components=2)
    X_pca = tsne.fit_transform(predicted_odd_data)

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
                if cluster_sizes[i] >= adapted_size_threshold:
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
        unique_labels = np.arange(new_label)
        print(f"unique_labels: {unique_labels}")
        current_models[new_label] = DummyModel()
        max_known_label += 1
        scores = np.zeros((X_test.shape[0], len(current_models)))
        for i, key in enumerate(current_models):
            model = current_models[key]
            scores[:, i] = model.score_samples(X_test)

        results, lists = get_metrics(
            scores, y_test, max_known_label, initial_n_known, is_rotation=False
        )
        # results["ood_b_acc"] = round(balanced_acc, 4)
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
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            hue=kmean_preds,
            palette="tab10",
            s=25,
            ax=ax[0],
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
        plt.savefig(
            f"figures/{k}_{data_sources[k]}_undetected_selected_cluster_true.png"
        )
        continue

    print(
        f"Selected cluster: {selected_cluster} with variance: {cluster_variances[selected_cluster]} and size: {cluster_sizes[selected_cluster]}"
    )
    selected_cluster_mask = cluster_labels == selected_cluster
    selected_cluster_points = predicted_odd_data[selected_cluster_mask]
    selected_cluster_labels = ood_data_true_labels[selected_cluster_mask]
    selected_cluster_size = len(selected_cluster_points)
    refined_cluster_points, refined_cluster_labels = refine_cluster(
        selected_cluster_points,
        selected_cluster_labels,
        current_models,
        best_threshold,
        factor=refine_factor,
    )
    all_labels = np.arange(len(data_sources) - 1)
    scores_cluster = np.zeros((refined_cluster_points.shape[0], len(current_models)))

    for i, key in enumerate(current_models):
        model = current_models[key]
        scores_cluster[:, i] = model.score_samples(refined_cluster_points)
    voted_scores = np.argmax(scores_cluster, axis=1)

    cm_before = confusion_matrix(
        refined_cluster_labels, voted_scores, labels=all_labels
    )

    # plot the selected cluster with the true labels
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2)
    X_pca_refined = tsne.fit_transform(refined_cluster_points)
    X_pca_selected = tsne.fit_transform(selected_cluster_points)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].set_xlabel("PCA component 1")
    ax[0].set_ylabel("PCA component 2")
    ax[1].set_title(f"Refined cluster")
    ax[1].set_xlabel("PCA component 1")
    ax[1].set_ylabel("PCA component 2")
    ax[0].set_title(
        f"True labels of selected cluster for emerging source {emerging_source}"
    )
    sns.scatterplot(
        x=X_pca_refined[:, 0],
        y=X_pca_refined[:, 1],
        hue=refined_cluster_labels,
        palette="tab10",
        s=25,
        ax=ax[1],
    )
    sns.scatterplot(
        x=X_pca_selected[:, 0],
        y=X_pca_selected[:, 1],
        hue=selected_cluster_labels,
        palette="tab10",
        s=25,
        ax=ax[0],
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{k}_{data_sources[k]}_selected_cluster_refined.png")

    max_known_label += 1
    new_label = max_known_label
    unique_labels = np.append(unique_labels, new_label)
    print(f"unique_labels: {unique_labels}")
    X_train_ae = np.concatenate(
        [X_train[y_train < max_known_label], selected_cluster_points]
    )
    y_train_ae = np.concatenate(
        [y_train[y_train < max_known_label], np.ones(selected_cluster_size) * new_label]
    )
    # make sure there are the same number of samples for each class
    class_counts = Counter(y_train_ae)
    min_class_count = min(class_counts.values())
    new_X_train_ae = []
    new_y_train_ae = []
    for label in unique_labels:
        mask = y_train_ae == label
        new_X_train_ae.extend(X_train_ae[mask][:min_class_count])
        new_y_train_ae.extend(y_train_ae[mask][:min_class_count])
    X_train_ae = np.array(new_X_train_ae)
    y_train_ae = np.array(new_y_train_ae)
    X_train_ae, X_val_ae, y_train_ae, y_val_ae = train_test_split(
        X_train_ae, y_train_ae, test_size=0.2, random_state=42
    )
    print(
        f"X_train_ae: {X_train_ae.shape}, y_train_ae: {y_train_ae.shape}, X_val_ae: {X_val_ae.shape}, y_val_ae: {y_val_ae.shape}"
    )
    print(f"new_label: {new_label}")
    print(f"unique labels train: {np.unique(y_train_ae, return_counts=True)}")
    print(f"unique labels val: {np.unique(y_val_ae, return_counts=True)}")

    if use_autoencoder:
        auto_encoder = train_autoencoder(
            X_train_ae,
            y_train_ae,
            X_val_ae,
            y_val_ae,
            source_name=data_sources[k],
            # pretrained_model=current_auto_encoder,
            max_known_label=max_known_label,
            initial_n_known=initial_n_known,
            kwargs=training_kwargs,
        )
    else:
        auto_encoder = torch.nn.Identity()
    current_auto_encoder = auto_encoder

    # Plot the selected cluster
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
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
        hue=ood_data_true_labels,
        palette="tab10",
        s=25,
        ax=ax[1],
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{k}_{data_sources[k]}_selected_cluster_true.png")

    # Transform all data into the new embedding space using the learned auto_encoder
    with torch.no_grad():
        X_train_ae_transformed = (
            auto_encoder(torch.tensor(X_train_ae)).numpy()
            if use_autoencoder
            else X_train_ae
        )
        X_test_transformed = (
            auto_encoder(torch.tensor(X_test)).numpy() if use_autoencoder else X_test
        )

    X_train_ae_pca = tsne.fit_transform(X_train_ae)
    X_train_ae_transformed_pca = tsne.fit_transform(X_train_ae_transformed)
    X_test_pca = tsne.fit_transform(X_test[y_test <= max_known_label])
    X_test_ae_transformed_pca = tsne.fit_transform(
        X_test_transformed[y_test <= max_known_label]
    )
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

    sns.scatterplot(
        x=X_train_ae_pca[:, 0],
        y=X_train_ae_pca[:, 1],
        hue=y_train_ae,
        palette="tab10",
        s=25,
        ax=ax[0][0],
    )
    sns.scatterplot(
        x=X_train_ae_transformed_pca[:, 0],
        y=X_train_ae_transformed_pca[:, 1],
        hue=y_train_ae,
        palette="tab10",
        s=25,
        ax=ax[0][1],
    )
    sns.scatterplot(
        x=X_test_pca[:, 0],
        y=X_test_pca[:, 1],
        hue=y_test[y_test <= max_known_label],
        palette="tab10",
        s=25,
        ax=ax[1][0],
    )
    sns.scatterplot(
        x=X_test_ae_transformed_pca[:, 0],
        y=X_test_ae_transformed_pca[:, 1],
        hue=y_test[y_test <= max_known_label],
        palette="tab10",
        s=25,
        ax=ax[1][1],
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{k}_{data_sources[k]}_data_transformed.png")
    # plot X_train_ae before and after transformation on the same figure
    # pca = PCA(n_components=2)
    # tsne = TSNE(
    #     n_components=2,
    # )
    # X_train_ae_pca = tsne.fit_transform(X_train_ae)
    # X_train_ae_transformed_pca = tsne.fit_transform(X_train_ae_transformed)
    # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # ax[0].set_xlabel("PCA component 1")
    # ax[0].set_ylabel("PCA component 2")
    # ax[0].set_title(
    #     f"Training Data used for Autoencoder when emerging source is {emerging_source}"
    # )
    # ax[1].set_xlabel("PCA component 1")
    # ax[1].set_ylabel("PCA component 2")
    # ax[1].set_title(
    #     f"Transformed Training Data when emerging source is {emerging_source}"
    # )
    # sns.scatterplot(
    #     x=X_train_ae_pca[:, 0],
    #     y=X_train_ae_pca[:, 1],
    #     hue=y_train_ae,
    #     palette="tab10",
    #     s=25,
    #     ax=ax[0],
    # )
    # sns.scatterplot(
    #     x=X_train_ae_transformed_pca[:, 0],
    #     y=X_train_ae_transformed_pca[:, 1],
    #     hue=y_train_ae,
    #     palette="tab10",
    #     s=25,
    #     ax=ax[1],
    # )
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"figures/{k}_{data_sources[k]}_X_train_ae_transformed.png")

    # plot X_test before and after transformation on the same figure
    # x_test_pca = tsne.fit_transform(X_test)
    # X_test_ae_transformed_pca = tsne.fit_transform(X_test_transformed)
    # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # sns.scatterplot(
    #     x=x_test_pca[:, 0],
    #     y=x_test_pca[:, 1],
    #     hue=y_test,
    #     palette="tab10",
    #     s=25,
    #     ax=ax[0],
    # )
    # sns.scatterplot(
    #     x=X_test_ae_transformed_pca[:, 0],
    #     y=X_test_ae_transformed_pca[:, 1],
    #     hue=y_test,
    #     palette="tab10",
    #     s=25,
    #     ax=ax[1],
    # )
    # for a in ax:
    #     a.set_xlabel("PCA component 1")
    #     a.set_ylabel("PCA component 2")
    # ax[0].set_title(f"X_test")
    # ax[1].set_title(f"X_test_transformed")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"figures/{k}_{data_sources[k]}_X_test_transformed.png")

    # Recreate GMMs with the transformed data
    if not do_nothing:
        new_models = {}
        for label in unique_labels:
            X_train_label_transformed = X_train_ae_transformed[y_train_ae == label]
            gmm_model = BayesianGaussianMixture(
                n_components=training_kwargs["n_components"],
                covariance_type=training_kwargs["cov_type"],
            )
            gmm_model.fit(X_train_label_transformed)
            new_models[label] = gmm_model

        # Update current models with the new GMMs
        current_models = new_models

    with torch.no_grad():
        refined_cluster_transformed = auto_encoder(
            torch.tensor(refined_cluster_points)
        ).numpy()
    scores_cluster = np.zeros((refined_cluster_points.shape[0], len(current_models)))
    for i, key in enumerate(current_models):
        model = current_models[key]
        scores_cluster[:, i] = model.score_samples(refined_cluster_transformed)
    voted_scores = np.argmax(scores_cluster, axis=1)

    cm_after = confusion_matrix(refined_cluster_labels, voted_scores, labels=all_labels)
    # save cm visualization
    fig, ax_cm = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Confusion matrices for emerging source {emerging_source}")
    sns.heatmap(
        cm_before,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=all_labels,
        yticklabels=all_labels,
        ax=ax_cm[0],
    )
    ax_cm[0].set_title(f"CM before updating GMMs for source {emerging_source}")
    ax_cm[0].set_xlabel("Predicted")
    ax_cm[0].set_ylabel("True")
    sns.heatmap(
        cm_after,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=all_labels,
        yticklabels=all_labels,
        ax=ax_cm[1],
    )
    ax_cm[1].set_title(f"CM after updating GMMs for source {emerging_source}")
    ax_cm[1].set_xlabel("Predicted")
    ax_cm[1].set_ylabel("True")
    plt.savefig(f"figures/{k}_{data_sources[k]}_refined_cluster_confusion_matrix.png")

    # Calculate scores with the new GMMs
    scores = np.zeros((X_test_transformed.shape[0], len(current_models)))
    for i, key in enumerate(current_models):
        model = current_models[key]
        scores[:, i] = model.score_samples(X_test_transformed)

    # X_test = X_test_transformed

    results, lists = get_metrics(
        scores,
        y_test,
        max_known_label,
        initial_n_known,
        is_rotation=False,
    )
    # results["ood_b_acc"] = round(balanced_acc, 4)
    results["ood_acc"] = round(acc, 4)
    results["ood_tpr"] = round(tpr, 4)
    results["ood_fpr"] = round(fpr, 4)
    new_key = k
    results_dict[data_sources[k]] = results
    cm = lists["cm"]

    # plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion matrix for emerging source {emerging_source}")
    plt.savefig(f"figures/{k}_{data_sources[k]}_confusion_matrix.png")

    # Convert results_dict to a DataFrame and save it to a CSV file
    results_df = pd.DataFrame(results_dict)
    # print average of the accuracy over the three last rows
    results_df.to_csv(
        f"results/'continual'-{current_time}/results_dict.csv", index_label="Metric"
    )

    for key in results_dict:
        print(f"{key}: {results_dict[key]}")

    print(f"max_known_label: {max_known_label}")
    print()
