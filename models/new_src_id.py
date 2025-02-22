from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import normalized_mutual_info_score


def identify_new_sources(
    predicted_ood_data,
    ood_data_true_labels,
    emerging_source_name,
    num_trials,
    v_threshold,
    size_threshold,
    size_adaptive_coeff,
):

    s = 1
    for j in range(s, num_trials + s):
        kmeans = KMeans(n_clusters=j, n_init=10)
        kmeans.fit(predicted_ood_data)
        kmean_preds = kmeans.predict(predicted_ood_data)
        # check if exists a cluster that is sufficiently tight by calculating the variance of distances to the cluster center
        cluster_centers = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_
        cluster_sizes = [np.sum(cluster_labels == i) for i in range(j)]
        cluster_variances = []

        all_distances = []

        for cluster in range(j):
            cluster_mask = cluster_labels == cluster
            cluster_center = cluster_centers[cluster]
            cluster_points = predicted_ood_data[cluster_mask]
            distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
            all_distances.extend(distances)
            variance = np.var(distances)
            cluster_variances.append(variance)

        cluster_variances = [v / np.var(all_distances) for v in cluster_variances]

        print(f"Trial {j - s}")
        print(f"Cluster variances: {cluster_variances}")
        print(f"Cluster sizes: {cluster_sizes}")

        # print(
        #     f"Silhouette score: {silhouette_score(predicted_odd_data, cluster_labels)}"
        # )
        print(f"NMI: {normalized_mutual_info_score(ood_data_true_labels, kmean_preds)}")

        selected_cluster = None
        adapted_v_threshold = v_threshold  # + (j - S) * VARIANCE_ADAPTIVE_COEFF
        adapted_size_threshold = size_threshold / ((j + 0) / size_adaptive_coeff)
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
            f"No new source was identified among OOD samples for emerging source {emerging_source_name}"
        )


    selected_cluster_mask = cluster_labels == selected_cluster
    selected_cluster_points = predicted_ood_data[selected_cluster_mask]
    selected_cluster_labels = ood_data_true_labels[selected_cluster_mask]
    selected_cluster_size = len(selected_cluster_points)

    return selected_cluster_points, selected_cluster_labels, kmean_preds, selected_cluster, selected_cluster_mask
