from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import normalized_mutual_info_score


def trimmed_variance_multivariate(data, trim_fraction=0.1, center_func=np.mean):
    """
    Computes a robust variance (covariance matrix and overall variance)
    for multivariate data by trimming the highest distance points from the cluster center.

    Parameters:
        data (np.array): A 2D array of shape (n_samples, n_features).
        trim_fraction (float): The fraction (0 < trim_fraction < 1) of points to trim based on the
                               highest distances from the cluster center. For example, 0.1 will trim the top 10%.
        center_func (callable): Function to compute the center of the cluster (default: np.mean).
                                You can pass np.median for a more robust center estimate.

    Returns:
        cov_matrix (np.array): The covariance matrix computed on the trimmed data.
        overall_variance (float): A scalar measure of variance (trace of the covariance matrix).
        trimmed_data (np.array): The data points that remain after trimming.
    """
    if data.ndim != 2:
        raise ValueError("data must be a 2D array with shape (n_samples, n_features)")

    # Compute the cluster center
    center = center_func(data, axis=0)

    # Compute Euclidean distances from the center
    distances = np.linalg.norm(data - center, axis=1)

    # Determine the threshold distance below which points are kept
    # For example, if trim_fraction=0.1, we keep the 90% closest points.
    threshold_distance = np.percentile(distances, 100 * (1 - trim_fraction))

    # Create a mask to select the inlier points
    inlier_mask = distances <= threshold_distance
    trimmed_data = data[inlier_mask]

    # Compute covariance matrix on the trimmed data
    cov_matrix = np.cov(trimmed_data, rowvar=False)

    # Compute a scalar overall variance as the trace of the covariance matrix
    overall_variance = np.trace(cov_matrix)

    return overall_variance, cov_matrix, trimmed_data


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
            # cluster_trimmed_variance, _, _ = trimmed_variance_multivariate(
            #     cluster_points, trim_fraction=0.05, center_func=np.median
            # )
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
            (i, v, cluster_sizes[i])
            for i, v in enumerate(cluster_variances)
            if v <= adapted_v_threshold
        ]
        cluster_variances_below_threshold = sorted(
            cluster_variances_below_threshold, key=lambda x: x[2], reverse=True
        )
        print(f"Cluster variances below threshold: {cluster_variances_below_threshold}")
        # find the cluster with the smallest variance
        if len(cluster_variances_below_threshold) > 0:
            # iteratively find the next smallest variance cluster and check if it is above the size threshold
            for i, v, _ in cluster_variances_below_threshold:
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

    return (
        selected_cluster_points,
        selected_cluster_labels,
        kmean_preds,
        selected_cluster,
        selected_cluster_mask,
    )
