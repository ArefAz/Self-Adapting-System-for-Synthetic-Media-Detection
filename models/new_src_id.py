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


def identify_new_sources0(
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


def identify_new_sources(
    predicted_ood_data,
    ood_data_true_labels,
    emerging_source_name,
    emerging_source_label,
    num_trials,
    v_threshold,
    size_threshold,
    size_adaptive_coeff,
    min_samples=5,
):
    import numpy as np
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import normalized_mutual_info_score, pairwise_distances
    from sklearn.metrics import accuracy_score, confusion_matrix

    selected_cluster = None
    cluster_labels = None  # will hold the latest DBSCAN labels

    # Set up a range of eps values.
    # Here we compute pairwise distances and take a 90th percentile as an upper bound.
    dists = pairwise_distances(predicted_ood_data)
    upper_bound = np.percentile(dists, 90)
    # We avoid eps=0 by starting at a small positive value.
    eps_values = np.linspace(0.1, upper_bound, num_trials)

    for trial, eps in enumerate(eps_values, start=1):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(predicted_ood_data)
        cluster_labels = dbscan.labels_
        # Exclude noise (label == -1)
        unique_clusters = [lab for lab in np.unique(cluster_labels) if lab != -1]

        print(f"Trial {trial} with eps = {eps:.3f}")
        if len(unique_clusters) == 0:
            print("No clusters found (only noise).")
            continue

        cluster_centers = {}
        cluster_sizes = {}
        cluster_variances = {}
        all_distances = []

        # Compute centers, sizes, and per-cluster variance.
        for cluster in unique_clusters:
            cluster_mask = cluster_labels == cluster
            cluster_points = predicted_ood_data[cluster_mask]
            # Compute centroid for the cluster.
            center = np.mean(cluster_points, axis=0)
            cluster_centers[cluster] = center
            cluster_sizes[cluster] = np.sum(cluster_mask)
            # Compute distances of points to centroid.
            distances = np.linalg.norm(cluster_points - center, axis=1)
            all_distances.extend(distances)
            cluster_variances[cluster] = np.var(distances)

        all_distances = np.array(all_distances)
        overall_variance = np.var(all_distances)

        # Normalize cluster variances for comparison.
        normalized_cluster_variances = {
            cluster: var / overall_variance for cluster, var in cluster_variances.items()
        }

        print(f"  Normalized cluster variances: {normalized_cluster_variances}")
        print(f"  Cluster sizes: {cluster_sizes}")

        # Set adaptive thresholds.
        adapted_v_threshold = v_threshold
        adapted_size_threshold = size_threshold
        print(
            f"  Adapted variance threshold: {adapted_v_threshold}, "
            f"Adapted size threshold: {adapted_size_threshold}"
        )

        # Identify candidate clusters whose normalized variance is below the threshold.
        candidate_clusters = [
            (cluster, normalized_cluster_variances[cluster], cluster_sizes[cluster])
            for cluster in unique_clusters
            if normalized_cluster_variances[cluster] <= adapted_v_threshold
        ]
        # Sort candidates by size (largest first)
        candidate_clusters = sorted(candidate_clusters, key=lambda x: x[1], reverse=False)
        print(f"  Candidate clusters: {candidate_clusters}")

        if candidate_clusters:
            for cluster, var, size in candidate_clusters:
                if size >= adapted_size_threshold:
                    selected_cluster = cluster
                    break

        if selected_cluster is not None:
            break
        print()  # For visual separation across trials

    if selected_cluster is None:
        print(
            f"No new source was identified among OOD samples for emerging source {emerging_source_name}"
        )
        # If no cluster meets criteria, return an empty mask.
        selected_cluster_mask = np.zeros(predicted_ood_data.shape[0], dtype=bool)
        selected_cluster_points = predicted_ood_data[selected_cluster_mask]
        selected_cluster_labels = ood_data_true_labels[selected_cluster_mask]
    else:
        selected_cluster_mask = cluster_labels == selected_cluster
        selected_cluster_points = predicted_ood_data[selected_cluster_mask]
        selected_cluster_labels = ood_data_true_labels[selected_cluster_mask]

    nmi = normalized_mutual_info_score(ood_data_true_labels, cluster_labels)
    y_true = (ood_data_true_labels == emerging_source_label)
    y_pred = selected_cluster_mask
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    acc = accuracy_score(y_true, y_pred)
    print(f"Clustering FPR: {fpr:.4f}, TPR: {tpr:.4f}, ACC: {acc:.4f}, nmi: {nmi:.4f}")
    # print selected cluster size and variance
    print(f"Selected cluster size: {len(selected_cluster_points)}, variance: {normalized_cluster_variances[selected_cluster]}")

    metrics = {
        "tpr": round(tpr, 4),
        "fpr": round(fpr, 4),
        "acc": round(acc, 4),
        "nmi": round(nmi, 4),
    }

    return (
        selected_cluster_points,
        selected_cluster_labels,
        cluster_labels,
        selected_cluster,
        selected_cluster_mask,
        metrics
    )


def identify_new_sources_new(
    predicted_ood_data,
    ood_data_true_labels,
    emerging_source_name,
    emerging_source_label,
    num_trials,
    size_threshold,
    size_adaptive_coeff,
    min_samples=10  # default value; adjust as needed
):
    import numpy as np
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import normalized_mutual_info_score, pairwise_distances
    from sklearn.metrics import accuracy_score, confusion_matrix

    selected_cluster = None
    cluster_labels = None  # will hold the latest DBSCAN labels

    # Set up a range of eps values.
    # Compute pairwise distances and take the 90th percentile as an upper bound.
    dists = pairwise_distances(predicted_ood_data)
    upper_bound = np.percentile(dists, 90)
    # Avoid eps=0 by starting at a small positive value.
    eps_values = np.linspace(0.1, upper_bound, num_trials)

    for trial, eps in enumerate(eps_values, start=1):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(predicted_ood_data)
        cluster_labels = dbscan.labels_
        # Exclude noise (label == -1)
        unique_clusters = [lab for lab in np.unique(cluster_labels) if lab != -1]

        print(f"Trial {trial} with eps = {eps:.3f}")
        if len(unique_clusters) == 0:
            print("  No clusters found (only noise).")
            continue

        # Compute the size of each cluster.
        cluster_sizes = {}
        for cluster in unique_clusters:
            cluster_mask = cluster_labels == cluster
            cluster_sizes[cluster] = np.sum(cluster_mask)

        # Compute the adapted size threshold (if desired).
        adapted_size_threshold = size_threshold / (trial / size_adaptive_coeff)
        print(f"  Adapted size threshold: {adapted_size_threshold}")
        print(f"  Cluster sizes: {cluster_sizes}")

        # Select the first cluster (in ascending order) that meets the size criterion.
        for cluster in sorted(unique_clusters):
            if cluster_sizes[cluster] >= adapted_size_threshold:
                selected_cluster = cluster
                print(f"  Selected cluster: {selected_cluster}")
                break

        if selected_cluster is not None:
            break  # Stop if a candidate cluster is found.
        print()  # Visual separation for trials.

    # If no cluster meets the criteria, return an empty mask.
    if selected_cluster is None:
        print(
            f"No new source was identified among OOD samples for emerging source {emerging_source_name}"
        )
        selected_cluster_mask = np.zeros(predicted_ood_data.shape[0], dtype=bool)
        selected_cluster_points = predicted_ood_data[selected_cluster_mask]
        selected_cluster_labels = ood_data_true_labels[selected_cluster_mask]
    else:
        selected_cluster_mask = cluster_labels == selected_cluster
        selected_cluster_points = predicted_ood_data[selected_cluster_mask]
        selected_cluster_labels = ood_data_true_labels[selected_cluster_mask]

    # Evaluate the clustering performance for the selected cluster.
    nmi = normalized_mutual_info_score(ood_data_true_labels, cluster_labels)
    y_true = (ood_data_true_labels == emerging_source_label)
    y_pred = selected_cluster_mask
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        # In case confusion_matrix returns a smaller array (e.g., no positives), set metrics to 0.
        tn = fp = fn = tp = 0
        tpr = fpr = 0.0
    acc = accuracy_score(y_true, y_pred)
    print(f"Clustering FPR: {fpr:.4f}, TPR: {tpr:.4f}, ACC: {acc:.4f}, NMI: {nmi:.4f}")
    print(f"Selected cluster size: {np.sum(selected_cluster_mask)}")

    metrics = {
        "tpr": round(tpr, 4),
        "fpr": round(fpr, 4),
        "acc": round(acc, 4),
        "nmi": round(nmi, 4),
    }

    return (
        selected_cluster_points,
        selected_cluster_labels,
        cluster_labels,
        selected_cluster,
        selected_cluster_mask,
        metrics,
    )
