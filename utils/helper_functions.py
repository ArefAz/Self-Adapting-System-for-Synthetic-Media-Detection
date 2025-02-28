from models import (
    OpenSetModel,
    identify_new_sources,
    train_autoencoder,
    identify_new_sources_iter,
)
from .visualizaitons import *
from copy import deepcopy
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances, confusion_matrix
import numpy as np


def propose_an_update(
    emerging_source,
    emerging_label,
    X_emerging,
    y_emerging,
    learned_X,
    learned_X_val,
    learned_y,
    learned_y_val,
    open_set_model,
    current_ae,
    datasets,
    validation_set,
    results_dict,
    validation_results_dict,
    log_dir,
    configs,
):

    emerging_buffer_X = X_emerging[y_emerging <= emerging_label]
    emerging_buffer_y = y_emerging[y_emerging <= emerging_label]

    emerging_buffer_X = np.concatenate(
        (
            emerging_buffer_X,
            datasets["init_known"]["learning"][0],
        )
    )
    emerging_buffer_y = np.concatenate(
        (
            emerging_buffer_y,
            datasets["init_known"]["learning"][1],
        )
    )

    print(
        f"Number of unique samples in emerging buffer {np.unique(emerging_buffer_y, return_counts=True)}"
    )

    ood_binary_labels = np.concatenate(
        (
            np.ones_like(y_emerging[y_emerging == emerging_label]),
            np.zeros_like(y_emerging[y_emerging < emerging_label]),
            np.zeros_like(datasets["init_known"]["learning"][1]),
        )
    )

    ood_decisions = open_set_model.ood_detect(current_ae.embed(emerging_buffer_X))
    if ood_binary_labels.shape != ood_decisions.shape:
        raise ValueError(
            f"Shapes of ood_binary_labels and ood_decisions do not match: {ood_binary_labels.shape} vs {ood_decisions.shape}"
        )
    results_ood = open_set_model.evaluate_ood(ood_decisions, ood_binary_labels)

    predicted_ood_data = current_ae.embed(emerging_buffer_X[ood_decisions == 1])
    predicted_ood_true_labels = emerging_buffer_y[ood_decisions == 1]

    print(f"Number of Emerging samples: {len(emerging_buffer_X)}")
    print(
        f"Number of predicted OOD samples: {len(predicted_ood_data)}, number of true OOD samples: {ood_binary_labels.sum()}"
    )
    print(
        f"OOD FPR: {results_ood['ood_fpr']}, OOD TPR: {results_ood['ood_tpr']}, OOD ACC: {results_ood['ood_acc']}"
    )
    for key, value in results_ood.items():
        if key == "ood_cm":
            continue
        results_dict[emerging_source][key] = value

    # clustering_results = identify_new_sources_iter(
    #     predicted_ood_data,
    #     emerging_buffer_y[ood_decisions == 1],
    #     emerging_source_name=emerging_source,
    #     emerging_source_label=emerging_label,
    #     num_trials=configs["num_trials"],
    #     v_threshold=configs["v_threshold"],
    #     size_threshold=configs["size_threshold"],
    #     size_adaptive_coeff=configs["size_adaptive_coeff"],
    #     min_samples=configs["training_kwargs"]["cluster_min_samples"],
    # )

    dists = pairwise_distances(predicted_ood_data)
    upper_bound = np.percentile(dists, 60)
    eps_values = np.linspace(5, 20, configs["num_trials"])
    found_good_cluster = False
    for eps in eps_values:
        print(f"Trying DBSCAN with eps={round(eps, 2)}")
        dbscan = DBSCAN(eps=eps, min_samples=configs["training_kwargs"]["cluster_min_samples"])
        dbscan.fit(predicted_ood_data)
        cluster_preds = dbscan.labels_
        unique_clusters = [lab for lab in np.unique(cluster_preds) if lab != -1]
        n_clusters = len(np.unique(cluster_preds))
        if n_clusters <= 1:
            print(f"DBSCAN with eps={round(eps, 2)} resulted in {n_clusters} cluster")
            continue
        for cluster_id in unique_clusters:
            cluster_mask = cluster_preds == cluster_id
            cluster_points = predicted_ood_data[cluster_mask]

            tmp_open_set_model = deepcopy(open_set_model)
            tmp_ae = deepcopy(current_ae)
            if len(cluster_points) <= configs["size_threshold"]:
                # print(f"Cluster {cluster_id} is too small")
                print(f"Cluster {cluster_id} is too small with {len(cluster_points)} points. Trying another cluster")
                continue
            else:
                print(f"Trying to update the model with cluster {cluster_id} with {len(cluster_points)} points")

            y_true = (predicted_ood_true_labels == emerging_label)
            y_pred = cluster_mask
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn)
            tpr = tp / (tp + fn)
            clustering_metrics = {
                "tpr": round(tpr, 4),
                "fpr": round(fpr, 4),
            }
            
            # for key, value in clustering_metrics.items():
            #     results_dict[emerging_source][f"cluster_{key}"] = value

            identified_new_src = emerging_buffer_X[ood_decisions == 1][cluster_mask]
            identified_new_src_labels = emerging_buffer_y[ood_decisions == 1][cluster_mask]
            old_os_model = deepcopy(open_set_model)
            old_ae = deepcopy(current_ae)

            new_src_assigned_labels = np.ones_like(identified_new_src_labels) * open_set_model.n_known_sources
            new_X, new_X_val, new_y, new_y_val = train_test_split(
                identified_new_src, new_src_assigned_labels, test_size=0.25, random_state=configs["random_seed"]
            )
            learned_X.extend(new_X)
            learned_y.extend(new_y)
            learned_X_val.extend(new_X_val)
            learned_y_val.extend(new_y_val)

            X_ae = np.concatenate(
                (
                    datasets["init_known"]["train"][0],
                    np.array(learned_X),
                )
            )
            y_ae = np.concatenate(
                (
                    datasets["init_known"]["train"][1],
                    np.array(learned_y),
                )
            )
            X_ae_val = np.concatenate(
                (
                    datasets["init_known"]["val"][0],
                    np.array(learned_X_val),
                )
            )
            y_ae_val = np.concatenate(
                (
                    datasets["init_known"]["val"][1],
                    np.array(learned_y_val),
                )
            )

            X_ae_test = np.concatenate(
                (
                    datasets["init_known"]["test"][0],
                    datasets["emerging"]["test"][0][
                        datasets["emerging"]["test"][1] <= emerging_label
                    ],
                )
            )
            y_ae_test = np.concatenate(
                (
                    datasets["init_known"]["test"][1],
                    datasets["emerging"]["test"][1][
                        datasets["emerging"]["test"][1] <= emerging_label
                    ],
                )
            )
            class_counts = Counter(y_ae)
            min_class_count = min(class_counts.values())
            new_X_ae = []
            new_y_ae = []
            for label in np.unique(y_ae):
                cluster_mask = y_ae == label
                new_X_ae.extend(X_ae[cluster_mask][:min_class_count])
                new_y_ae.extend(y_ae[cluster_mask][:min_class_count])
            X_ae = np.array(new_X_ae)
            y_ae = np.array(new_y_ae)

            # class_counts_val = Counter(y_ae_val)
            # min_class_count_val = min(class_counts_val.values())
            # new_X_ae_val = []
            # new_y_ae_val = []
            # for label in np.unique(y_ae_val):
            #     cluster_mask = y_ae_val == label
            #     new_X_ae_val.extend(X_ae_val[cluster_mask][:min_class_count_val])
            #     new_y_ae_val.extend(y_ae_val[cluster_mask][:min_class_count_val])
            # X_ae_val = np.array(new_X_ae_val)
            # y_ae_val = np.array(new_y_ae_val)

            tmp_ae = train_autoencoder(
                X_ae,
                y_ae,
                X_ae_val,
                y_ae_val,
                source_name=emerging_source,
                log_dir=log_dir,
                training_kwargs=configs["training_kwargs"],
            )

            tmp_open_set_model = OpenSetModel(
                n_components=configs["training_kwargs"]["n_components"],
                covariance_type=configs["training_kwargs"]["cov_type"],
                is_bayesian=configs["training_kwargs"]["is_bayesian"],
                min_ood_tpr=configs["min_ood_tpr"],
            )
            tmp_open_set_model.fit(
                tmp_ae.embed(X_ae),
                y_ae,
            )

            validation_dataset_X = tmp_ae.embed(X_ae_val)
            validation_dataset_y = y_ae_val
            tmp_open_set_model.find_best_thresholds(
                validation_dataset_X,
                validation_dataset_y,
            )

            validation_results = tmp_open_set_model.evaluate(
                validation_dataset_X,
                validation_dataset_y,
            )

            if "prev" in validation_results_dict:
                prev_validation_results = validation_results_dict["prev"]
            else:
                prev_validation_results = validation_results_dict["initial"]
            prev_acc = prev_validation_results.get("balanced_accuracy")
            acc = validation_results.get("balanced_accuracy")
            current_acc = validation_results.get("balanced_accuracy_current")
            prev_det_acc = prev_validation_results.get("detection_accuracy")
            current_det_acc = validation_results.get("detection_accuracy")
            delta = prev_acc - acc
            delta_det = prev_det_acc - current_det_acc
            print(f"Validation results for selected cluster {cluster_id}: {validation_results}")
            if (delta <= 0.10 and delta_det <= 0.10) or (delta <= 0.15 and delta_det <= 0.15 and current_acc > 0.70):
                print(f"Validation results for selected cluster {cluster_id} are GOOD enough, UPDATING the model")
                validation_results_dict["prev"] = validation_results
                open_set_model = tmp_open_set_model
                current_ae = tmp_ae
                found_good_cluster = True
                cluster_preds_final = cluster_preds
                break
            else:
                print(f"Validation results for selected cluster {cluster_id} are NOT GOOD enough")
            
                print("Previous results:")
                for results_key in prev_validation_results:
                    if results_key != "confusion_matrix":
                        print(f"{results_key}: {prev_validation_results[results_key]}")
                continue
        if found_good_cluster:
            break
                


    if not found_good_cluster:
        raise ValueError("No good cluster found")

    # Plotting
    plot_clustering_results(
        predicted_ood_data,
        predicted_ood_true_labels,
        cluster_preds_final,
        cluster_id,
        clustering_metrics,
        emerging_source,
        emerging_label,
        log_dir,
    )
    plot_tsne_before_after_ae(
        X_ae,
        y_ae,
        X_ae_test,
        y_ae_test,
        current_ae,
        emerging_source,
        emerging_label,
        log_dir,
    )
    cm_before, cm_labels_before = old_os_model.get_cm_labels(
        old_ae.embed(identified_new_src),
        identified_new_src_labels,
        add_new_source=True,
    )
    cm_after, cm_labels_after = open_set_model.get_cm_labels(
        current_ae.embed(identified_new_src), identified_new_src_labels
    )
    plot_cm_cluster_before_after(
        cm_before,
        cm_labels_before,
        cm_after,
        cm_labels_after,
        emerging_source,
        emerging_label,
        log_dir,
    )

    return (
        open_set_model,
        current_ae,
        learned_X,
        learned_y,
        identified_new_src,
        identified_new_src_labels,
        old_os_model,
        old_ae,
        results_dict,
    )
