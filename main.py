import os
import argparse
import shutil
import torch
import numpy as np
import pandas as pd

from collections import Counter
from copy import deepcopy
from utils.data_utils import get_datasets
from utils.visualizaitons import *
from utils import set_seeds
from log_tools import logger, save_results
from configs import load_configs
from models import OpenSetModel, train_autoencoder, identify_new_sources

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--config",
    "-c",
    type=str,
    default="configs/config.yaml",
    help="Path to the config file",
)
args = arg_parser.parse_args()
config_path = args.config


if __name__ == "__main__":

    configs = load_configs(config_path)
    log_dir = logger.initialize_logger(configs["base_log_dir"])
    print(f"Logging to {log_dir}")
    shutil.copy(config_path, log_dir)

    set_seeds(configs["random_seed"])

    results_dict = {}

    # Get datasets
    datasets = get_datasets(configs["data"])

    # Initialize GMMs and the AE, and the threshold
    initial_ae = train_autoencoder(
        datasets["init_known"]["train"][0],
        datasets["init_known"]["train"][1],
        source_name="init_known",
        log_dir=log_dir,
        training_kwargs=configs["training_kwargs"],
    )
    init_known_dataset = datasets["init_known"]["train"]

    open_set_model = OpenSetModel(
        n_components=configs["training_kwargs"]["n_components"],
        covariance_type=configs["training_kwargs"]["cov_type"],
        is_bayesian=configs["training_kwargs"]["is_bayesian"],
        min_ood_tpr=configs["min_ood_tpr"],
    )
    open_set_model.fit(
        initial_ae.embed(init_known_dataset[0]),
        init_known_dataset[1],
    )

    threshold_dataset_X = np.concatenate(
        (
            initial_ae.embed(datasets["init_known"]["train"][0]),
            # initial_ae.embed(datasets["emerging"]["test"][0]),
            initial_ae.embed(datasets["ood"][0]),
        )
    )
    threshold_dataset_y = np.concatenate(
        (
            datasets["init_known"]["train"][1],
            # datasets["emerging"]["test"][1],
            datasets["ood"][1],
        )
    )
    open_set_model.find_best_thresholds(
        threshold_dataset_X,
        threshold_dataset_y,
    )
    # open_set_model.set_threshold(init_threshold)

    validation_results = open_set_model.evaluate(
        initial_ae.embed(init_known_dataset[0]),
        init_known_dataset[1],
    )
    validation_results_dict = {"initial": validation_results}

    # initial evaluation
    eval_results = open_set_model.evaluate(
        initial_ae.embed(datasets["init_known"]["test"][0]),
        datasets["init_known"]["test"][1],
    )

    results_dict["initial"] = {}
    for key, value in eval_results.items():
        if key == "confusion_matrix":
            continue
        results_dict["initial"][key] = value

    save_results(results_dict, log_dir)

    # Add initial visualizations
    plot_labels = np.arange(-1, np.max(init_known_dataset[1]) + 1)
    plot_confusion_matrix(eval_results["confusion_matrix"], log_dir, labels=plot_labels, filename="0_initial_cm")

    plot_tsne_before_after_ae(
        init_known_dataset[0],
        init_known_dataset[1],
        datasets["init_known"]["test"][0],
        datasets["init_known"]["test"][1],
        initial_ae,
        "init_known",
        0,
        log_dir,
    )

    # Main loop
    X_emerging = datasets["emerging"]["learning"][0]
    y_emerging = datasets["emerging"]["learning"][1]
    emerging_labels = np.unique(y_emerging)
    current_ae = deepcopy(initial_ae)
    init_os_model = deepcopy(open_set_model)

    learned_X = []
    learned_y = []

    for i, emerging_label in enumerate(emerging_labels):
        emerging_source = configs["data"]["emerging_sources"][i]
        print()
        print("=====================================================================")
        print(f"Emerging Source: {emerging_source} with label {emerging_label}")
        print("=====================================================================")
        results_dict[emerging_source] = {}
        # input("Press Enter to continue...")

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

        print(f"Number of unique samples in emerging buffer {np.unique(emerging_buffer_y, return_counts=True)}")

        ood_binary_labels = np.concatenate(
            (
                np.ones_like(y_emerging[y_emerging == emerging_label]),
                np.zeros_like(y_emerging[y_emerging < emerging_label]),
                np.zeros_like(datasets["init_known"]["learning"][1]),
            )
        )


        ood_decisions = open_set_model.ood_detect(current_ae.embed(emerging_buffer_X))
        if ood_binary_labels.shape != ood_decisions.shape:
            raise ValueError(f"Shapes of ood_binary_labels and ood_decisions do not match: {ood_binary_labels.shape} vs {ood_decisions.shape}")
        results_ood = open_set_model.evaluate_ood(ood_decisions, ood_binary_labels)


        predicted_ood_data = current_ae.embed(emerging_buffer_X[ood_decisions == 1])
        predicted_ood_true_labels = emerging_buffer_y[ood_decisions == 1]

        print(f"Number of Emerging samples: {len(emerging_buffer_X)}")
        print(f"Number of predicted OOD samples: {len(predicted_ood_data)}, number of true OOD samples: {ood_binary_labels.sum()}")
        print(f"OOD FPR: {results_ood['ood_fpr']}, OOD TPR: {results_ood['ood_tpr']}, OOD ACC: {results_ood['ood_acc']}")
        for key, value in results_ood.items():
            if key == "ood_cm":
                continue
            results_dict[emerging_source][key] = value

        cluster_points, cluster_labels, kmean_preds, cluster_num, cluster_mask, clustering_metrics = (
            identify_new_sources(
                predicted_ood_data,
                emerging_buffer_y[ood_decisions == 1],
                emerging_source_name=emerging_source,
                emerging_source_label=emerging_label,
                num_trials=configs["num_trials"],
                v_threshold=configs["v_threshold"],
                size_threshold=configs["size_threshold"],
                size_adaptive_coeff=configs["size_adaptive_coeff"],
                min_samples=configs["training_kwargs"]["cluster_min_samples"],
            )
        )
        if cluster_points is None:
            print("No new sources identified")
            continue
        

        for key, value in clustering_metrics.items():
            results_dict[emerging_source][f"cluster_{key}"] = value
        # ID evaluation
        # Evaluate the cluster
        # Visualize the cluster
        plot_clustering_results(
            predicted_ood_data,
            predicted_ood_true_labels,
            kmean_preds,
            cluster_num,
            clustering_metrics,
            emerging_source,
            emerging_label,
            log_dir,
        )
        identified_new_src = emerging_buffer_X[ood_decisions == 1][cluster_mask]
        if len(identified_new_src) == 0:
            print("No new sources identified")
            continue
        identified_new_src_labels = emerging_buffer_y[ood_decisions == 1][cluster_mask]
        old_os_model = deepcopy(open_set_model)
        old_ae = deepcopy(current_ae)
        # System update
        # Update the GMMs
        # Update the AE
        learned_X.extend(identified_new_src)
        learned_y.extend(np.ones_like(cluster_labels) * open_set_model.n_known_sources)
        
        X_ae = np.concatenate(
            (
                datasets["init_known"]["train"][0],
                # datasets["emerging"]["learning"][0][datasets["emerging"]["learning"][1] < emerging_label],
                np.array(learned_X),
                # identified_new_src,
            )
        )
        y_ae = np.concatenate(
            (
                datasets["init_known"]["train"][1],
                # datasets["emerging"]["learning"][1][datasets["emerging"]["learning"][1] < emerging_label],
                np.array(learned_y),
                # np.ones_like(cluster_labels) * open_set_model.n_known_sources,
            )
        )
        X_ae_test = np.concatenate(
            (
                datasets["init_known"]["test"][0],
                datasets["emerging"]["test"][0][datasets["emerging"]["test"][1] <= emerging_label],
            )
        )
        y_ae_test = np.concatenate(
            (
                datasets["init_known"]["test"][1],
                datasets["emerging"]["test"][1][datasets["emerging"]["test"][1] <= emerging_label],
            )
        )
        # from ipdb import set_trace; set_trace()
        class_counts = Counter(y_ae)
        min_class_count = min(class_counts.values())
        new_X_ae = []
        new_y_ae = []

        for label in np.unique(y_ae):
            mask = y_ae == label
            new_X_ae.extend(X_ae[mask][:min_class_count])
            new_y_ae.extend(y_ae[mask][:min_class_count])
        X_ae = np.array(new_X_ae)
        y_ae = np.array(new_y_ae)
        current_ae = train_autoencoder(
            X_ae,
            y_ae,
            source_name=emerging_source,
            log_dir=log_dir,
            training_kwargs=configs["training_kwargs"],
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
        open_set_model = OpenSetModel(
            n_components=configs["training_kwargs"]["n_components"],
            covariance_type=configs["training_kwargs"]["cov_type"],
            is_bayesian=configs["training_kwargs"]["is_bayesian"],
            min_ood_tpr=configs["min_ood_tpr"],
        )
        open_set_model.fit(
            current_ae.embed(X_ae),
            y_ae,
        )

        # Update the threshold
        threshold_dataset_X = np.concatenate(
            (
                current_ae.embed(datasets["init_known"]["train"][0]),
                current_ae.embed(X_ae),
                # current_ae.embed(datasets["ood"][0]),
            )   
        )
        threshold_dataset_y = np.concatenate(
            (
                datasets["init_known"]["train"][1],
                y_ae,
                # datasets["ood"][1],
            )
        )
        open_set_model.find_best_thresholds(
            threshold_dataset_X,
            threshold_dataset_y,
        )
        # open_set_model.set_threshold(best_threshold)
        
        # Evaluation and visualization
        cm_before, cm_labels_before = old_os_model.get_cm_labels(
            old_ae.embed(identified_new_src), identified_new_src_labels, add_new_source=True
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

        y_emerging_test = datasets["emerging"]["test"][1]
        eval_datasets_X = np.concatenate(
            (
                datasets["init_known"]["test"][0],
                datasets["emerging"]["test"][0][y_emerging_test <= emerging_label],
            )
        )
        eval_datasets_y = np.concatenate(
            (
                datasets["init_known"]["test"][1],
                y_emerging_test[y_emerging_test <= emerging_label],
            )
        )

        eval_results = open_set_model.evaluate(
            current_ae.embed(eval_datasets_X),
            eval_datasets_y,
        )
        for key, value in eval_results.items():
            print(f"{key}: {value}")
            if key == "confusion_matrix":
                continue
            results_dict[emerging_source][key] = value
        save_results(results_dict, log_dir)

        # Evaluate the system and save the results
        # Visulizations
        plot_labels = np.arange(-1, np.max(eval_datasets_y) + 1)
        plot_confusion_matrix(
            eval_results["confusion_matrix"],
            log_dir,
            plot_labels,
            filename=f"{emerging_label}_{emerging_source}_cm",
        )

        # Save the state (log, GMMs, AE, threshold)
        logger.save_state(
            open_set_model,
            current_ae,
            results_dict,
            emerging_source,
            emerging_label,
            log_dir,
        )

        # input(f"Training for {emerging_source} is done. Press Enter to continue...")
