import os
import argparse
import shutil
import torch
import numpy as np
import pandas as pd

from copy import deepcopy
from utils.data_utils import get_datasets
from utils.visualizaitons import *
from utils import set_seeds
from log_tools import logger
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
            initial_ae.embed(datasets["init_known"]["test"][0]),
            # initial_ae.encode(datasets["emerging"]["learning"][0]),
            initial_ae.embed(datasets["ood"][0]),
        )
    )
    threshold_dataset_y = np.concatenate(
        (
            datasets["init_known"]["test"][1],
            # datasets["emerging"]["learning"][1],
            datasets["ood"][1],
        )
    )
    open_set_model.set_threshold(
        open_set_model.find_best_threshold(
            threshold_dataset_X,
            threshold_dataset_y,
            max_known_label=np.max(init_known_dataset[1]),
        )
    )

    # initial evaluation
    eval_results = open_set_model.evaluate(
        initial_ae.embed(datasets["init_known"]["test"][0]),
        datasets["init_known"]["test"][1],
    )

    # Add initial visualizations
    plot_labels = np.arange(-1, np.max(init_known_dataset[1]) + 1)
    plot_confusion_matrix(eval_results["confusion_matrix"], log_dir, labels=plot_labels)

    plot_embeddings_tsne(
        initial_ae.embed(init_known_dataset[0]),
        init_known_dataset[1],
        log_dir=log_dir,
        title="Initial Known Training Data",
    )
    plot_embeddings_tsne(
        initial_ae.embed(datasets["init_known"]["test"][0]),
        datasets["init_known"]["test"][1],
        log_dir=log_dir,
        title="Initial Known Test Data",
    )

    # Main loop
    X_emerging = datasets["emerging"]["learning"][0]
    y_emerging = datasets["emerging"]["learning"][1]
    emerging_labels = np.unique(y_emerging)

    current_ae = deepcopy(initial_ae)
    for i, emerging_label in enumerate(emerging_labels[:1]):
        emerging_source = configs["data"]["emerging_sources"][i]
        print()
        print("=====================================================================")
        print(f"Emerging Source: {emerging_source} with label {emerging_label}")
        print("=====================================================================")

        # Create an emerging source dataset
        emerging_buffer = (
            X_emerging[y_emerging == emerging_label],
            y_emerging[y_emerging == emerging_label],
        )
        emerging_buffer_X = emerging_buffer[0]
        emerging_buffer_y = emerging_buffer[1]

        emerging_buffer_X = np.concatenate(
            (
                emerging_buffer_X,
                datasets["init_known"]["learning"][0],
            )
        )

        ood_binary_labels = np.concatenate(
            (
                np.ones_like(emerging_buffer_y),
                np.zeros_like(datasets["init_known"]["learning"][1]),
            )
        )

        ood_decisions = open_set_model.ood_detect(current_ae.embed(emerging_buffer_X))
        ood_acc = np.sum(ood_decisions == ood_binary_labels) / len(ood_binary_labels)
        print(f"OOD Detection Accuracy: {ood_acc}")

        emerging_buffer_y = np.concatenate(
            (
                emerging_buffer_y,
                datasets["init_known"]["learning"][1],
            )
        )
        predicted_ood_data = current_ae.embed(emerging_buffer_X[ood_decisions == 1])
        predicted_ood_true_labels = emerging_buffer_y[ood_decisions == 1]

        cluster_points, cluster_labels, kmean_preds, cluster_num, cluster_mask = (
            identify_new_sources(
                predicted_ood_data,
                emerging_buffer_y[ood_decisions == 1],
                emerging_source_name=emerging_source,
                num_trials=configs["num_trials"],
                v_threshold=configs["v_threshold"],
                size_threshold=configs["size_threshold"],
                size_adaptive_coeff=configs["size_adaptive_coeff"],
            )
        )

        if cluster_points is None:
            print("No new sources identified")
            continue

        # ID evaluation
        # Evaluate the cluster
        # Visualize the cluster
        plot_clustering_results(
            predicted_ood_data,
            predicted_ood_true_labels,
            kmean_preds,
            cluster_num,
            emerging_source,
            log_dir,
        )
        identified_new_src = emerging_buffer_X[ood_decisions == 1][cluster_mask]
        # System update
        # Update the GMMs
        # Update the AE
        X_ae = np.concatenate(
            (
                datasets["init_known"]["learning"][0],
                identified_new_src,
            )
        )
        y_ae = np.concatenate(
            (
                datasets["init_known"]["learning"][1],
                np.ones_like(cluster_labels) * open_set_model.n_known_sources,
            )
        )
        current_ae = train_autoencoder(
            X_ae,
            y_ae,
            source_name=emerging_source,
            log_dir=log_dir,
            training_kwargs=configs["training_kwargs"],
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

        print(f"Number of known sources: {open_set_model.n_known_sources}")
        print(np.unique(y_ae, return_counts=True))
        exit()

        # Update the threshold
        threshold_dataset_X = np.concatenate(
            (
                current_ae.embed(datasets["init_known"]["learning"][0]),
                current_ae.embed(identified_new_src),
                current_ae.embed(datasets["ood"][0]),
            )
        )
        threshold_dataset_y = np.concatenate(
            (
                datasets["init_known"]["learning"][1],
                np.ones_like(cluster_labels) * open_set_model.n_known_sources,
                datasets["ood"][1],
            )
        )
        open_set_model.set_threshold(
            open_set_model.find_best_threshold(
                threshold_dataset_X,
                threshold_dataset_y,
                max_known_label=np.max(y_ae),
            )
        )

        # Evaluation
        eval_results = open_set_model.evaluate(
            current_ae.embed(datasets["init_known"]["test"][0]),
            datasets["init_known"]["test"][1],
        )

        for key, value in eval_results.items():
            print(f"{key}: {value}")


        # Evaluate the system and save the results
        # Visulizations

        # Save the state (log, GMMs, AE, threshold)
