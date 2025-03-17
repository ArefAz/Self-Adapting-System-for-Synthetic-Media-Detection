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
from utils.helper_functions import *
from utils import set_seeds
from log_tools import logger, save_results
from configs import load_configs
from models import OpenSetModel, train_autoencoder, identify_new_sources
from models.autoencoder import DummyAutoencoder

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
    if configs["use_autoencoder"]:
        initial_ae = train_autoencoder(
            datasets["init_known"]["train"][0],
            datasets["init_known"]["train"][1],
            datasets["init_known"]["val"][0],
            datasets["init_known"]["val"][1],
            source_name="init_known",
            log_dir=log_dir,
            training_kwargs=configs["training_kwargs"],
        )
    else:
        initial_ae = DummyAutoencoder()
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
            initial_ae.embed(datasets["init_known"]["val"][0]),
            # initial_ae.embed(datasets["emerging"]["test"][0]),
            initial_ae.embed(datasets["ood"][0]),
        )
    )
    threshold_dataset_y = np.concatenate(
        (
            datasets["init_known"]["val"][1],
            # datasets["emerging"]["test"][1],
            datasets["ood"][1],
        )
    )
    open_set_model.find_best_thresholds(
        threshold_dataset_X,
        threshold_dataset_y,
    )
    # open_set_model.set_threshold(init_threshold)

    validation_set = {"initial": [init_known_dataset[0], init_known_dataset[1]]}
    validation_results = open_set_model.evaluate(
        initial_ae.embed(validation_set["initial"][0]),
        validation_set["initial"][1],
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

    logger.save_state(
        open_set_model,
        initial_ae,
        results_dict,
        "initial",
        "0",
        log_dir,
    )

    save_results(results_dict, log_dir)

    # Add initial visualizations
    plot_labels = np.arange(-1, np.max(init_known_dataset[1]) + 1)
    plot_confusion_matrix(
        eval_results["confusion_matrix"],
        log_dir,
        labels=plot_labels,
        filename="0_initial_cm",
    )

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
    learned_X_val = []
    learned_y_val = []

    for i, emerging_label in enumerate(emerging_labels):
        emerging_source = configs["data"]["emerging_sources"][i]
        print()
        print("=====================================================================")
        print(f"Emerging Source: {emerging_source} with label {emerging_label}")
        print("=====================================================================")
        results_dict[emerging_source] = {}
        # input("Press Enter to continue...")

        if not configs["do_nothing"]:
            open_set_model, current_ae, learned_X, learned_y, identified_new_src, identified_new_src_labels, old_os_model, old_ae, results_dict = (
                propose_an_update(
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
                )
            )
            
            
            # Evaluation and visualization
            if identified_new_src is not None:
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
