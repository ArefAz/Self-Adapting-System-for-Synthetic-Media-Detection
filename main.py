import os
import argparse
import shutil
import torch

from utils.data_utils import get_datasets
from utils import set_seeds
from log_tools import logger
from configs import load_configs
from models import OpenSetModel, train_autoencoder

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

    # Get datasets
    datasets = get_datasets(configs["data"])

    # Initialize GMMs and the AE, and the threshold
    init_ae = train_autoencoder(
        datasets["init_known"]["train"][0],
        datasets["init_known"]["train"][1],
        source_name="init_known",
        log_dir=log_dir,
        training_kwargs=configs["training_kwargs"],
    )
    init_known_dataset = datasets["init_known"]["train"]

    with torch.no_grad():
        init_known_dataset[0] = init_ae(torch.tensor(init_known_dataset[0])).cpu().numpy()

    open_set_model = OpenSetModel(
        n_components=configs["training_kwargs"]["n_components"],
        covariance_type=configs["training_kwargs"]["cov_type"],
        is_bayesian=True,
        min_ood_tpr=configs["min_ood_tpr"],
    )
    open_set_model.fit(
        init_known_dataset[0],
        init_known_dataset[1],
    )
    exit()

    # initial evaluation
    # Add initial visualizations

    # Main loop

    # Emergence

    # Identify new data sources
    # Propose a cluster

    # ID evaluation
    # Evaluate the cluster
    # Visualize the cluster

    # System update
    # Update the GMMs
    # Update the AE
    # Update the threshold

    # Evaluation
    # Evaluate the system
    # Visulizations

    # Save the state (log, GMMs, AE, threshold)
