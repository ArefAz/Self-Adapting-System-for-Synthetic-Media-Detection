import torch
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split


def get_embeddings(dirs):
    if isinstance(dirs, str):
        dirs = [dirs]
    model_paths = []
    labels = []
    for dir, label in dirs:
        model_path_list = [
            p for p in glob.glob(f"{dir}/*.pth") if os.path.getsize(p) > 0
        ]
        model_paths += model_path_list
        labels += [label] * len(model_path_list)
    embeddings = []
    for path in model_paths:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        embeddings.append(
            torch.cat([v for k, v in state_dict.items() if "weight" in k])
        )
    embeddings = torch.stack(embeddings).squeeze(1)
    embeddings = embeddings.view(embeddings.size(0), -1)
    return np.array(embeddings), np.array(labels)


def get_fsd_paths(source_list, dataset_path, offset=0):
    fsd_paths = []
    for i, source in enumerate(source_list):
        glob_path = f"{dataset_path}/*{source}*"
        fsd_paths.append((glob_path, i + offset))
    return fsd_paths[:1000]


def get_datasets(configs_data):
    dataset_path = configs_data["dataset_path"]
    init_source_list = configs_data["init_sources"]
    emerging_source_list = configs_data["emerging_sources"]
    ood_source_list = configs_data["ood_sources"]
    test_size = configs_data["test_size"]
    seed = configs_data["seed"]
    train_size = 400 * len(init_source_list)
    learning_size = 400 * len(init_source_list)
    test_size = 200 * len(init_source_list)
    datasets = {}

    init_fsd_paths = get_fsd_paths(init_source_list, dataset_path)
    X, y = get_embeddings(init_fsd_paths)
    (X_train, y_train), (X_learning, y_learning), (X_test, y_test) = split_dataset(
        X, y, (train_size, learning_size, test_size), seed=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=seed
    )
    datasets["init_known"] = {
        "train": [X_train, y_train],
        "val": [X_val, y_val],
        "learning": [X_learning, y_learning],
        "test": [X_test, y_test],
    }

    emerging_fsd_paths = get_fsd_paths(
        emerging_source_list, dataset_path, offset=len(init_source_list)
    )
    X, y = get_embeddings(emerging_fsd_paths)
    X_emerging, X_test, y_emerging, y_test = train_test_split(
        X, y, test_size=0.5, random_state=seed
    )
    X_test, _, y_test, _ = train_test_split(
        X_test, y_test, test_size=0.5, random_state=seed
    )
    datasets["emerging"] = {
        "learning": [X_emerging, y_emerging],
        "test": [X_test, y_test],
    }

    ood_fsd_paths = get_fsd_paths(
        ood_source_list,
        dataset_path,
        offset=len(init_source_list) + len(emerging_source_list),
    )
    X, y = get_embeddings(ood_fsd_paths)
    datasets["ood"] = [X, y]
    return datasets


def split_dataset(X, y, sizes, seed=None):
    assert len(X) == len(y), "X and y must have the same length."
    assert sum(sizes) == len(
        X
    ), f"Sum of sizes must be equal to the length of X: {len(X)}, got {sum(sizes)}."

    if seed is not None:
        np.random.seed(seed)

    # Shuffle indices
    indices = np.random.permutation(len(X))

    # Compute split indices
    split1, split2 = sizes[0], sizes[0] + sizes[1]

    # Split the dataset and labels
    X1, y1 = X[indices[:split1]], y[indices[:split1]]
    X2, y2 = X[indices[split1:split2]], y[indices[split1:split2]]
    X3, y3 = X[indices[split2:]], y[indices[split2:]]

    return (X1, y1), (X2, y2), (X3, y3)
