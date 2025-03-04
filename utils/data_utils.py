import torch
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict


def get_embeddings(dirs, max_num=None):
    if isinstance(dirs, str):
        dirs = [dirs]
    model_paths = []
    labels = []
    for dir, label in dirs:
        model_path_list = [
            p for p in glob.glob(f"{dir}/*.pth") if os.path.getsize(p) > 0
        ]
        if max_num is not None:
            model_path_list = model_path_list[:max_num]
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
    return fsd_paths


def get_datasets(configs_data):
    dataset_path = configs_data["dataset_path"]
    init_source_list = configs_data["init_sources"]
    emerging_source_list = configs_data["emerging_sources"]
    ood_source_list = configs_data["ood_sources"]
    datasets = {}

    init_fsd_paths = get_fsd_paths(init_source_list, dataset_path)
    X, y = get_embeddings(init_fsd_paths)
    data_dict = convert_xy_to_dict(X, y)
    fractions = [0.375, 0.125, 0.375, 0.125]
    splits = split_dataset_by_class_fractions(data_dict, fractions, seed=42)
    combined_dict = combine_splits_to_xy(splits)
    X_train, y_train = combined_dict[0]
    X_val, y_val = combined_dict[1]
    X_learning, y_learning = combined_dict[2]
    X_test, y_test = combined_dict[3]

    datasets["init_known"] = {
        "train": [X_train, y_train],
        "val": [X_val, y_val],
        "learning": [X_learning, y_learning],
        "test": [X_test, y_test],
    }

    emerging_fsd_paths = get_fsd_paths(
        emerging_source_list, dataset_path, offset=len(init_source_list)
    )
    X, y = get_embeddings(emerging_fsd_paths, max_num=1000)
    fractions = [0.75, 0.25]
    splits = split_dataset_by_class_fractions(convert_xy_to_dict(X, y), fractions, seed=42)
    combined_dict = combine_splits_to_xy(splits)
    X_emerging, y_emerging = combined_dict[0]
    X_test, y_test = combined_dict[1]

    # print(f"Emerging: {X_emerging.shape}, {np.unique(y_emerging, return_counts=True)}")
    # print(f"Test: {X_test.shape}, {np.unique(y_test, return_counts=True)}")
    # exit()

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


def convert_xy_to_dict(X, y):
    """
    Converts (X, y) data into a dictionary where each key is a class label
    and the corresponding value is an array of samples belonging to that class.
    """
    data_dict = defaultdict(list)
    for xi, yi in zip(X, y):
        data_dict[yi].append(xi)
    # Convert lists to numpy arrays for consistency
    for label in data_dict:
        data_dict[label] = np.array(data_dict[label])
    return data_dict

def split_dataset_by_class_fractions(data_dict, fractions, seed=42):
    """
    Splits the dataset by class into subsets according to specified fractions.
    
    Parameters:
        data_dict (dict): Dictionary mapping each class label to its samples.
        fractions (list or array): Fractions for each subset (e.g., [0.3, 0.5, 0.2]).
                                   These do not have to sum exactly to 1; they will be normalized.
        seed (int): Random seed for reproducibility.
    
    Returns:
        dict: A dictionary where each key is a subset index and the value is a dictionary
              mapping class labels to the samples assigned to that subset.
    """
    # Normalize fractions in case they don't sum to 1 exactly
    fractions = np.array(fractions, dtype=float)
    fractions = fractions / fractions.sum()
    n_subsets = len(fractions)
    
    # Initialize output: one dict per subset
    subsets = {i: {} for i in range(n_subsets)}
    
    # Process each class independently
    for cls, samples in data_dict.items():
        samples = np.array(samples)  # Ensure we have a NumPy array
        # Set seed and shuffle indices for reproducibility
        np.random.seed(seed)
        indices = np.arange(len(samples))
        np.random.shuffle(indices)
        
        total = len(samples)
        # Determine base counts using floor, then compute leftover samples
        counts = np.floor(fractions * total).astype(int)
        remainder = total - counts.sum()
        
        # Distribute the remaining samples based on the fractional remainders
        fractional_parts = fractions * total - counts
        # Get indices of subsets sorted by descending fractional part
        extra_order = np.argsort(-fractional_parts)
        for i in range(remainder):
            counts[extra_order[i]] += 1
        
        # Now, split the indices according to these counts
        start = 0
        for i, count in enumerate(counts):
            selected_indices = indices[start:start + count]
            subsets[i][cls] = samples[selected_indices]
            start += count
    
    return subsets

def combine_splits_to_xy(splits):
    """
    Combine split data from different classes into (X_subset, y_subset) tuples.
    
    Parameters:
        splits (dict): Dictionary where each key is a split id and each value is another
                       dictionary mapping class labels to numpy arrays of samples.
    
    Returns:
        dict: A dictionary where each key is a split id and the corresponding value is a tuple
              (X_subset, y_subset), where:
                  - X_subset is a numpy array of data samples combined from all classes.
                  - y_subset is a numpy array of corresponding labels.
    """
    combined = {}
    
    for split_id, class_dict in splits.items():
        X_list = []
        y_list = []
        for cls, samples in class_dict.items():
            X_list.append(samples)
            # Create an array of the same length as samples filled with the label cls
            y_list.append(np.full(len(samples), cls))
            
        # Concatenate the samples from all classes for this split
        X_subset = np.concatenate(X_list, axis=0)
        y_subset = np.concatenate(y_list, axis=0)
        combined[split_id] = (X_subset, y_subset)
    
    return combined

if __name__ == "__main__":
    X = np.concatenate([np.arange(1000).reshape(-1, 1) + i*1000 for i in range(10)])
    y = np.concatenate([[i]*1000 for i in range(10)])

    # Convert (X, y) into the required dictionary format
    data_dict = convert_xy_to_dict(X, y)

    # Define arbitrary fractions for the subsets
    fractions = [0.3, 0.5, 0.2]  # These fractions will be applied per class

    # Split the dataset ensuring each subset gets roughly the specified fraction of samples per class
    splits = split_dataset_by_class_fractions(data_dict, fractions, seed=42)

    # Check the distribution:
    for subset_id, class_dict in splits.items():
        print(f"Subset {subset_id}:")
        for cls, samples in class_dict.items():
            print(f"  Class {cls}: {len(samples)} samples")

    # Combine the splits back into (X, y) format
    combined_dict = combine_splits_to_xy(splits)
    # For a given split_id:
    for split_id in combined_dict:
        X_subset, y_subset = combined_dict[split_id]
        print(f"Split {split_id}: X shape: {X_subset.shape}, y shape: {y_subset.shape}")
        print(f"  Unique labels: {np.unique(y_subset, return_counts=True)}")