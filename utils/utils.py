import torch
import torch.nn as nn
import torch.optim as optim
import glob
import numpy as np
import os
import torch.nn.functional as F
import time
import random

from .eval_att import *
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class DummyModel:

    def __init__(self):
        self.model = None

    def score_samples(self, X):
        return -np.inf * np.ones(X.shape[0])


class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, latent_dim),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


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


def get_metrics(
    scores, y_test, max_known_label, initial_n_known, is_rotation=False, do_print=True
):

    # if is_rotation:
    #     y_test[y_test >= max_known_label] = initial_n_known

    if not is_rotation:
        auc_crr, fpr, tpr, thresholds = compute_auc_crr(
            scores, y_test, max_known_label, return_lists=True
        )
    else:
        auc_crr, fpr, tpr, thresholds = compute_auc_crr(
            scores,
            y_test,
            max_known_label,
            return_lists=True,
            is_rotation=True,
            initial_n_known=initial_n_known,
        )
    voted_preds = np.argmax(scores, axis=1)

    y_inital_known = y_test[y_test < initial_n_known]
    voted_preds_initial_known = voted_preds[y_test < initial_n_known]
    accuracy_initial = balanced_accuracy_score(
        y_inital_known, voted_preds_initial_known
    )

    if not is_rotation:
        y_known = y_test[y_test <= max_known_label]
        voted_preds_known = voted_preds[y_test <= max_known_label]
    else:
        y_known = y_test[(y_test < initial_n_known) | (y_test == max_known_label)]
        voted_preds_known = voted_preds[
            (y_test < initial_n_known) | (y_test == max_known_label)
        ]
        y_known[y_known == max_known_label] = initial_n_known

    accuracy = balanced_accuracy_score(y_known, voted_preds_known)
    cm = confusion_matrix(y_known, voted_preds_known)
    if do_print:
        print(
            np.unique(y_known, return_counts=True)[1],
            np.unique(voted_preds_known, return_counts=True)[1],
            sep="\n",
        )
        print(cm)

    y_current = y_test[y_test == max_known_label]
    if is_rotation:
        y_current = np.ones_like(y_current) * initial_n_known
    voted_preds_current = voted_preds[y_test == max_known_label]
    if len(y_current) > 0:
        accuracy_current = sum(y_current == voted_preds_current) / len(y_current)
    else:
        accuracy_current = 0

    y_is_synthetic = (y_test > 0).astype(int)
    preds_synthetic = (voted_preds > 0).astype(int)
    y_is_synthetic_initial = y_is_synthetic[y_test < initial_n_known]
    preds_synthetic_initial = preds_synthetic[y_test < initial_n_known]
    accuracy_synthetic_initial = balanced_accuracy_score(
        y_is_synthetic_initial, preds_synthetic_initial
    )

    if not is_rotation:
        y_is_synthetic_learned = y_is_synthetic[y_test <= max_known_label]
        preds_synthetic_learned = preds_synthetic[y_test <= max_known_label]
    else:
        y_is_synthetic_learned = y_is_synthetic[
            (y_test < initial_n_known) | (y_test == max_known_label)
        ]
        preds_synthetic_learned = preds_synthetic[
            (y_test < initial_n_known) | (y_test == max_known_label)
        ]
    acc_syn_learned = balanced_accuracy_score(
        y_is_synthetic_learned, preds_synthetic_learned
    )

    y_is_synthetic_current = y_is_synthetic[y_test == max_known_label]
    preds_synthetic_current = preds_synthetic[y_test == max_known_label]
    accuracy_synthetic_current = balanced_accuracy_score(
        y_is_synthetic_current, preds_synthetic_current
    )

    try:
        oscr = compute_oscr(
            scores[y_test <= max_known_label],
            scores[y_test > max_known_label],
            y_test[y_test <= max_known_label],
        )
    except ZeroDivisionError:
        oscr = 0

    # round all metrics to only have 4 decimal places
    auc_crr = round(auc_crr, 4)
    accuracy = round(accuracy, 4)
    accuracy_initial = round(accuracy_initial, 4)
    accuracy_current = round(accuracy_current, 4)
    acc_syn_learned = round(acc_syn_learned, 4)
    accuracy_synthetic_initial = round(accuracy_synthetic_initial, 4)
    accuracy_synthetic_current = round(accuracy_synthetic_current, 4)
    oscr = round(oscr, 4)

    results = {
        "acc_init": accuracy_initial,
        "acc": accuracy,
        "acc_curr": accuracy_current,
        "acc_syn_init": accuracy_synthetic_initial,
        "acc_syn": acc_syn_learned,
        "acc_syn_curr": accuracy_synthetic_current,
        "auc_crr": auc_crr,
        "oscr": oscr,
    }
    lists = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "cm": cm}
    return results, lists



def get_datasets(source_list, prefix):
    seen_paths = []
    for i, source in enumerate(source_list):
        glob_path = f"{prefix}/*{source}*"
        seen_paths.append((glob_path, i))

    X, y = get_embeddings(seen_paths)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def compute_class_weights(sample_counts, method="inverse"):
    """
    Compute class weights for balancing a loss function.

    Args:
        sample_counts (list or torch.Tensor): A list of sample counts per class.
        method (str): Weighting method. Options:
                      - "inverse": Uses 1 / count.
                      - "inverse_sqrt": Uses 1 / sqrt(count).
                      - "effective_num": Uses (1 - β) / (1 - β^n) (Focal loss method).

    Returns:
        torch.Tensor: Class weights.
    """
    # # Example usage
    # sample_counts = [100, 500, 1000]  # Example: 3 classes with different sample sizes
    # class_weights = compute_class_weights(sample_counts, method="inverse")
    sample_counts = torch.tensor(sample_counts, dtype=torch.float32)

    if method == "inverse":
        weights = 1.0 / sample_counts
    elif method == "inverse_sqrt":
        weights = 1.0 / torch.sqrt(sample_counts)
    elif method == "effective_num":
        beta = 0.999  # Common choice for effective number weighting
        weights = (1 - beta) / (1 - beta ** sample_counts)
    else:
        raise ValueError("Invalid method. Choose from 'inverse', 'inverse_sqrt', or 'effective_num'.")

    return weights / weights.sum()  # Normalize so weights sum to 1

# class OnlineTripletLoss(nn.Module):
#     def __init__(self, margin=1.0, num_hard=1):
#         """
#         Args:
#             margin: Triplet loss margin.
#             num_hard: Number of hardest positives and negatives to consider.
#         """
#         super(OnlineTripletLoss, self).__init__()
#         self.margin = margin
#         self.num_hard = num_hard

#     def forward(self, embeddings, labels):
#         distances = torch.cdist(embeddings, embeddings, p=2)
#         batch_size = embeddings.shape[0]
#         loss = 0.0
#         valid_triplets = 0
#         r = torch.arange(batch_size).to(embeddings.device)
#         for i in range(batch_size):
#             anchor_label = labels[i]
#             anchor_dist = distances[i]

#             # Find positive and negative samples
#             positive_mask = (labels == anchor_label) & (r != i)
#             negative_mask = labels != anchor_label

#             positive_distances = anchor_dist[positive_mask]
#             negative_distances = anchor_dist[negative_mask]

#             if len(positive_distances) < self.num_hard or len(negative_distances) < self.num_hard:
#                 continue  # Skip if not enough hard positives or negatives

#             # Select hardest positives (farthest)
#             hard_positives = torch.topk(positive_distances, self.num_hard, largest=True).values
#             # Select hardest negatives (closest)
#             hard_negatives = torch.topk(negative_distances, self.num_hard, largest=False).values

#             # Compute triplet loss
#             triplet_loss = F.relu(hard_positives.mean() - hard_negatives.mean() + self.margin)
#             loss += triplet_loss
#             valid_triplets += 1

#         return loss / valid_triplets if valid_triplets > 0 else torch.tensor(0.0, device=embeddings.device)


class OnlineTripletLoss(nn.Module):
    def __init__(self, margin=1.0, num_hard=1, class_weights=None):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.num_hard = num_hard
        self.class_weights = class_weights

    def forward(self, embeddings, labels):
        distances = torch.cdist(embeddings, embeddings, p=2)  # Compute all pairwise distances
        batch_size = embeddings.shape[0]

        # Masks for positive and negative pairs
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.T) & ~torch.eye(batch_size, device=labels.device, dtype=torch.bool)
        negative_mask = labels != labels.T

        # Extract distances for all positive and negative pairs
        positive_distances = distances[positive_mask].view(-1)  # Flatten for indexing
        negative_distances = distances[negative_mask].view(-1)  # Flatten for indexing

        # Ensure we have enough positives and negatives
        k_positive = min(self.num_hard, positive_distances.shape[0])
        k_negative = min(self.num_hard, negative_distances.shape[0])

        # Check for available triplet pairs
        if k_positive == 0 or k_negative == 0:
            return torch.tensor(0.0, device=embeddings.device)  # Return zero loss if no valid triplets

        # Select the hardest positives and negatives
        hard_positives = torch.topk(positive_distances, k_positive, largest=True).values.mean()
        hard_negatives = torch.topk(negative_distances, k_negative, largest=False).values.mean()
        # print(f"{positive_distances.mean().item():.4f}", f"{negative_distances.mean().item():.4f}", f"{hard_positives.item():.4f}", f"{hard_negatives.item():.4f}")

        # Compute loss and return mean over batch
        triplet_loss = F.relu(hard_positives - hard_negatives + self.margin)
        return triplet_loss


def train_autoencoder(
    X_train,
    y_train,
    X_val,
    y_val,
    source_name,
    pretrained_model=None,
    max_known_label=None,
    initial_n_known=None,
    kwargs=None,
):
    """
    Trains an autoencoder using in-memory datasets with weighted loss terms.

    Args:
        model: The autoencoder model.
        X_train: Training images (tensor).
        y_train: Training labels (tensor).
        X_val: Validation images (tensor).
        y_val: Validation labels (tensor).
        discriminative_loss_fn: Predefined discriminative loss function.
        num_epochs: Number of training epochs.
        lr: Initial learning rate.
        weight_decay: Weight decay for optimizer.
        device: Device to run training on.
        alpha: Weight for L1 reconstruction loss.
        beta: Weight for discriminative loss.
    """
    num_epochs = 20
    alpha = 1.0
    beta = 100.0
    batch_size = 64
    lr = 1e-4
    top_k = 1

    num_epochs = kwargs.get("num_epochs", num_epochs)
    alpha = kwargs.get("alpha", alpha)
    beta = kwargs.get("beta", beta)
    batch_size = kwargs.get("batch_size", batch_size)
    lr = kwargs.get("lr", lr)
    pl_patience = kwargs.get("pl_patience", 1)
    es_patience = kwargs.get("es_patience", 10)
    top_k = kwargs.get("top_k", top_k)
    weight_decay = kwargs.get("weight_decay", 1e-1)
    device = torch.device(kwargs.get("device", "cuda"))
    n_components = kwargs.get("n_components", 3)
    cov_type = kwargs.get("cov_type", "full")

    # Move model and data to device
    input_dim, hidden_dim, latent_dim = 640, 640, 640
    if pretrained_model is None:
        model = Autoencoder(input_dim, hidden_dim, latent_dim)
    else:
        model = pretrained_model
        lr = lr * kwargs.get("ft_lr_factor", 1.0)

    print(f"Model numel: {sum(p.numel() for p in model.parameters())}")

    model = model.to(device)

    # shuffle the data
    # X_train, y_train = shuffle(X_train, y_train)
    # X_val, y_val = shuffle(X_val, y_val)

    X_train, X_val = torch.tensor(X_train).float(), torch.tensor(X_val).float()
    y_train, y_val = torch.tensor(y_train).long(), torch.tensor(y_val).long()

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    # Loss function and optimizer
    reconstruction_loss = nn.L1Loss()
    discriminative_loss_fn = OnlineTripletLoss(margin=1.0, num_hard=top_k)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=pl_patience, verbose=True
    )

    # TensorBoard writer
    writer = SummaryWriter(log_dir=f"runs/{source_name}")

    # Early stopping parameters
    best_val_loss = float("inf")
    early_stop_counter = 0

    num_train_batches = (len(X_train) + batch_size - 1) // batch_size
    num_val_batches = (len(X_val) + batch_size - 1) // batch_size
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        t0 = time.perf_counter()
        # Validation phase
        model.eval()
        recon_val_loss = 0.0
        disc_val_loss = 0.0
        total_val_loss = 0.0

        with torch.no_grad():
            for i in range(num_val_batches):
                start, end = i * batch_size, (i + 1) * batch_size
                batch_X, labels = X_val[start:end], y_val[start:end]

                outputs = model(batch_X)

                loss_reconstruction = reconstruction_loss(outputs, batch_X) * alpha
                # loss_discriminative = discriminative_loss_fn(outputs, labels, top_k=top_k) * beta
                loss_discriminative = (
                    discriminative_loss_fn(
                        outputs,
                        labels,
                    )
                    * beta
                )

                loss = loss_reconstruction + loss_discriminative

                recon_val_loss += loss_reconstruction.item()
                if loss_discriminative > 0 and isinstance(
                    loss_discriminative, torch.Tensor
                ):
                    disc_val_loss += loss_discriminative.item()
                total_val_loss += loss.item()

            X_train_transformed = model(X_train).detach().cpu().numpy()
            X_val_transformed = model(X_val).detach().cpu().numpy()

        gmms = {}
        y_train_np = y_train.cpu().numpy()
        y_val_np = y_val.cpu().numpy()
        for label in np.unique(y_train_np):
            X_train_transformed_label = X_train_transformed[y_train_np == label]
            gmm_model = BayesianGaussianMixture(
                # n_components=n_components,
                n_components=1,
                # covariance_type=cov_type,
                covariance_type='spherical',
            )
            gmm_model.fit(X_train_transformed_label)
            gmms[label] = gmm_model

        scores = np.zeros((X_val_transformed.shape[0], len(gmms)))
        for i, key in enumerate(gmms):
            gmm_model = gmms[key]
            scores[:, i] = gmm_model.score_samples(X_val_transformed)

        preds = np.argmax(scores, axis=1)
        # cm = confusion_matrix(y_val_np, preds)
        # print(cm)

        results = {}
        accuracy = np.mean(preds == y_val_np)
        y_is_synthetic = (y_val_np > 0).astype(int)
        preds_synthetic = (preds > 0).astype(int)
        acc_synthetic = np.mean(y_is_synthetic == preds_synthetic)
        results["acc"] = accuracy
        results["acc_syn"] = acc_synthetic
        for key, value in results.items():
            writer.add_scalar(f"Metrics/{key}", value, epoch)

        # Compute average validation losses
        recon_val_loss /= num_val_batches
        disc_val_loss /= num_val_batches
        total_val_loss /= num_val_batches

        # Training phase
        model.train()
        train_recon_loss = 0.0
        train_disc_loss = 0.0
        total_train_loss = 0.0

        for i in range(num_train_batches):
            start, end = i * batch_size, (i + 1) * batch_size
            batch_X, labels = X_train[start:end], y_train[start:end]

            optimizer.zero_grad()
            outputs = model(batch_X)

            # Forward pass
            loss_reconstruction = reconstruction_loss(outputs, batch_X) * alpha
            # loss_discriminative = discriminative_loss_fn(outputs, labels, top_k=top_k) * beta
            loss_discriminative = (
                discriminative_loss_fn(
                    outputs,
                    labels,
                )
                * beta
            )

            loss = loss_reconstruction + loss_discriminative

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Track losses
            train_recon_loss += loss_reconstruction.item()
            if loss_discriminative > 0 and isinstance(
                loss_discriminative, torch.Tensor
            ):
                train_disc_loss += loss_discriminative.item()
            total_train_loss += loss.item()

        # Compute average training losses
        train_recon_loss /= num_train_batches
        train_disc_loss /= num_train_batches
        total_train_loss /= num_train_batches

        # Adjust learning rate if validation loss plateaus
        scheduler.step(total_val_loss)
        # Log losses to TensorBoard
        writer.add_scalar("Loss/Train_Recon", train_recon_loss, epoch)
        writer.add_scalar("Loss/Train_Discriminative", train_disc_loss, epoch)
        writer.add_scalar("Loss/Train_Total", total_train_loss, epoch)
        writer.add_scalar("Loss/Val_Recon", recon_val_loss, epoch)
        writer.add_scalar("Loss/Val_Discriminative", disc_val_loss, epoch)
        writer.add_scalar("Loss/Val_Total", total_val_loss, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)
        # Print progress
        print(
            f"Epoch [{epoch+1}/{num_epochs}] - Train Loss1: {train_recon_loss:.4f}, Train Loss2: {train_disc_loss:.4f}, Train TLoss: {total_train_loss:.4f}",
            end="",
        )
        print(
            f", Val Loss1: {recon_val_loss:.4f}, Val Loss2: {disc_val_loss:.4f}, Val TLoss: {total_val_loss:.4f}, Accuracy: {accuracy:.4f}, Detection Acc: {acc_synthetic:.4f}",
            end="",
        )
        t1 = time.perf_counter()
        print(f", Epoch took {t1 - t0:.2f} seconds")

        # Early stopping logic
        val_loss_eps = 0.005
        val_acc_eps = 0.01
        # if total_val_loss < best_val_loss and abs(total_val_loss - best_val_loss) > val_loss_eps:
        if (
            disc_val_loss < best_val_loss
            and abs(disc_val_loss - best_val_loss) > val_loss_eps
        ):
            # if accuracy > best_accuracy and abs(accuracy - best_accuracy) > val_acc_eps:
            best_val_loss = disc_val_loss
            # best_val_loss = total_val_loss
            best_accuracy = accuracy
            best_detected_accuracy = acc_synthetic
            best_epoch = epoch
            early_stop_counter = 0  # Reset counter
            best_model = deepcopy(model)
        else:
            early_stop_counter += 1

        if early_stop_counter >= es_patience:
            print(
                f"Early stopping triggered after {epoch+1} epochs at epoch {best_epoch+1} ",
                end="",
            )
            print(
                f"with validation loss: {best_val_loss:.4f} and accuracy: {best_accuracy:.4f}, detection accuracy: {best_detected_accuracy:.4f}"
            )
            return best_model.eval().cpu()

    # Close TensorBoard writer
    writer.close()

    return model.eval().cpu()



