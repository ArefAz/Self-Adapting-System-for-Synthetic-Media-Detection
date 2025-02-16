import torch
import glob
import numpy as np
import os
import torch.nn.functional as F

from .eval_att import *
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.utils.tensorboard import SummaryWriter


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
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(latent_dim, hidden_dim),
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


# adapted for transformed FSDs
# def get_embeddings(dirs):
#     if isinstance(dirs, str):
#         dirs = [dirs]
#     model_paths = []
#     labels = []
#     for dir, label in dirs:
#         model_path_list = [
#             p for p in glob.glob(f"{dir}/*.pth") if os.path.getsize(p) > 0
#         ]
#         model_paths += model_path_list
#         labels += [label] * len(model_path_list)
#     embeddings = []
#     for path in model_paths:
#         embedding = torch.load(path, map_location="cpu", weights_only=True)
#         embeddings.append(embedding)
#     embeddings = torch.stack(embeddings).squeeze(1)
#     embeddings = embeddings.view(embeddings.size(0), -1)
#     return embeddings, np.array(labels)


def get_metrics(scores, y_test, max_known_label, initial_n_known, is_rotation=False):

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
    print(
        np.unique(y_known, return_counts=True)[1],
        np.unique(voted_preds_known, return_counts=True)[1],
        sep="\n",
    )
    print(cm)
    # input()

    y_current = y_test[y_test == max_known_label]
    if is_rotation:
        y_current = np.ones_like(y_current) * initial_n_known
    voted_preds_current = voted_preds[y_test == max_known_label]
    accuracy_current = sum(y_current == voted_preds_current) / len(y_current)

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

    oscr = compute_oscr(
        scores[y_test <= max_known_label],
        scores[y_test > max_known_label],
        y_test[y_test <= max_known_label],
    )

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
    lists = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
    return results, lists


def get_datasets(source_list, prefix):
    seen_paths = []
    for i, source in enumerate(source_list):
        glob_path = f"{prefix}/*{source}*"
        seen_paths.append((glob_path, i))

    X, y = get_embeddings(seen_paths)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def online_triplet_mining(embeddings, labels, class_weights=None, margin=1.0):
    batch_size = embeddings.size(0)
    triplet_loss = 0.0

    for i in range(batch_size):
        anchor = embeddings[i]
        anchor_label = labels[i]

        # Find positives (same class) excluding the anchor itself
        positive_mask = (labels == anchor_label) & (
            torch.arange(batch_size).cuda() != i
        )
        positives = embeddings[positive_mask]

        # Find negatives (different class)
        negative_mask = labels != anchor_label
        negatives = embeddings[negative_mask]

        if positives.size(0) > 0 and negatives.size(0) > 0:
            # Pick the hardest positive and hardest negative
            positive_distances = torch.norm(positives - anchor, dim=1)
            hardest_positive = positives[torch.argmin(positive_distances)]

            negative_distances = torch.norm(negatives - anchor, dim=1)
            hardest_negative = negatives[torch.argmin(negative_distances)]

            # Compute the triplet loss for this anchor
            loss = F.relu(
                torch.norm(anchor - hardest_positive, p=2) ** 2
                - torch.norm(anchor - hardest_negative, p=2) ** 2
                + margin
            )

            # Scale loss by the class weight of the anchor
            loss = loss #* class_weights[anchor_label]

            triplet_loss += loss

    return triplet_loss / batch_size


def compute_class_weights(y_train):
    unique_classes, counts = np.unique(y_train.cpu().numpy(), return_counts=True)
    total_samples = len(y_train)
    class_weights = {
        cl: total_samples / (len(unique_classes) * count)
        for cl, count in zip(unique_classes, counts)
    }
    # sum_class_weights = sum(class_weights.values())
    # class_weights = {cl: weight / sum_class_weights for cl, weight in class_weights.items()}
    return class_weights


# def train_autoencoder(X_train, y_train, X_val, y_val):

#     print("==== Training Autoencoder ====")

#     # shuffle the data
#     X_train, y_train = shuffle(X_train, y_train)
#     X_val, y_val = shuffle(X_val, y_val)

#     X_train, X_val = torch.tensor(X_train).float(), torch.tensor(X_val).float()
#     y_train, y_val = torch.tensor(y_train).long(), torch.tensor(y_val).long()

#     # move to cuda
#     X_train, y_train = X_train.cuda(), y_train.cuda()
#     X_val, y_val = X_val.cuda(), y_val.cuda()

#     # print unique values with their counts in y_train and y_val
#     print(np.unique(y_train.cpu().numpy(), return_counts=True))
#     print(np.unique(y_val.cpu().numpy(), return_counts=True))

#     num_epochs = 20
#     alpha = 1.0
#     beta = 100.0
#     batch_size = 64
#     lr = 1e-4

#     input_dim, hidden_dim, latent_dim = 640, 640, 640
#     # Compute class weights as a dictionary
#     class_weights_dict = compute_class_weights(y_train)
#     class_weights = (
#         torch.tensor([class_weights_dict[i] for i in range(len(class_weights_dict))])
#         .float()
#         .cuda()
#     )

#     model = Autoencoder(input_dim, hidden_dim, latent_dim)
#     model = model.cuda()
#     reconstruction_loss = torch.nn.L1Loss()

#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)

#     train_loss = []
#     val_loss = []
#     train_loss1 = []
#     train_loss2 = []
#     val_loss1 = []
#     val_loss2 = []
#     train_loss_epoch = 0

#     # Initialize TensorBoard writer
#     writer = SummaryWriter()

#     for epoch in range(num_epochs):
#         model.train()
#         optimizer.zero_grad()
#         train_loss_epoch = 0
#         for i in range(0, X_train.size(0), batch_size):
#             batch_X = X_train[i : i + batch_size]
#             outputs = model(batch_X)
#             loss1 = reconstruction_loss(outputs, batch_X) * alpha
#             loss2 = (
#                 online_triplet_mining(
#                     model(batch_X), y_train[i : i + batch_size], class_weights
#                 )
#                 * beta
#             )
#             loss = loss1 + loss2
#             loss.backward()
#             optimizer.step()
#             train_loss.append(loss.item())
#             train_loss1.append(loss1.item())
#             train_loss2.append(loss2.item())
#             train_loss_epoch += loss.item()

#         model.eval()
#         with torch.no_grad():
#             val_loss_epoch = 0
#             for i in range(0, X_val.size(0), batch_size):
#                 batch_X = X_val[i : i + batch_size]
#                 outputs = model(batch_X)
#                 loss1 = reconstruction_loss(outputs, batch_X) * alpha
#                 loss2 = (
#                     online_triplet_mining(
#                         model(batch_X), y_val[i : i + batch_size], class_weights
#                     )
#                     * beta
#                 )
#                 loss = loss1 + loss2
#                 val_loss_epoch += loss.item()
#                 val_loss1.append(loss1.item())
#                 if isinstance(loss2, torch.Tensor):
#                     val_loss2.append(loss2.item())
#                 else:
#                     val_loss2.append(loss2)
#             val_loss.append(val_loss_epoch / (X_val.size(0) // batch_size))

#         scheduler.step(val_loss[-1])

#         # Log metrics to TensorBoard
#         writer.add_scalar('Loss/train', train_loss_epoch / (X_train.size(0) // batch_size), epoch)
#         writer.add_scalar('Loss/val', val_loss[-1], epoch)
#         writer.add_scalar('Loss1/train', train_loss1[-1], epoch)
#         writer.add_scalar('Loss2/train', train_loss2[-1], epoch)
#         writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

#         print(
#             f"Epoch {epoch+1}/{num_epochs}, loss_epoch = {train_loss_epoch / (X_train.size(0) // batch_size):.4f},\
#             Loss1: {train_loss1[-1]:.4f}, Loss2: {train_loss2[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}, lr: {optimizer.param_groups[0]['lr']}"
#         )

#     writer.close()
#     return model.cpu().eval()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

def train_autoencoder(X_train, y_train, X_val, y_val):
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
    beta = 10.0
    batch_size = 64
    lr = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move model and data to device
    input_dim, hidden_dim, latent_dim = 640, 640, 640
    model = Autoencoder(input_dim, hidden_dim, latent_dim)
    model = model.to(device)
    X_train = torch.tensor(X_train).float().to(device)

    # shuffle the data
    # X_train, y_train = shuffle(X_train, y_train)
    # X_val, y_val = shuffle(X_val, y_val)

    X_train, X_val = torch.tensor(X_train).float(), torch.tensor(X_val).float()
    y_train, y_val = torch.tensor(y_train).long(), torch.tensor(y_val).long()

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    # Loss function and optimizer
    reconstruction_loss = nn.L1Loss()
    discriminative_loss_fn = online_triplet_mining
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=0, verbose=True)
    
    # TensorBoard writer
    writer = SummaryWriter()

    batch_size = 64  # Set batch size manually
    num_train_batches = (len(X_train) + batch_size - 1) // batch_size
    num_val_batches = (len(X_val) + batch_size - 1) // batch_size

    for epoch in range(num_epochs):
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
            loss_discriminative = online_triplet_mining(outputs, labels) * beta
            
            loss = alpha * loss_reconstruction + beta * loss_discriminative
            loss = alpha * loss_reconstruction + beta * loss_discriminative
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Track losses
            train_recon_loss += loss_reconstruction.item()
            train_disc_loss += loss_discriminative.item()
            total_train_loss += loss.item()
        
        # Compute average training losses
        train_recon_loss /= num_train_batches
        train_disc_loss /= num_train_batches
        total_train_loss /= num_train_batches

        # Validation phase
        model.eval()
        val_recon_loss = 0.0
        val_disc_loss = 0.0
        total_val_loss = 0.0
        
        with torch.no_grad():
            for i in range(num_val_batches):
                start, end = i * batch_size, (i + 1) * batch_size
                batch_X, labels = X_val[start:end], y_val[start:end]
                
                outputs = model(batch_X)
                
                # loss_reconstruction = reconstruction_loss(reconstructed, batch_X)
                # loss_discriminative = discriminative_loss_fn(embeddings, labels)

                loss_reconstruction = reconstruction_loss(outputs, batch_X) * alpha
                loss_discriminative = online_triplet_mining(outputs, labels) * beta

                loss = alpha * loss_reconstruction + beta * loss_discriminative
                
                val_recon_loss += loss_reconstruction.item()
                val_disc_loss += loss_discriminative.item()
                total_val_loss += loss.item()
        
        # Compute average validation losses
        val_recon_loss /= num_val_batches
        val_disc_loss /= num_val_batches
        total_val_loss /= num_val_batches

        # Adjust learning rate if validation loss plateaus

        scheduler.step(total_val_loss)
        # Log losses to TensorBoard
        writer.add_scalar('Loss/Train_Recon', train_recon_loss, epoch)
        writer.add_scalar('Loss/Train_Discriminative', train_disc_loss, epoch)
        writer.add_scalar('Loss/Train_Total', total_train_loss, epoch)
        writer.add_scalar('Loss/Val_Recon', val_recon_loss, epoch)
        writer.add_scalar('Loss/Val_Discriminative', val_disc_loss, epoch)
        writer.add_scalar('Loss/Val_Total', total_val_loss, epoch)

        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {total_train_loss:.4f}, Val Loss: {total_val_loss:.4f}")

    # Close TensorBoard writer
    writer.close()

    return model.eval().cpu()