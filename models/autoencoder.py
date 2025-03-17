import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import time

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.utils import shuffle
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from .loss import TripletLoss


class DummyAutoencoder(nn.Module):
    def __init__(self):
        super(DummyAutoencoder, self).__init__()

    def forward(self, x):
        return x
    
    def embed(self, x: np.ndarray) -> np.ndarray:
        return x


class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            # torch.nn.Linear(hidden_dim, hidden_dim),
            # torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            # torch.nn.Linear(hidden_dim, hidden_dim),
            # torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def embed(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.encoder(torch.tensor(x)).cpu().numpy()
    

def train_autoencoder(
    X_train,
    y_train,
    X_val,
    y_val,
    source_name,
    # val_size=0.2,
    log_dir="runs",
    pretrained_model=None,
    max_known_label=None,
    initial_n_known=None,
    training_kwargs=None,
    do_early_stopping=True,
    use_scheduler=True,
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
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=42)

    num_epochs = training_kwargs.get("num_epochs", num_epochs)
    alpha = training_kwargs.get("alpha", alpha)
    beta = training_kwargs.get("beta", beta)
    batch_size = training_kwargs.get("batch_size", batch_size)
    lr = float(training_kwargs.get("lr", lr))
    pl_patience = training_kwargs.get("pl_patience", 1)
    es_patience = training_kwargs.get("es_patience", 10)
    top_k = training_kwargs.get("top_k", top_k)
    weight_decay = float(training_kwargs.get("weight_decay", 1e-1))
    device = torch.device(training_kwargs.get("device", "cuda"))
    n_components = training_kwargs.get("n_components", 3)
    cov_type = training_kwargs.get("cov_type", "full")

    # Move model and data to device
    input_dim, hidden_dim, latent_dim = 640, 1024, 640
    if pretrained_model is None:
        model = Autoencoder(input_dim, hidden_dim, latent_dim)
    else:
        model = pretrained_model
        lr = lr * training_kwargs.get("ft_lr_factor", 1.0)

    print(f"Model numel: {sum(p.numel() for p in model.parameters())}")

    model = model.to(device)

    # shuffle the data
    # X_train, y_train = shuffle(X_train, y_train)
    # X_val, y_val = shuffle(X_val, y_val)

    X_train, X_val = torch.tensor(X_train).float(), torch.tensor(X_val).float()
    y_train, y_val = torch.tensor(y_train).long(), torch.tensor(y_val).long()

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    print(f"Autoencoder training with {len(X_train)} samples and {len(X_val)} validation samples")

    # Loss function and optimizer
    reconstruction_loss = nn.MSELoss()
    discriminative_loss_fn = TripletLoss(margin=1.0, num_hard=top_k)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=pl_patience, verbose=True
    )

    # TensorBoard writer
    writer = SummaryWriter(log_dir=f"{log_dir}/runs/{source_name}")

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
                
                f_X = model.encoder(batch_X)
                outputs = model.decoder(f_X)

                loss_reconstruction = reconstruction_loss(outputs, batch_X) * alpha
                # loss_discriminative = discriminative_loss_fn(outputs, labels, top_k=top_k) * beta
                loss_discriminative = (
                    discriminative_loss_fn(
                        f_X,
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

            X_train_transformed = model.encoder(X_train).cpu().numpy()
            X_val_transformed = model.encoder(X_val).cpu().numpy()

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

        # shuffle the data
        X_train, y_train = shuffle(X_train, y_train)

        for i in range(num_train_batches):
            start, end = i * batch_size, (i + 1) * batch_size
            batch_X, labels = X_train[start:end], y_train[start:end]

            optimizer.zero_grad()
            f_X = model.encoder(batch_X)
            outputs = model.decoder(f_X)

            # Forward pass
            loss_reconstruction = reconstruction_loss(outputs, batch_X) * alpha
            # loss_discriminative = discriminative_loss_fn(outputs, labels, top_k=top_k) * beta
            loss_discriminative = (
                discriminative_loss_fn(
                    f_X,
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
        if use_scheduler:
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
            end="\n",
        )
        t1 = time.perf_counter()
        # print(f", Epoch took {t1 - t0:.2f} seconds")

        # Early stopping logic
        val_loss_eps = 0.01
        val_acc_eps = 0.01
        # if total_val_loss < best_val_loss and abs(total_val_loss - best_val_loss) > val_loss_eps:
        if do_early_stopping:
            if (
                total_val_loss < best_val_loss
                and abs(total_val_loss - best_val_loss) > val_loss_eps
            ):
                # if accuracy > best_accuracy and abs(accuracy - best_accuracy) > val_acc_eps:
                # best_val_loss = disc_val_loss
                best_val_loss = total_val_loss
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