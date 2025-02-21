import torch
import torch.nn as nn
import torch.nn.functional as F

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