import torch
from torch.utils.data import Dataset


def generate_pseudo_labels(model, dataloader, device, threshold=0.9):
    """
    Generate pseudo-labels for unlabeled data using a trained model.
    
    Args:
    model (torch.nn.Module): The trained model for generating predictions.
    dataloader (torch.utils.data.DataLoader): DataLoader for unlabeled data.
    device (torch.device): The device on which the model is deployed (e.g., 'cuda' or 'cpu').
    threshold (float): Minimum confidence for a prediction to be accepted as a pseudo-label.
    
    Returns:
    Tuple of (pseudo_data, pseudo_labels) where:
    - pseudo_data is a tensor of data that has been pseudo-labeled.
    - pseudo_labels is a tensor of labels for the pseudo-labeled data.
    """
    model.eval()  # Set the model to evaluation mode
    pseudo_labels = []
    pseudo_data = []

    with torch.no_grad():  # No need to track gradients for this operation
        for data in dataloader:
            data = data.to(device)
            outputs = model(data)
            probabilities = torch.softmax(outputs, dim=1)
            max_probs, predictions = torch.max(probabilities, dim=1)
            confident_mask = max_probs > threshold

            # Filter data and labels where the model is sufficiently confident
            if confident_mask.sum() > 0:
                pseudo_data.append(data[confident_mask])
                pseudo_labels.append(predictions[confident_mask])

    if pseudo_data:
        pseudo_data = torch.cat(pseudo_data)
        pseudo_labels = torch.cat(pseudo_labels)
        return pseudo_data, pseudo_labels
    else:
        return None, None

class CIFAR10WithPseudoLabels(Dataset):
    def __init__(self, labeled_data, labels, unlabeled_data, transform=None):
        """
        Initialize dataset with labeled and unlabeled data.
        labeled_data: Tensor - The images of the initially labeled dataset.
        labels: Tensor - The labels of the initially labeled dataset.
        unlabeled_data: Tensor - The images of the unlabeled dataset.
        transform: Callable - Optional transform to be applied on a sample.
        """
        self.labeled_data = labeled_data
        self.labels = labels
        self.unlabeled_data = unlabeled_data
        self.transform = transform
        self.pseudo_labels = None  # This will store the pseudo-labels when they are generated

    def update_pseudo_labels(self, pseudo_data, pseudo_labels):
        """
        Update the dataset with pseudo-labeled data.
        pseudo_data: Tensor - The subset of unlabeled data that has been pseudo-labeled.
        pseudo_labels: Tensor - The pseudo-labels for the data.
        """
        self.labeled_data = torch.cat([self.labeled_data, pseudo_data], dim=0)
        self.labels = torch.cat([self.labels, pseudo_labels], dim=0)

        # Update unlabeled data by removing the pseudo-labeled instances
        mask = torch.ones(len(self.unlabeled_data), dtype=torch.bool)
        for data in pseudo_data:
            mask &= (self.unlabeled_data != data).any(dim=1)
        self.unlabeled_data = self.unlabeled_data[mask]

    def __len__(self):
        return len(self.labeled_data) + (len(self.unlabeled_data) if self.pseudo_labels is None else 0)

    def __getitem__(self, idx):
        if idx < len(self.labeled_data):
            img = self.labeled_data[idx]
            label = self.labels[idx]
        else:
            img = self.unlabeled_data[idx - len(self.labeled_data)]
            label = self.pseudo_labels[idx - len(self.labeled_data)] if self.pseudo_labels else -1  # -1 for unlabeled

        if self.transform:
            img = self.transform(img)

        return img, label
