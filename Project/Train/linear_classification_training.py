import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
from transformers import AutoImageProcessor
import torch
from tqdm import tqdm
import torch.distributed as dist
from torch.distributed import all_gather, get_rank, get_world_size
import numpy as np
import json
import random

def combined_train_validate(model, train_loader, val_loader, optimizer, epoch, device):
    """
    Train and validate a model within the same loop, capturing not only the loss and accuracy 
    but also detailed predictions, confidences, true labels, and image pixels for analysis.

    Args:
        model (torch.nn.Module): The model to be trained and validated.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        epoch (int): Current epoch number.
        device (torch.device): Device to run the training and validation on.

    Returns:
        dict: A dictionary containing aggregated results and detailed data from the validation.
    """
    classification_loss_fn = nn.CrossEntropyLoss().to(device)
    model.to(device)

    # Training Phase
    model.train()
    total_train_loss = 0
    total_train_correct = 0
    total_train_samples = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}, Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = classification_loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        _, predicted = outputs.max(1)
        total_train_correct += predicted.eq(labels).sum().item()
        total_train_samples += labels.size(0)

    # Validation Phase
    model.eval()
    total_val_loss = 0
    total_val_correct = 0
    total_val_samples = 0
    all_val_predictions = []
    all_val_confidences = []
    val_true_labels = []
    val_image_pixels = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}, Validation")):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = classification_loss_fn(outputs, labels)

            total_val_loss += loss.item()
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = probabilities.max(1)

            total_val_correct += predicted.eq(labels).sum().item()
            total_val_samples += labels.size(0)

            all_val_predictions.append(predicted.cpu())
            all_val_confidences.append(probabilities.cpu())
            val_true_labels.append(labels.cpu())
            # Only store pixel data for the first few images
            val_image_pixels.append(inputs.cpu().permute(0, 2, 3, 1)) #.numpy().tolist())

    if all_val_predictions:
        all_val_predictions = torch.cat(all_val_predictions)
        all_val_confidences = torch.cat(all_val_confidences)
        val_true_labels = torch.cat(val_true_labels)
        val_image_pixels = torch.cat(val_image_pixels).numpy().tolist()
        val_image_pixels = val_image_pixels[-25:]
        print(len(val_image_pixels))
        if len(val_image_pixels) >= len(val_loader) - 25:
            val_image_pixels = torch.cat(val_image_pixels)
        print(len(val_image_pixels))
    else:
        all_val_predictions = torch.tensor([], dtype=torch.long)
        all_val_confidences = torch.tensor([], dtype=torch.float)
        val_true_labels = torch.tensor([], dtype=torch.long)
        val_image_pixels = torch.tensor([], dtype=torch.long)
        

    if torch.cuda.device_count() >= 1:
        gathered_predictions = [torch.zeros_like(all_val_predictions).to(device) for _ in range(get_world_size())]
        gathered_confidences = [torch.zeros_like(all_val_confidences).to(device) for _ in range(get_world_size())]
        gathered_val_true_labels = [torch.zeros_like(val_true_labels).to(device) for _ in range(get_world_size())]
        gathered_pixel_data = [torch.zeros_like(val_image_pixels).to(device) if val_image_pixels.numel() > 0 else torch.tensor([]).to(device) for _ in range(get_world_size())]

        all_gather(gathered_predictions, all_val_predictions.to(device))
        all_gather(gathered_confidences, all_val_confidences.to(device))
        all_gather(gathered_val_true_labels, val_true_labels.to(device))
        all_gather(gathered_pixel_data, val_image_pixels.to(device) if val_image_pixels.numel() > 0 else torch.tensor([]).to(device))

        if get_rank() == 0:
            all_val_predictions = torch.cat(gathered_predictions)
            all_val_confidences = torch.cat(gathered_confidences)
            val_true_labels = torch.cat(gathered_val_true_labels)
            val_image_pixels = torch.cat(gathered_pixel_data)
    else:
        gathered_predictions = [torch.zeros_like(all_val_predictions).to(device) for _ in range(get_world_size())]
        gathered_confidences = [torch.zeros_like(all_val_confidences).to(device) for _ in range(get_world_size())]
        gathered_val_true_labels = [torch.zeros_like(val_true_labels).to(device) for _ in range(get_world_size())]

        all_val_predictions = torch.cat(gathered_predictions)
        all_val_confidences = torch.cat(gathered_confidences)
        val_true_labels = torch.cat(gathered_val_true_labels)

    if get_rank() == 0:
        print(f'End of Epoch {epoch+1}. Training Loss: {total_train_loss / len(train_loader):.4f}, '
              f'Training Accuracy: {100. * total_train_correct / total_train_samples:.2f}%, '
              f'Validation Loss: {total_val_loss / len(val_loader):.4f}, '
              f'Validation Accuracy: {100. * total_val_correct / total_val_samples:.2f}%')

    return {
        "train_loss": total_train_loss / len(train_loader),
        "train_accuracy": 100. * total_train_correct / total_train_samples,
        "val_loss": total_val_loss / len(val_loader),
        "val_accuracy": 100. * total_val_correct / total_val_samples,
        "val_predictions": all_val_predictions.tolist(),
        "val_confidences": all_val_confidences.tolist(),
        "val_true_labels": val_true_labels.tolist(),
        "val_image_pixels": val_image_pixels
    }

