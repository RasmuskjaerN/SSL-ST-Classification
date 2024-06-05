import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
import Optimizer

import transformers
from Loss.distillation_loss import distillation_loss
from Loss.feature_dist_loss import feature_distillation_loss, TemperatureScaledFeatureDistillationLoss
from Loss.consistency_loss import consistency_loss
from Loss.ts_agreement_rate import calculate_batch_kl_divergence
from SSL_Student_Teacher.student import Student
from SSL_Student_Teacher.teacher import Teacher





def train_student_with_teacher(
    student_model: nn.Module,
    teacher_model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
    temperature: float = 2.0,
    alpha: float = 0.5,
    teacher_channels=1024, 
    student_channels=384, 
    output_size=None
):
    student_model.train()
    teacher_model.eval()

    distillation_loss_module = TemperatureScaledFeatureDistillationLoss(
        teacher_channels, 
        student_channels, 
        output_size, 
        temperature
    ).to(device)

    classification_loss_fn = nn.CrossEntropyLoss().to(device)
    total_loss, total_classification_loss, total_dist_loss = 0.0, 0.0, 0.0

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")

    for batch_idx, (inputs, labels) in progress_bar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Assuming inputs are raw images. Adjust this based on your DataLoader setup.
        # Preprocess inputs for both student and teacher models. This step typically involves converting
        # PIL images or raw image tensors to the format expected by the model, including resizing, normalization, etc.
        # This code snippet assumes the preprocess method is correctly implemented in both student and teacher models
        # to handle batches of images and prepare them for the model.
        student_inputs = student_model.module.preprocess(inputs)  # Assuming preprocess method returns what's needed for the model
        teacher_inputs = teacher_model.module.preprocess(inputs)


        # Forward pass
        
        student_logits, student_features = student_model(student_inputs)
        with torch.no_grad():
            teacher_features = teacher_model(teacher_inputs)

        # Compute losses
        distillation_loss_val = feature_distillation_loss(student_features, teacher_features, temperature)
        classification_loss_val = classification_loss_fn(student_logits, labels)

        # Backpropagation

        total_loss_val = alpha * classification_loss_val + (1 - alpha) * distillation_loss_val
        optimizer.zero_grad()
        total_loss_val.backward()
        optimizer.step()

        # Logging
        total_loss += total_loss_val.item()
        total_classification_loss += classification_loss_val.item()
        total_dist_loss += distillation_loss_val.item()

        progress_bar.set_postfix({
            'Cls Loss': f'{total_classification_loss / (batch_idx + 1):.4f}',
            'Dist Loss': f'{total_dist_loss / (batch_idx + 1):.4f}',
            'Total Loss': f'{total_loss / (batch_idx + 1):.4f}'
        })

    print(f'End of Epoch {epoch+1}. Total Loss: {total_loss / len(train_loader):.4f}, Classification Loss: {total_classification_loss / len(train_loader):.4f}, Distillation Loss: {total_dist_loss / len(train_loader):.4f}')
    torch.save(student_model.state_dict(), f'student_model_scaled.pth')



