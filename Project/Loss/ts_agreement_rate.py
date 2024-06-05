import torch
import torch.nn.functional as F

def calculate_batch_kl_divergence(student_output: torch.Tensor, teacher_output: torch.Tensor):
    """
    Calculate the KL divergence between teacher and student model predictions for a single batch, with optional temperature scaling.

    Args:
        teacher_model (torch.nn.Module): The teacher model.
        student_model (torch.nn.Module): The student model.
        inputs (Tensor): Input tensor for the batch.
        temperature (float, optional): Temperature parameter for scaling the logits. Default is 1.0.

    Returns:
        Tensor: KL divergence for the batch.
    """
    temperature = 1.5

    # Apply temperature scaling if specified
    if temperature != 1.0:
        teacher_output = teacher_output / temperature
        student_output = student_output / temperature

    # Calculate the softmax and log softmax with temperature applied
    teacher_soft_labels = F.softmax(teacher_output, dim=1)
    student_log_probs = F.log_softmax(student_output, dim=1)

    # Calculate and return the KL divergence
    kl_divergence = F.kl_div(student_log_probs, teacher_soft_labels, reduction='batchmean') * (temperature ** 2)  # Adjust for the temperature effect
    return kl_divergence