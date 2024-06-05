import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def total_variation_distance(
    student_model: nn.Module,
    teacher_model: nn.Module,
    data_loader: DataLoader,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> float:
    """
    Computes the total variation distance between the student and teacher model predictions.

    Args:
        student_model (nn.Module): The student model to compare.
        teacher_model (nn.Module): The teacher model to compare.
        data_loader (DataLoader): DataLoader for the data to compare on.

    Returns:
        float: The total variation distance between the student and teacher model predictions.
    """
    student_model.eval()
    teacher_model.eval()
    total_variation = 0.0
    num_samples = 0
    for input_tensor, _ in data_loader:
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            student_output = student_model(input_tensor)
            teacher_output = teacher_model(input_tensor)
        student_probs = F.softmax(student_output, dim=1)
        teacher_probs = F.softmax(teacher_output, dim=1)
        total_variation += (
            (0.5 * (student_probs - teacher_probs).abs()).sum(dim=1).mean().item()
        )
        num_samples += 1
    return total_variation / num_samples
