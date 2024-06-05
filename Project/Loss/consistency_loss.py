import torch
import torch.nn.functional as F

def consistency_loss(student_output: torch.Tensor, teacher_output: torch.Tensor, temperature: float) -> torch.Tensor:
    '''
    Implement the Consistency Regularization Process
    Use a loss function like Mean Squared Error to measure the difference between the teacher's and student's outputs.
    Apply a temperature scaling to smooth the output probabilities, which is crucial for consistency regularization.

    Parameters
    ----------
    student_output: torch.Tensor
        The output from the student model.
    teacher_output: torch.Tensor
        The output from the teacher model.

    Returns
    -------
    torch.Tensor
        The computed consistency loss.
    '''
    student_output_scaled = student_output / temperature
    teacher_output_scaled = teacher_output / temperature

    loss = F.mse_loss(student_output_scaled, teacher_output_scaled)
    
    return loss