import torch

def distillation_loss(student_output: torch.Tensor, teacher_output: torch.Tensor, temperature: float) -> torch.Tensor:
    '''
    Implement the Distillation Process
    Use a loss function like Kullback-Leibler divergence to measure the difference between the teacher's and student's outputs.
    Apply a temperature scaling in the softmax function to smooth the output probabilities, which is crucial for distillation.

    Parameters
    ----------
    student_output: torch.Tensor
        The output from the student model.
    teacher_output: torch.Tensor
        The output from the teacher model.

    Returns
    -------
    torch.Tensor
        The computed distillation loss.
    '''

    teacher_probs = torch.nn.functional.softmax(teacher_output / temperature, dim=1)
    student_log_probs = torch.nn.functional.log_softmax(student_output / temperature, dim=1)
    loss = torch.nn.functional.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    return loss