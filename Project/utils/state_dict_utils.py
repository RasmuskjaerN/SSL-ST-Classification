import torch
def adjust_state_dict(state_dict):
    """
    Adjusts the keys in the loaded state dictionary by removing the 'student_model.' prefix.
    
    Args:
        state_dict (dict): The state dictionary loaded from the custom-trained model.
    
    Returns:
        dict: The adjusted state dictionary with corrected keys.
    """
    return {key.replace("student_model.", ""): value for key, value in state_dict.items()}

