import torch
import torch.nn.functional as F

def feature_distillation_loss(student_features: torch.Tensor, teacher_features: torch.Tensor, temperature: float) -> torch.Tensor:
    '''
    Calculate distillation loss based on feature representations.
    
    Parameters
    ----------
    student_features: torch.Tensor
        The feature representations from the student model.
    teacher_features: torch.Tensor
        The feature representations from the teacher model.
    temperature: float
        Temperature scaling is applied if using soft targets. Not directly used here for MSE-based feature distillation.

    Returns
    -------
    torch.Tensor
        The computed distillation loss.
    '''
        # Assuming student_features and teacher_features are 4D tensors of shape [batch_size, channels, height, width]
    # and you need to match the spatial dimensions (height, width)
    teacher_features_downsampled = F.adaptive_avg_pool2d(teacher_features, (student_features.size(1), student_features.size(2)))
    
    loss = F.mse_loss(student_features, teacher_features_downsampled)
    return loss

class ConvolutionalAdaptationLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, output_size=None):
        """
        Initializes the convolutional adaptation layer to adjust feature map dimensions.
        
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            output_size (tuple[int, int], optional): The desired spatial dimensions (H, W) of the output feature maps.
        """
        super(ConvolutionalAdaptationLayer, self).__init__()
        self.output_size = output_size
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 1x1 convolution
        
        if self.output_size:
            self.upsample = torch.nn.Upsample(size=output_size, mode='bilinear', align_corners=False)
        else:
            self.upsample = None

    def forward(self, x):
        """
        Forward pass for adapting feature map dimensions.
        
        Args:
            x (torch.Tensor): Input feature map.
            
        Returns:
            torch.Tensor: Adapted feature map.
        """
        x = self.conv(x)
        if self.upsample:
            x = self.upsample(x)
        return x

class TemperatureScaledFeatureDistillationLoss(torch.nn.Module):
    def __init__(self, teacher_channels, student_channels, output_size, temperature=1.0):
        """
        Initializes the feature distillation loss module with convolutional adaptation and temperature scaling.
        
        Args:
            teacher_channels (int): Number of channels in the teacher's feature map.
            student_channels (int): Number of channels in the student's feature map.
            output_size (tuple[int, int]): Desired output spatial dimensions (H, W) for the adapted feature maps.
            temperature (float): Temperature parameter to scale the feature maps, enhancing or smoothing the feature contrast.
        """
        super(TemperatureScaledFeatureDistillationLoss, self).__init__()
        self.teacher_adapt = ConvolutionalAdaptationLayer(teacher_channels, student_channels, output_size)
        self.student_adapt = ConvolutionalAdaptationLayer(student_channels, student_channels, output_size)
        self.temperature = temperature
    
    def forward(self, teacher_features, student_features):
        """
        Computes the distillation loss between adapted and temperature-scaled teacher and student features.
        
        Args:
            teacher_features (torch.Tensor): Feature map from the teacher model.
            student_features (torch.Tensor): Feature map from the student model.
            
        Returns:
            torch.Tensor: The computed distillation loss.
        """
        # Adapt features
        teacher_adapted = self.teacher_adapt(teacher_features)
        student_adapted = self.student_adapt(student_features)
        
        # Apply temperature scaling
        if self.temperature != 1.0:  # Apply scaling only if temperature is not 1
            teacher_adapted = teacher_adapted / self.temperature
            student_adapted = student_adapted / self.temperature
        
        # Compute loss
        loss = F.mse_loss(student_adapted, teacher_adapted)
        return loss

