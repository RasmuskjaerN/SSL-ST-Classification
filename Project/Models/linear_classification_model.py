import torch
import torch.nn as nn
from transformers import DeiTConfig, DeiTModel, AutoImageProcessor, PretrainedConfig

class SimpleLinearModelConfig(PretrainedConfig):
    model_type = "simple_linear_model"
    
    def __init__(self, student_model_checkpoint='facebook/deit-small-distilled-patch16-224', num_classes=100, hidden_dim=128, **kwargs):
        super().__init__()
        self.student_model_checkpoint = student_model_checkpoint
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.deit_config = DeiTConfig.from_pretrained(student_model_checkpoint, **kwargs).to_dict()

class SimpleLinearModel(nn.Module):
    def __init__(self, config: SimpleLinearModelConfig, state_dict_path=None):
        super(SimpleLinearModel, self).__init__()
        self.config = config
        self.student_model = DeiTModel.from_pretrained(config.student_model_checkpoint, config=DeiTConfig(**config.deit_config))
        self.classifier = nn.Sequential(
            nn.Linear(384, config.hidden_dim),  # Example dimension
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )

        if state_dict_path:
            # Load the state dictionary
            self.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu')), strict=False)

    def load_custom_state_dict(self, state_dict):
        """
        Load a custom state dictionary, handling the classifier weights separately.
        """
        deit_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
        self.student_model.load_state_dict(deit_state_dict, strict=False)

        classifier_state_dict = {k: v for k, v in state_dict.items() if k.startswith('classifier')}
        if classifier_state_dict:
            self.classifier.load_state_dict(classifier_state_dict, strict=False)

    def init_preprocessor(self) -> AutoImageProcessor:
        """
        Initialize and return the image processor.
        """
        return AutoImageProcessor.from_pretrained(self.config.student_model_checkpoint)

    def preprocess(self, images, image_processor: AutoImageProcessor) -> torch.Tensor:
        """
        Preprocess images using the provided image processor.
        """
        processed_images = image_processor.preprocess(images=images, return_tensors="pt")
        return processed_images

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        outputs = self.student_model(pixel_values=pixel_values)
        pooled_output = outputs.last_hidden_state[:, 0] 
        logits = self.classifier(pooled_output)
        return logits