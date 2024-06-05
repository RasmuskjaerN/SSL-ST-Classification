from transformers import DeiTModel, DeiTConfig, AutoImageProcessor, AutoFeatureExtractor
import torch
import torch.nn as nn
from typing import Tuple
import os
import requests
from PIL import Image
import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Student(nn.Module):
    def __init__(self, num_labels: int):
        super(Student, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_checkpoint = "facebook/deit-small-distilled-patch16-224"

        self.student_config = DeiTConfig.from_pretrained(self.model_checkpoint)
        self.student_model = DeiTModel.from_pretrained(self.model_checkpoint, config=self.student_config).to(self.device)
        self.classifier = nn.Linear(self.student_config.hidden_size, num_labels).to(self.device)

        self.image_processor = AutoImageProcessor.from_pretrained(self.model_checkpoint)

    def preprocess(self, images):
        # Log the size of the input images before processing
        logging.info(f"Original image size: {images[0].size() if images.nelement() > 0 else 'No images'}")

        processed_images = self.image_processor(images=images, return_tensors="pt").to(self.device)
        # Log the size of the processed images (should be 1, 3, 224, 224)
        logging.info(f"Processed image tensor size: {processed_images['pixel_values'].shape}")
        return processed_images['pixel_values']

    def forward(self, pixel_values):
        logging.info("Performing forward pass.")
        outputs = self.student_model(pixel_values=pixel_values)
        last_hidden_states = outputs.last_hidden_state
        pooled_output = last_hidden_states[:, 0]
        logits = self.classifier(pooled_output)
        return logits, last_hidden_states

if __name__ == "__main__":
    student = Student(num_labels=10)
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    
    inputs = student.preprocess([image])
    
    with torch.no_grad():
        logits, features = student(pixel_values=inputs["pixel_values"])
        print("Logits shape:", logits.shape)
        print("Features shape:", features.shape)



