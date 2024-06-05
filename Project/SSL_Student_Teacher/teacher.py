from transformers import ViTModel, ViTConfig, AutoImageProcessor, AutoFeatureExtractor
import torch
import torch.nn as nn
from typing import Tuple
import os
import requests
from PIL import Image
import numpy as np
import logging

class Teacher(nn.Module):
    def __init__(self):
        super(Teacher, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_checkpoint = "google/vit-large-patch16-224"

        self.teacher_model = ViTModel.from_pretrained(self.model_checkpoint).to(self.device)
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_checkpoint)

    def preprocess(self, images):
        # Log the size of the input images before processing
        logging.info(f"Original image size: {images[0].size() if images.nelement() > 0 else 'No images'}")
        processed_images = self.image_processor(images=images, return_tensors="pt").to(self.device)
        # Log the size of the processed images
        logging.info(f"Processed image tensor size: {processed_images['pixel_values'].shape}")
        return processed_images['pixel_values']

    def forward(self, pixel_values):
        logging.info("Performing forward pass.")
        outputs = self.teacher_model(pixel_values=pixel_values)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states

if __name__ == "__main__":
    teacher = Teacher()
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    # Assuming preprocess accepts a list of images
    inputs = teacher.preprocess([image])
    
    with torch.no_grad():
        features = teacher(pixel_values=inputs["pixel_values"])
        print("Features shape:", features.shape)




