# Transforming Vision: Model Proposal Leveraging Transformer Architectures for Semi-Supervised Image Classification

## Overview
This project develops two novel semi-supervised image classification models leveraging transformer architectures. The models, ST-Class and SSL-ST-Class, utilize a Student Teacher network for initialization, employing techniques like Knowledge Distillation and Pseudo-Labels to enhance learning from limited labeled data. The aim is to achieve high accuracy and confidence in classifications while being data-efficient.

### Background
Current supervised learning methods demand extensive labeled datasets, which are costly and time-consuming to produce. Semi-supervised approaches like SSL-ST-Class reduce this requirement significantly, leveraging both labeled and unlabeled data to train models. This project builds on the capabilities of transformers, as discussed in Dosovitskiy et al. (2021) and Touvron et al. (2021), to address these challenges effectively.

## Installation

### Setup Instructions
*Clone the repository*

git clone [https://github.com/Username/SSL-ST-Classification.git](https://github.com/RasmuskjaerN/SSL-ST-Classification.git)

*Navigate into the project directory*

cd projectname

*Install dependencies*

pip install -r requirements.txt

### Usage

**Running the Application**
python main.py

## Documentation

### Code Structure
The project is structured into several modules:

- student_teacher_model.py: Implements the Student Teacher architecture.
- data_loader.py: Handles data preprocessing and loading.
- training.py: Contains the training loops for both ST-Class and SSL-ST-Class models.
- evaluation.py: For performance evaluation and metrics visualization.

## Results
Performance is quantified through accuracy metrics on CIFAR10 and CIFAR100 datasets, showing that our models perform comparably to traditional methods with significantly reduced data requirements. Graphs and tables detailing these metrics are included in the project thesis COMING SOON.
![Accuracy Score cifar100](/bin/cifar100Acc_Loss_Pr_Epoch.png "Accuracy score of ST_Class")

![Precision, Recall and F1-Score](/bin/cifar100scores_across_epochs.png "Precision, Recall and F1-Score")

![Confusion Matrix](/bin/cifar100metrics_and_confusion_matrix_cifar100.png "ST-Class Cifar100 Confusion Matrix")

## Discussion
While the ST-Class model reaches an accuracy of 93.37% on CIFAR10 and 71.32% on CIFAR100, the SSL-ST-Class model achieves 89.44% and 47.52%, respectively. These results validate the effectiveness of semi-supervised learning models in utilizing less labeled data.

## Contributing
How to Contribute
Encourage others to contribute to your project by explaining how they can do so. Provide guidelines for submitting issues, pull requests, and code review standards.

## Acknowledgements
Special thanks to the project supervisor, Hua Lu, and all contributors who have provided insights and support throughout the development of this project.
