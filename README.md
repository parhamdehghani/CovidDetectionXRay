# COVID-19 X-Ray Detection Using Deep Learning

## Project Overview
A deep learning system that detects COVID-19 from chest X-ray images using transfer learning with DenseNet201. The model classifies X-rays into four categories: COVID-19, Lung Opacity, Normal, and Viral Pneumonia.

## Key Features
- Transfer learning implementation using DenseNet201 architecture
- Data augmentation for improved model robustness
- 82.4% accuracy on test set
- Real-time prediction capabilities with probability scores
- Conservative COVID-19 detection threshold (>30% probability)

## Technical Details
- Framework: PyTorch
- Model: DenseNet201 (pretrained)
- Training: 20 epochs with Adadelta optimizer
- Data augmentation: Random rotation, flips, and crops
- Input size: 224x224 RGB images
- Output: 4-class classification with probability scores

## Dataset
The model is trained on the COVID-19 Radiography Dataset containing chest X-ray images in four categories:
- COVID-19 positive cases
- Normal cases
- Lung opacity cases
- Viral pneumonia cases

## Results
- Training Accuracy: 78.37%
- Validation Accuracy: 81.68%
- Test Accuracy: 82.40%

## Usage
1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Run inference on new images:
```python
from predict import predict_image
result = predict_image('path_to_xray_image.jpg')
