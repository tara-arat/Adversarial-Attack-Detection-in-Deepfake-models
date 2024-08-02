# Adversarial-Attack-Detection-in-Deepfake-models

Great! Here's a consolidated README for your single notebook "Deepfake Detection".

---

# Deepfake Detection

This repository contains the implementation of deepfake detection models, including XceptionNet, CNNs, and ResNet, all consolidated in a single Jupyter notebook titled "Deepfake Detection". The models are designed to detect manipulated or synthetic images commonly used in deepfakes.

## Table of Contents

- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Model Training and Evaluation](#model-training-and-evaluation)
  - [XceptionNet](#xceptionnet)
  - [Custom CNN](#custom-cnn)
  - [ResNet](#resnet)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Requirements

Ensure you have the following packages installed before running the notebook:

- Python 3.8+
- Torch
- torchvision
- transformers
- PIL
- NumPy
- OpenCV
- Scikit-learn
- TensorFlow
- Keras

You can install these packages using pip:

```bash
pip install torch torchvision transformers pillow numpy opencv-python scikit-learn tensorflow keras
```

## Data Preparation

The datasets used for training and validation should be organized in the following structure:

```
data/
├── train/
│   ├── real/
│   └── fake/
└── val/
    ├── real/
    └── fake/
```

Each directory (`real` and `fake`) should contain the respective images.

## Model Training and Evaluation

The "Deepfake Detection" notebook covers the training and evaluation of three models: XceptionNet, Custom CNN, and ResNet. Below is a summary of the key steps involved in each model's implementation.

### XceptionNet

The XceptionNet model is fine-tuned for deepfake detection.

#### Key Steps:

1. **Data Transformation**:
   - Resize images to 299x299.
   - Apply data augmentation for training data.
   - Normalize images.

2. **Load Pre-trained Model**:
   - Load the XceptionNet model pre-trained on ImageNet.
   - Replace the final fully connected layer to match the number of classes (real and fake).

3. **Freeze Initial Layers**:
   - Freeze the initial layers to retain the learned features.
   - Only train the last few layers.

4. **Define Loss, Optimizer, and Scheduler**:
   - Use CrossEntropyLoss.
   - Use Adam optimizer with weight decay.
   - Use StepLR scheduler for learning rate decay.

5. **Train and Validate**:
   - Train the model for the specified number of epochs.
   - Evaluate on the validation set and save the best model.

### Custom CNN

The custom CNN is designed specifically for the binary classification of real and fake images.

#### Key Steps:

1. **Data Transformation**:
   - Resize images to 224x224.
   - Normalize images.

2. **Define CNN Architecture**:
   - Three convolutional layers with ReLU activation.
   - Max pooling after each convolutional layer.
   - Flatten layer to convert 3D feature maps to 1D feature vectors.
   - Dense layer with dropout for regularization.
   - Output layer with sigmoid activation for binary classification.

3. **Compile Model**:
   - Use RMSprop optimizer.
   - Use binary crossentropy loss.

4. **Train and Validate**:
   - Train the model for the specified number of epochs.
   - Evaluate on the validation set and save the best model.

### ResNet

The ResNet model is fine-tuned for deepfake detection.

#### Key Steps:

1. **Data Transformation**:
   - Resize images to 224x224.
   - Apply data augmentation for training data.
   - Normalize images.

2. **Load Pre-trained Model**:
   - Load the ResNet model pre-trained on ImageNet.
   - Replace the final fully connected layer to match the number of classes (real and fake).

3. **Freeze Initial Layers**:
   - Freeze the initial layers to retain the learned features.
   - Only train the last few layers.

4. **Define Loss, Optimizer, and Scheduler**:
   - Use CrossEntropyLoss.
   - Use Adam optimizer with weight decay.
   - Use StepLR scheduler for learning rate decay.

5. **Train and Validate**:
   - Train the model for the specified number of epochs.
   - Evaluate on the validation set and save the best model.


## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

---

This README provides a comprehensive guide to the repository, making it easy for others to understand and use the deepfake detection models. Make sure to update any placeholder paths and URLs with your actual data paths and GitHub repository URL.
