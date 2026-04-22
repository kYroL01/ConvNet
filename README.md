[![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)
[![Python 3](https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square&logo=python&logoColor=yellow)](https://www.python.org/)
[![GitHub release](https://img.shields.io/github/v/tag/kYroL01/ConvNet?style=flat-square)](https://github.com/kYroL01/ConvNet/tags)

# ConvNet

A convolutional neural network for image recognition using AlexNet architecture implemented with TensorFlow 2.x and Keras API.

## Overview

This project implements an AlexNet-based CNN for multi-class image classification. The model is designed to classify images into 4 categories: Cat, Tree, Horse, and Dog. It uses TensorFlow 2.x with the Keras API for a modern, efficient implementation.

## Documentation

For detailed documentation, advanced topics, and in-depth guides, please visit the **[Wiki](../../wiki)**:

- **[AlexNet Architecture Theory](../../wiki/AlexNet-Architecture-Theory)** - Deep dive into the model architecture and theory
- **[Advanced Training Configuration](../../wiki/Advanced-Training-Configuration)** - Detailed guide on all training parameters
- **[Data Preparation Guide](../../wiki/Data-Preparation-Guide)** - Complete guide to preparing your dataset
- **[Hyperparameter Tuning](../../wiki/Hyperparameter-Tuning)** - Strategies for optimizing model performance
- **[Model Performance and Metrics](../../wiki/Model-Performance-and-Metrics)** - Understanding evaluation metrics
- **[Advanced Troubleshooting](../../wiki/Advanced-Troubleshooting)** - Solutions to common and advanced issues
- **[TensorFlow 2.x Migration Notes](../../wiki/TensorFlow-2x-Migration-Notes)** - Migration details from TensorFlow 1.x

The README below covers the basics to get you started quickly. For comprehensive information, theory, and advanced usage, refer to the Wiki.

## Model Architecture

The AlexNet model consists of:
- **5 Convolutional blocks** with ReLU activation, max pooling, batch normalization, and dropout
- **2 Fully connected layers** (4096 and 1024 neurons)
- **Output layer** with 4 classes
- **Regularization**: Dropout (0.8 for input, 0.5 for hidden layers) and Batch Normalization
- **Input size**: 224x224x3 RGB images
- **Optimizer**: Adam (default learning rate: 0.001, epsilon: 0.1)
- **Loss function**: Categorical cross-entropy

## Requirements

- Python 3.x (Note: Originally designed for Python 2.7, but updated for Python 3.x compatibility)
- [TensorFlow](https://www.tensorflow.org/) 2.x
- [NumPy](https://github.com/numpy/numpy)
- [Matplotlib](https://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

### Installation

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Dataset Structure

Organize your dataset in the following directory structure:

```
ConvNet/
├── dataset/              # Training images
│   ├── Cat/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── Tree/
│   ├── Horse/
│   └── Dog/
└── test_dataset/         # Test images
    ├── Cat/
    ├── Tree/
    ├── Horse/
    └── Dog/
```

**Image Requirements**:
- Format: JPEG (.jpg or .jpeg)
- Recommended size: 224x224 pixels (images will be automatically resized)
- Color: RGB (3 channels)

## Usage

### 1. Preprocessing Training Data

Convert your training images to a compressed pickle format:

```bash
python my_alexnet_cnn.py preprocessing_training -f images_dataset.pkl
```

**Shuffle the training dataset** (recommended for better training):

```bash
python my_alexnet_cnn.py preprocessing_training -f images_dataset.pkl --shuffle
```

This creates an `images_shuffled.pkl` file with randomly shuffled training data.

### 2. Preprocessing Test Data

Convert your test images to a compressed pickle format:

```bash
python my_alexnet_cnn.py preprocessing_test -t images_test_dataset.pkl
```

### 3. Training the Model

Train the model with default parameters:

```bash
python my_alexnet_cnn.py train
```

**Training with custom parameters**:

```bash
python my_alexnet_cnn.py train \
  --learning-rate 0.0001 \
  --max_epochs 50 \
  --display-step 5 \
  --dataset_training images_shuffled.pkl
```

**Available training parameters**:
- `-lr, --learning-rate`: Learning rate (default: 0.001)
- `-e, --max_epochs`: Maximum number of epochs (default: 100)
- `-ds, --display-step`: Display step for logging (default: 10)
- `-dtr, --dataset_training`: Training dataset file (default: 'images_shuffled.pkl')

**Training outputs**:
- Model checkpoint: `ckpt_dir/model.ckpt`
- TensorBoard logs: `ckpt_dir/`
- Training log: `FileLog.log`
- ROC curve visualization (displayed after training)

### 4. Making Predictions

Run predictions on test data:

```bash
python my_alexnet_cnn.py predict --dataset_test images_test_dataset.pkl
```

The prediction will:
- Load the trained model from `ckpt_dir/model.ckpt`
- Process all test images
- Output classification report with precision, recall, and F1-score

## Examples

### Complete Workflow Example

```bash
# Step 1: Prepare training data
python my_alexnet_cnn.py preprocessing_training -f images_dataset.pkl --shuffle

# Step 2: Prepare test data
python my_alexnet_cnn.py preprocessing_test -t images_test_dataset.pkl

# Step 3: Train the model
python my_alexnet_cnn.py train \
  --learning-rate 0.001 \
  --max_epochs 100 \
  --display-step 10 \
  --dataset_training images_shuffled.pkl

# Step 4: Evaluate on test set
python my_alexnet_cnn.py predict --dataset_test images_test_dataset.pkl
```

### Quick Training Example

For a quick test with fewer epochs:

```bash
python my_alexnet_cnn.py train --max_epochs 10 --display-step 2
```

## Output and Metrics

After training, the model provides:

1. **Training Metrics**:
   - Training accuracy and loss per batch
   - Validation accuracy

2. **Classification Report**:
   - Precision, Recall, and F1-score for each class
   - Overall accuracy

3. **Visualizations**:
   - ROC curve for model performance

4. **TensorBoard Support**:
   ```bash
   tensorboard --logdir=ckpt_dir/
   ```

## File Structure

```
ConvNet/
├── my_alexnet_cnn.py       # Main script with AlexNet model and training logic
├── Dataset.py              # Dataset preprocessing utilities
├── README.md               # This file
├── LICENSE                 # MIT License
├── dataset/                # Training images directory
├── test_dataset/           # Test images directory
├── ckpt_dir/               # Model checkpoints and TensorBoard logs
├── FileLog.log             # Training and prediction logs
├── images_dataset.pkl      # Preprocessed training data
├── images_shuffled.pkl     # Shuffled training data
└── images_test_dataset.pkl # Preprocessed test data
```

## Classes and Labels

The model classifies images into 4 categories:

| Class ID | Label  |
|----------|--------|
| 0        | Cat    |
| 1        | Tree   |
| 2        | Horse  |
| 3        | Dog    |

## Technical Details

- **Batch Size**: 64
- **Image Processing**: Images are resized to 224x224 using crop/pad operations
- **Data Format**: Compressed pickle files (.pkl) with gzip compression
- **Checkpointing**: Model weights saved in TensorFlow checkpoint format (.ckpt)

## Troubleshooting

**Issue**: `No model checkpoint found to restore - ERROR`
- **Solution**: Make sure to train the model first before running predictions

**Issue**: Memory errors during training
- **Solution**: Reduce batch size in `my_alexnet_cnn.py` (BATCH_SIZE variable)

**Issue**: Images not loading
- **Solution**: Ensure images are in JPEG format (.jpg or .jpeg) and organized in the correct directory structure

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

Copyright (c) 2017 Michele Campus

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

[![Buy me a coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-PayPal-00457C?logo=paypal&logoColor=white)](https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=fci1908%40gmail.com&currency_code=USD)

PayPal: **fci1908@gmail.com**

