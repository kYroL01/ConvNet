# AlexNet Architecture Theory

## Overview

AlexNet is a pioneering deep convolutional neural network architecture that won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012. This implementation adapts the AlexNet architecture for a 4-class image classification task.

## Historical Context

AlexNet, developed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, demonstrated that deep convolutional neural networks could significantly outperform traditional computer vision methods. It achieved a top-5 error rate of 15.3% on ImageNet, compared to 26.2% from the runner-up.

## Network Architecture

### Input Layer
- **Input size**: 224×224×3 (RGB images)
- **Input dropout**: 0.8 (keeps 80% of neurons during training)
- Images are automatically resized using crop/pad operations to ensure consistent input dimensions

### Convolutional Layers

The network consists of 5 convolutional blocks:

#### Conv Block 1
- **Conv2D**: 64 filters, 11×11 kernel, stride 4
- **Activation**: ReLU (Rectified Linear Unit)
- **Max Pooling**: 3×3 pool, stride 2
- **Batch Normalization**: Normalizes activations
- **Dropout**: 0.2 (input dropout rate)

#### Conv Block 2
- **Conv2D**: 128 filters, 5×5 kernel, stride 1
- **Activation**: ReLU
- **Max Pooling**: 3×3 pool, stride 2
- **Batch Normalization**: Normalizes activations

#### Conv Block 3
- **Conv2D**: 256 filters, 3×3 kernel, stride 1
- **Activation**: ReLU
- **Max Pooling**: 3×3 pool, stride 2
- **Batch Normalization**: Normalizes activations
- **Dropout**: 0.5 (hidden dropout rate)

#### Conv Block 4
- **Conv2D**: 256 filters, 3×3 kernel, stride 1
- **Activation**: ReLU
- **Max Pooling**: 3×3 pool, stride 2
- **Batch Normalization**: Normalizes activations
- **Dropout**: 0.5 (hidden dropout rate)

#### Conv Block 5
- **Conv2D**: 256 filters, 3×3 kernel, stride 1
- **Activation**: ReLU
- **Max Pooling**: 3×3 pool, stride 2

### Fully Connected Layers

After the convolutional blocks, the network includes:

1. **Flatten Layer**: Converts 3D feature maps to 1D feature vectors
2. **FC1**: Dense layer with 4096 neurons, ReLU activation
3. **FC2**: Dense layer with 1024 neurons (2×2×256), ReLU activation
4. **Dropout**: 0.5 dropout rate
5. **Output Layer**: Dense layer with 4 neurons (one per class)

### Output
- **Softmax activation** (applied during loss calculation)
- **4 classes**: Cat, Tree, Horse, Dog

## Key Concepts

### Convolution Operation

Convolution applies a filter (kernel) across the input image to detect features:
- Early layers detect simple features (edges, colors)
- Deeper layers detect complex patterns (shapes, objects)
- Stride controls how much the filter moves
- Padding maintains spatial dimensions

### ReLU Activation

ReLU (Rectified Linear Unit) is defined as: `f(x) = max(0, x)`

Benefits:
- Prevents vanishing gradient problem
- Computationally efficient
- Introduces non-linearity
- Sparse activation (many zeros)

### Max Pooling

Max pooling reduces spatial dimensions by taking the maximum value in each region:
- Reduces computational cost
- Provides translation invariance
- Helps prevent overfitting
- Reduces number of parameters

### Batch Normalization

Normalizes layer inputs to have zero mean and unit variance:
- Accelerates training
- Reduces internal covariate shift
- Acts as a form of regularization
- Allows higher learning rates

### Dropout

Randomly drops neurons during training:
- Prevents overfitting
- Forces network to learn robust features
- Acts as ensemble learning
- Not applied during inference

## Loss Function: Categorical Cross-Entropy

For multi-class classification, the network uses categorical cross-entropy:

```
L = -Σ(y_true * log(y_pred))
```

Where:
- `y_true` is the one-hot encoded true label
- `y_pred` is the predicted probability distribution

## Optimizer: Adam

Adam (Adaptive Moment Estimation) combines:
- **Momentum**: Uses moving average of gradients
- **RMSprop**: Adapts learning rate per parameter
- **Bias correction**: Corrects initialization bias

Default parameters:
- Learning rate: 0.001
- Epsilon: 0.1

## Weight Initialization

Weights are initialized using a normal distribution with standard deviation 0.1:
- Prevents symmetry breaking
- Helps gradient flow
- Avoids saturation

## Regularization Techniques

This implementation uses multiple regularization techniques:

1. **Dropout**: 0.8 for input, 0.5 for hidden layers
2. **Batch Normalization**: After most convolutional layers
3. **Data Augmentation**: Through preprocessing (crop/pad)
4. **Weight Initialization**: Proper initialization prevents exploding/vanishing gradients

## Forward Propagation

The forward pass flow:
```
Input (224×224×3)
→ Conv Block 1 → Conv Block 2 → Conv Block 3 → Conv Block 4 → Conv Block 5
→ Flatten
→ FC1 (4096) → FC2 (1024) → Dropout
→ Output (4)
→ Softmax (during loss calculation)
```

## Backpropagation

During training:
1. Forward pass computes predictions
2. Loss is calculated using categorical cross-entropy
3. Gradients are computed via automatic differentiation
4. Adam optimizer updates weights
5. Process repeats for each batch

## Why AlexNet Works

1. **Hierarchical Feature Learning**: Learns features automatically from data
2. **Deep Architecture**: Multiple layers allow complex pattern recognition
3. **Regularization**: Prevents overfitting despite large capacity
4. **GPU Acceleration**: Efficient implementation on modern hardware
5. **Proven Track Record**: Validated on large-scale image recognition tasks

## Differences from Original AlexNet

This implementation differs from the original 2012 AlexNet:

1. **Framework**: TensorFlow 2.x/Keras instead of custom CUDA code
2. **Batch Normalization**: Added for faster convergence (not in original)
3. **Fewer Classes**: 4 classes instead of 1000 (ImageNet)
4. **Simplified Architecture**: Adapted for smaller datasets
5. **Modern Optimizer**: Adam instead of SGD with momentum
6. **Input Size**: Maintains 224×224 from original

## References

- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *Advances in neural information processing systems*, 25.
- Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *International conference on machine learning*.
- Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.
