# Advanced Training Configuration

## Overview

This guide covers advanced training configuration options and best practices for optimizing your AlexNet model training.

## Command-Line Options

### Training Command Structure

```bash
python my_alexnet_cnn.py train [OPTIONS]
```

### Available Training Parameters

#### Learning Rate (`-lr`, `--learning-rate`)
- **Type**: Float
- **Default**: 0.001
- **Description**: Controls the step size during gradient descent

**Guidelines:**
- Start with 0.001 for most cases
- Use 0.0001 for fine-tuning or if training is unstable
- Use 0.01 for faster convergence (may be less stable)
- Monitor training loss; if it oscillates, reduce learning rate

**Example:**
```bash
python my_alexnet_cnn.py train --learning-rate 0.0001
```

#### Max Epochs (`-e`, `--max_epochs`)
- **Type**: Integer
- **Default**: 100
- **Description**: Maximum number of training epochs

**Guidelines:**
- Start with 50-100 epochs for initial experiments
- Monitor validation accuracy to avoid overfitting
- Use early stopping if validation accuracy plateaus
- More epochs may be needed for larger datasets

**Example:**
```bash
python my_alexnet_cnn.py train --max_epochs 50
```

#### Display Step (`-ds`, `--display-step`)
- **Type**: Integer
- **Default**: 10
- **Description**: Frequency of logging training metrics

**Guidelines:**
- Use 1 for detailed monitoring (slower)
- Use 10 for balanced logging
- Use 50+ for large datasets to reduce I/O overhead

**Example:**
```bash
python my_alexnet_cnn.py train --display-step 5
```

#### Training Dataset (`-dtr`, `--dataset_training`)
- **Type**: String
- **Default**: `images_shuffled.pkl`
- **Description**: Path to preprocessed training data

**Guidelines:**
- Always use shuffled data for better convergence
- Ensure dataset is preprocessed before training
- Use absolute or relative path from project root

**Example:**
```bash
python my_alexnet_cnn.py train --dataset_training custom_dataset.pkl
```

## Complete Training Example

### Basic Training
```bash
python my_alexnet_cnn.py train
```

### Production Training Configuration
```bash
python my_alexnet_cnn.py train \
  --learning-rate 0.001 \
  --max_epochs 100 \
  --display-step 10 \
  --dataset_training images_shuffled.pkl
```

### Quick Experimentation
```bash
python my_alexnet_cnn.py train \
  --learning-rate 0.01 \
  --max_epochs 20 \
  --display-step 2
```

### Fine-Tuning Configuration
```bash
python my_alexnet_cnn.py train \
  --learning-rate 0.0001 \
  --max_epochs 200 \
  --display-step 5
```

## Training Process

### What Happens During Training

1. **Initialization**
   - Model loads from checkpoint (if exists) or initializes randomly
   - Training dataset is loaded from pickle file
   - Adam optimizer is configured

2. **Epoch Loop**
   - For each epoch, all batches are processed
   - Each batch undergoes forward and backward propagation
   - Weights are updated using Adam optimizer

3. **Batch Processing**
   - Batch size: 64 images (configurable in code)
   - Images and labels are fed to the model
   - Loss is calculated using categorical cross-entropy
   - Gradients are computed and applied

4. **Checkpointing**
   - Model is saved after each epoch to `ckpt_dir/model.ckpt`
   - Can resume training from checkpoint

5. **Logging**
   - Training metrics logged to console and `FileLog.log`
   - TensorBoard logs saved to `ckpt_dir/`

### Training Output

The training process outputs:
- Batch-level training loss and accuracy
- Epoch-level validation accuracy (if validation set available)
- ROC curve visualization after training completes
- Model checkpoint files

## Monitoring Training

### TensorBoard

Launch TensorBoard to monitor training in real-time:

```bash
tensorboard --logdir=ckpt_dir/
```

Access at: `http://localhost:6006`

**Available visualizations:**
- Loss curves
- Accuracy trends
- Weight distributions
- Gradient histograms

### Log Files

Training logs are written to `FileLog.log`:

```bash
tail -f FileLog.log
```

## Advanced Configuration (Code-Level)

Some parameters require code modifications:

### Batch Size
**Location**: `my_alexnet_cnn.py:22`
```python
BATCH_SIZE = 64
```

**Guidelines:**
- Reduce if encountering memory errors (e.g., 32, 16)
- Increase for faster training on powerful GPUs (e.g., 128, 256)
- Smaller batches may improve generalization

### Dropout Rates
**Location**: `my_alexnet_cnn.py:28-29`
```python
input_dropout = 0.8    # Keep 80% of input neurons
hidden_dropout = 0.5   # Keep 50% of hidden neurons
```

**Guidelines:**
- Increase dropout to prevent overfitting (e.g., 0.6 for hidden)
- Decrease dropout if model is underfitting (e.g., 0.7 for hidden)
- Input dropout typically higher than hidden dropout

### Weight Initialization
**Location**: `my_alexnet_cnn.py:30`
```python
std_dev = 0.1
```

**Guidelines:**
- Use He initialization for ReLU: `math.sqrt(2/n_input)`
- Current value (0.1) works well for most cases
- Adjust if encountering gradient issues

### Adam Optimizer Epsilon
**Location**: When creating optimizer (search for "Adam" in code)

**Default**: 0.1 (relatively high)

**Guidelines:**
- Standard epsilon: 1e-7 or 1e-8
- Higher epsilon (current) may help with numerical stability
- Reduce if optimizer seems too conservative

## Training Strategies

### Transfer Learning
Although not implemented by default, you can:
1. Train on a larger dataset first
2. Save the checkpoint
3. Fine-tune on your specific dataset with lower learning rate

### Learning Rate Scheduling
Consider implementing:
- **Step decay**: Reduce learning rate every N epochs
- **Exponential decay**: Gradually reduce learning rate
- **Cosine annealing**: Cyclical learning rate

### Data Augmentation
Currently implemented via crop/pad. Consider adding:
- Random flips
- Random rotations
- Color jittering
- Random crops

### Early Stopping
Monitor validation accuracy and stop training when:
- Validation accuracy stops improving
- Validation loss starts increasing (overfitting)

### Regularization Tuning
If overfitting:
- Increase dropout rates
- Add L2 regularization to weights
- Use more data augmentation
- Reduce model complexity

If underfitting:
- Decrease dropout rates
- Increase model capacity
- Train for more epochs
- Increase learning rate

## Common Training Scenarios

### Scenario 1: Fast Convergence, Poor Generalization
**Symptoms**: Training accuracy high, validation accuracy low

**Solutions:**
- Increase dropout rates
- Add data augmentation
- Use more training data
- Reduce model complexity

### Scenario 2: Slow Convergence
**Symptoms**: Loss decreases very slowly

**Solutions:**
- Increase learning rate
- Check data preprocessing
- Verify batch normalization is working
- Ensure data is shuffled

### Scenario 3: Oscillating Loss
**Symptoms**: Loss jumps up and down

**Solutions:**
- Decrease learning rate
- Reduce batch size
- Check for data issues (corrupted images, incorrect labels)

### Scenario 4: Plateau
**Symptoms**: Metrics stop improving

**Solutions:**
- Reduce learning rate (fine-tuning)
- Try different optimizer
- Check if model capacity is sufficient
- Verify data quality

## Training Best Practices

1. **Always shuffle training data** before training
2. **Start with default parameters** for initial experiments
3. **Monitor both training and validation metrics**
4. **Save checkpoints regularly** to prevent data loss
5. **Use TensorBoard** for real-time monitoring
6. **Document your experiments** (parameters, results)
7. **Validate on a separate test set** after training
8. **Use GPU acceleration** for faster training

## Performance Expectations

### Training Time
- **Small dataset** (<1000 images): 5-10 minutes per epoch (CPU)
- **Medium dataset** (1000-10000 images): 1-2 minutes per epoch (GPU)
- **Large dataset** (>10000 images): 5-10 minutes per epoch (GPU)

### Accuracy Targets
- **Random baseline**: 25% (4 classes)
- **Good performance**: 70-80%
- **Excellent performance**: 85-95%
- **Perfect score**: >95% (may indicate overfitting)

## Troubleshooting Training Issues

See [Advanced Troubleshooting](Advanced-Troubleshooting) for detailed solutions to common training problems.
