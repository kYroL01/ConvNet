# Advanced Troubleshooting

## Overview

This guide provides solutions to advanced problems you may encounter while using the ConvNet AlexNet implementation.

## Training Issues

### Issue: Out of Memory Error

**Symptoms:**
```
tensorflow.python.framework.errors_impl.ResourceExhaustedError: OOM when allocating tensor
```

**Causes:**
- Batch size too large for GPU/CPU memory
- Model too large
- Accumulating tensors in memory

**Solutions:**

1. **Reduce batch size**
   ```python
   # In my_alexnet_cnn.py:22
   BATCH_SIZE = 32  # Reduce from 64
   # Or even smaller
   BATCH_SIZE = 16
   ```

2. **Clear GPU memory**
   ```python
   import tensorflow as tf
   tf.keras.backend.clear_session()
   ```

3. **Use mixed precision training** (TensorFlow 2.x)
   ```python
   from tensorflow.keras import mixed_precision
   policy = mixed_precision.Policy('mixed_float16')
   mixed_precision.set_global_policy(policy)
   ```

4. **Close other GPU applications**
   ```bash
   nvidia-smi  # Check GPU usage
   # Kill other processes using GPU
   ```

### Issue: NaN Loss or Infinite Loss

**Symptoms:**
```
Loss = nan
Loss = inf
```

**Causes:**
- Learning rate too high
- Gradient explosion
- Numerical instability
- Invalid data

**Solutions:**

1. **Reduce learning rate dramatically**
   ```bash
   python my_alexnet_cnn.py train --learning-rate 0.00001
   ```

2. **Add gradient clipping**
   ```python
   optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
   ```

3. **Check for invalid data**
   ```python
   # Add to preprocessing
   assert not np.any(np.isnan(image))
   assert not np.any(np.isinf(image))
   ```

4. **Use float64 instead of float32**
   ```python
   image = tf.image.convert_image_dtype(img_u8, tf.float64)
   ```

### Issue: Loss Not Decreasing

**Symptoms:**
- Loss stays constant or decreases very slowly
- Accuracy remains at baseline (~25%)

**Causes:**
- Learning rate too low
- Dead ReLU units
- Poor weight initialization
- Data not shuffled
- Bug in training loop

**Solutions:**

1. **Increase learning rate**
   ```bash
   python my_alexnet_cnn.py train --learning-rate 0.01
   ```

2. **Verify data is shuffled**
   ```bash
   python my_alexnet_cnn.py preprocessing_training -f images_dataset.pkl --shuffle
   ```

3. **Check labels are correct**
   ```python
   # Print some labels during loading
   for img, label in loadDataset('images_shuffled.pkl'):
       print(label)  # Should be one-hot: [1,0,0,0], [0,1,0,0], etc.
       break
   ```

4. **Reinitialize model**
   ```bash
   # Remove old checkpoint
   rm -rf ckpt_dir/
   # Train from scratch
   python my_alexnet_cnn.py train
   ```

### Issue: Training Very Slow

**Symptoms:**
- Takes hours per epoch
- Much slower than expected

**Causes:**
- Running on CPU instead of GPU
- Large images not preprocessed
- Inefficient data loading
- Debug mode enabled

**Solutions:**

1. **Verify GPU is being used**
   ```python
   import tensorflow as tf
   print("GPUs:", tf.config.list_physical_devices('GPU'))
   ```

2. **Check preprocessing was done**
   ```bash
   ls -lh images_shuffled.pkl  # Should exist and be reasonable size
   ```

3. **Monitor CPU/GPU usage**
   ```bash
   # CPU
   htop

   # GPU
   nvidia-smi -l 1
   ```

4. **Disable eager execution logging**
   ```python
   import os
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
   ```

### Issue: Model Overfitting Quickly

**Symptoms:**
- Training accuracy high, validation accuracy low
- Large gap between training and validation loss
- Overfitting within first few epochs

**Causes:**
- Too little data
- Model too complex for data
- Not enough regularization

**Solutions:**

1. **Increase dropout**
   ```python
   # In my_alexnet_cnn.py
   input_dropout = 0.6   # Decrease keep rate
   hidden_dropout = 0.3  # Decrease keep rate
   ```

2. **Add data augmentation**
   ```python
   # In Dataset.py, add to convertDataset()
   image = tf.image.random_flip_left_right(image)
   image = tf.image.random_brightness(image, max_delta=0.2)
   ```

3. **Collect more training data**
   - Aim for at least 500 images per class
   - Ensure diversity in training set

4. **Use early stopping**
   - Monitor validation accuracy
   - Stop when it stops improving

## Data Issues

### Issue: "No such file or directory"

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'dataset/Cat/image.jpg'
```

**Causes:**
- Dataset not in correct location
- Incorrect directory structure
- Case-sensitive filesystem

**Solutions:**

1. **Verify directory structure**
   ```bash
   ls -R dataset/
   # Should show: dataset/Cat/, dataset/Tree/, etc.
   ```

2. **Check case sensitivity**
   ```bash
   # Ensure exact names
   dataset/Cat/    # Not dataset/cat/
   dataset/Tree/   # Not dataset/tree/
   ```

3. **Use absolute paths**
   ```python
   # In my_alexnet_cnn.py
   TRAIN_IMAGE_DIR = '/absolute/path/to/dataset'
   ```

### Issue: Images Not Loading

**Symptoms:**
- Preprocessing completes but creates empty file
- "No images found" error

**Causes:**
- Wrong file format (not JPEG)
- Incorrect file extensions
- Hidden files or system files

**Solutions:**

1. **Check file extensions**
   ```bash
   find dataset/ -type f -name "*.jpg" | head
   find dataset/ -type f -name "*.jpeg" | head
   ```

2. **Convert to JPEG if needed**
   ```bash
   # Using ImageMagick
   mogrify -format jpg *.png
   ```

3. **Remove non-image files**
   ```bash
   # Remove .DS_Store, Thumbs.db, etc.
   find dataset/ -name "*.db" -delete
   find dataset/ -name ".DS_Store" -delete
   ```

### Issue: Corrupted Pickle File

**Symptoms:**
```
pickle.UnpicklingError
EOFError
```

**Causes:**
- Preprocessing interrupted
- Disk full during preprocessing
- File corruption

**Solutions:**

1. **Delete and regenerate**
   ```bash
   rm images_shuffled.pkl
   python my_alexnet_cnn.py preprocessing_training -f images_dataset.pkl --shuffle
   ```

2. **Check disk space**
   ```bash
   df -h
   ```

3. **Verify file integrity**
   ```bash
   # File should be >1MB for reasonable dataset
   ls -lh images_shuffled.pkl
   ```

## Model Loading Issues

### Issue: "No model checkpoint found to restore"

**Symptoms:**
```
No model checkpoint found to restore - ERROR
```

**Causes:**
- Model not trained yet
- Checkpoint directory doesn't exist
- Checkpoint files deleted

**Solutions:**

1. **Train model first**
   ```bash
   python my_alexnet_cnn.py train
   ```

2. **Verify checkpoint exists**
   ```bash
   ls -la ckpt_dir/
   # Should show model.ckpt.* files
   ```

3. **Check checkpoint path**
   ```python
   # In my_alexnet_cnn.py, verify:
   MODEL_CKPT = 'ckpt_dir/model.ckpt'
   ```

### Issue: Checkpoint Incompatible

**Symptoms:**
```
NotFoundError: Key not found in checkpoint
```

**Causes:**
- Model architecture changed
- TensorFlow version mismatch
- Corrupted checkpoint

**Solutions:**

1. **Remove old checkpoints and retrain**
   ```bash
   rm -rf ckpt_dir/
   python my_alexnet_cnn.py train
   ```

2. **Use compatible TensorFlow version**
   ```bash
   pip install tensorflow==2.10.0
   ```

3. **Don't modify model architecture between training and prediction**

## Performance Issues

### Issue: Poor Accuracy on Test Set

**Symptoms:**
- Training accuracy >80%, test accuracy <50%
- Large performance gap

**Causes:**
- Overfitting
- Test set different from training set
- Data leakage
- Bug in evaluation

**Solutions:**

1. **Check test set similarity**
   - Ensure test images are from same distribution
   - Verify preprocessing is identical

2. **Increase regularization**
   - See "Model Overfitting Quickly" above

3. **Verify no data leakage**
   ```bash
   # Ensure no overlap between train and test
   # Check that test_dataset/ and dataset/ are completely separate
   ```

4. **Debug evaluation code**
   ```python
   # Print predictions vs. ground truth
   print("Predicted:", y_pred[:10])
   print("Actual:", y_true[:10])
   ```

### Issue: All Predictions Same Class

**Symptoms:**
- Model predicts only one class for all images
- Recall=1.0 for one class, 0.0 for others

**Causes:**
- Class imbalance
- Bug in loss function
- Dead neurons
- Incorrect label encoding

**Solutions:**

1. **Check class balance**
   ```python
   from collections import Counter
   # Count images in each class
   for dirName in os.listdir('dataset/'):
       count = len(os.listdir(f'dataset/{dirName}'))
       print(f"{dirName}: {count}")
   ```

2. **Verify label encoding**
   ```python
   # Should have variation, not all same
   for img, label in loadDataset('images_shuffled.pkl'):
       print(label)
       if iteration > 10:
           break
   ```

3. **Retrain with balanced dataset**
   - Ensure similar number of images per class
   - Or use class weighting in loss function

## Installation Issues

### Issue: TensorFlow Not Found

**Symptoms:**
```
ModuleNotFoundError: No module named 'tensorflow'
```

**Solutions:**

```bash
pip install tensorflow
# Or for GPU support
pip install tensorflow-gpu
```

### Issue: CUDA/cuDNN Errors

**Symptoms:**
```
Could not load dynamic library 'libcudart.so.11.0'
```

**Solutions:**

1. **Install CUDA toolkit**
   - Check TensorFlow compatibility: https://www.tensorflow.org/install/gpu
   - Install matching CUDA version

2. **Use CPU-only version**
   ```bash
   pip uninstall tensorflow-gpu
   pip install tensorflow
   ```

### Issue: NumPy/SciKit-Learn Compatibility

**Symptoms:**
```
AttributeError: module 'numpy' has no attribute 'something'
```

**Solutions:**

```bash
# Update to compatible versions
pip install --upgrade numpy
pip install --upgrade scikit-learn
```

## Debugging Techniques

### Enable Verbose Logging

```python
# In my_alexnet_cnn.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Data at Each Step

```python
# After loading data
print("Data shape:", batch_imgs.shape)
print("Label shape:", batch_labels.shape)
print("Data range:", batch_imgs.min(), batch_imgs.max())

# After preprocessing
print("Preprocessed shape:", img.shape)
print("Label:", label)
```

### Visualize Images

```python
import matplotlib.pyplot as plt

# Visualize preprocessed image
img_reshaped = img.reshape(224, 224, 3)
plt.imshow(img_reshaped)
plt.title(f"Label: {label}")
plt.show()
```

### Profile Performance

```python
import time

start = time.time()
# Code to profile
elapsed = time.time() - start
print(f"Elapsed: {elapsed:.2f} seconds")
```

### Check GPU Utilization

```bash
# Monitor GPU usage in real-time
watch -n 1 nvidia-smi

# Check TensorFlow GPU config
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Common Error Messages

### "Failed to get convolution algorithm"

**Cause:** GPU memory issues or driver problems

**Solution:**
```python
# Allow memory growth
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### "Input 0 of layer is incompatible with the layer"

**Cause:** Input shape mismatch

**Solution:**
- Verify input images are 224×224×3
- Check batch dimensions
- Ensure preprocessing was done correctly

### "Attempting to use uninitialized value"

**Cause:** Variables not initialized

**Solution:**
```python
# Ensure model is built
model.build(input_shape=(None, 224*224, 3))
```

## Getting Help

If issues persist:

1. **Check logs**
   ```bash
   cat FileLog.log
   ```

2. **Search error message**
   - Google the exact error
   - Check TensorFlow GitHub issues
   - Search Stack Overflow

3. **Create minimal reproducible example**
   - Simplify to smallest failing case
   - Document steps to reproduce

4. **Report issue on GitHub**
   - Include error message
   - Include system info (TensorFlow version, Python version, OS)
   - Include steps to reproduce

## System Information

Gather this info when reporting issues:

```bash
# Python version
python --version

# TensorFlow version
python -c "import tensorflow as tf; print(tf.__version__)"

# GPU info
nvidia-smi

# OS info
uname -a

# Disk space
df -h

# Memory
free -h
```

## Prevention Best Practices

1. **Always backup checkpoints** before making changes
2. **Use version control** (git) for code
3. **Document experiments** and configurations
4. **Test with small dataset** before full training
5. **Monitor resources** (memory, disk, GPU)
6. **Keep dependencies updated** but stable
7. **Validate data** before preprocessing
8. **Use virtual environments** to avoid conflicts

## Summary Checklist

When encountering issues:

- [ ] Read error message carefully
- [ ] Check logs (FileLog.log, console output)
- [ ] Verify file paths and permissions
- [ ] Confirm data is preprocessed correctly
- [ ] Check TensorFlow/GPU setup
- [ ] Try with smaller batch size
- [ ] Verify model checkpoint exists (for prediction)
- [ ] Search error message online
- [ ] Create minimal reproducible example
- [ ] Report issue if needed
