# TensorFlow 2.x Migration Notes

## Overview

This project has been migrated from TensorFlow 1.x to TensorFlow 2.x. This document explains the changes made and provides guidance for understanding the modernized codebase.

## Major Changes

### 1. Keras API Integration

**TensorFlow 1.x (Old):**
```python
import tensorflow as tf

# Low-level API
conv1 = tf.layers.conv2d(inputs, filters=64, kernel_size=11)
pool1 = tf.layers.max_pooling2d(conv1, pool_size=3)
```

**TensorFlow 2.x (Current):**
```python
import tensorflow as tf

# Keras API (integrated)
conv1 = tf.keras.layers.Conv2D(64, (11, 11))
pool1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3))
```

**Benefits:**
- More intuitive, object-oriented API
- Better documentation and community support
- Easier to build complex models
- Consistent with Keras best practices

### 2. Eager Execution by Default

**TensorFlow 1.x:**
- Graph-based execution
- Requires session management
- Placeholders for inputs
- Difficult debugging

**TensorFlow 2.x:**
- Eager execution by default
- Immediate evaluation
- No sessions needed
- Easy debugging with Python debugger

**Example:**
```python
# TensorFlow 2.x - immediate execution
x = tf.constant([1, 2, 3])
print(x)  # Prints immediately, no session needed
```

### 3. Model Definition Using tf.keras.Model

**TensorFlow 1.x Pattern:**
```python
def alexnet(x, dropout):
    conv1 = tf.layers.conv2d(x, filters=64, kernel_size=11)
    # ... more layers
    return output
```

**TensorFlow 2.x Pattern (Current):**
```python
class AlexNetModel(tf.keras.Model):
    def __init__(self):
        super(AlexNetModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, (11, 11))
        # ... more layers

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        # ... forward pass
        return x
```

**Benefits:**
- Encapsulation of model logic
- Automatic variable tracking
- Easy to save/load models
- Support for training/inference modes

### 4. No More Placeholders

**TensorFlow 1.x:**
```python
x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
y = tf.placeholder(tf.float32, shape=[None, 4])
```

**TensorFlow 2.x:**
```python
# Data passed directly to model
predictions = model(batch_images, training=True)
```

**Benefits:**
- Simpler code
- No placeholder management
- Direct Python values

### 5. Automatic Differentiation with GradientTape

**TensorFlow 1.x:**
```python
loss = tf.losses.categorical_crossentropy(labels, predictions)
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss)
```

**TensorFlow 2.x:**
```python
with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = loss_fn(labels, predictions)

gradients = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

**In this project:**
- Uses `model.compile()` and `model.fit()` for simplified training
- Automatic gradient computation under the hood

### 6. Model Compilation and Training

**Current implementation uses high-level API:**

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(train_dataset, epochs=100, validation_data=val_dataset)
```

**Benefits:**
- Less boilerplate code
- Automatic metric tracking
- Built-in callbacks support
- TensorBoard integration

### 7. Checkpoint Management

**TensorFlow 1.x:**
```python
saver = tf.train.Saver()
saver.save(sess, 'model.ckpt')
saver.restore(sess, 'model.ckpt')
```

**TensorFlow 2.x:**
```python
# Save
model.save_weights('model.ckpt')

# Restore
model.load_weights('model.ckpt')
```

**In this project:**
- Checkpoints saved in TensorFlow format
- Compatible with TensorFlow 2.x checkpoint system
- Extension: `.ckpt` (not `.cktp`)

### 8. Image Processing

**TensorFlow 1.x:**
```python
# Required session.run()
img = tf.image.decode_jpeg(img_bytes)
img_resized = tf.image.resize_image_with_crop_or_pad(img, 224, 224)
img_array = sess.run(img_resized)
```

**TensorFlow 2.x:**
```python
# Eager execution - immediate evaluation
img_bytes = tf.io.read_file(img_path)
img = tf.image.decode_jpeg(img_bytes, channels=3)
img = tf.image.convert_image_dtype(img, tf.float32)
img_resized = tf.image.resize_with_crop_or_pad(img, 224, 224)
img_array = img_resized.numpy()  # Direct conversion
```

**Changes in Dataset.py:**
- `tf.io.read_file()` instead of `tf.gfile.FastGFile()`
- `tf.image.resize_with_crop_or_pad()` instead of `resize_image_with_crop_or_pad()`
- Direct `.numpy()` conversion

### 9. Batch Normalization

**TensorFlow 1.x:**
```python
bn = tf.layers.batch_normalization(conv, training=is_training)
```

**TensorFlow 2.x:**
```python
self.bn = tf.keras.layers.BatchNormalization()
# In call method:
x = self.bn(x, training=training)
```

**Important:**
- Must pass `training` argument
- Different behavior in training vs. inference
- Affects moving average statistics

### 10. Dropout

**TensorFlow 1.x:**
```python
dropout = tf.layers.dropout(inputs, rate=0.5, training=is_training)
```

**TensorFlow 2.x:**
```python
self.dropout = tf.keras.layers.Dropout(0.5)
# In call method:
x = self.dropout(x, training=training)
```

**Note:**
- Rate parameter is proportion to drop (not keep)
- Must pass `training` flag for proper behavior

## Migration Guide for Contributors

### If Modifying the Code

**DO:**
- Use `tf.keras.layers.*` for layers
- Use `tf.keras.Model` for model classes
- Pass `training` argument to BatchNorm and Dropout
- Use `model.compile()` and `model.fit()` for training
- Use eager execution debugging

**DON'T:**
- Use deprecated `tf.layers.*`
- Use `tf.Session()`
- Use `tf.placeholder()`
- Use `tf.train.Saver()` (use `model.save_weights()`)
- Use `feed_dict`

### Backwards Compatibility

**Not maintained:**
- Code will NOT work with TensorFlow 1.x
- Requires TensorFlow 2.0 or later
- Tested with TensorFlow 2.10+

**To use TensorFlow 1.x code:**
- Check git history for older versions
- Or use compatibility module: `tf.compat.v1` (not recommended)

## Performance Implications

### Speed

**TensorFlow 2.x vs 1.x:**
- Similar performance for most operations
- Eager execution slightly slower than graph mode
- Can use `@tf.function` decorator for graph compilation

**Current implementation:**
- Uses Keras high-level API
- Performance is comparable to TF 1.x version
- GPU acceleration works as expected

### Memory

**TensorFlow 2.x:**
- More aggressive garbage collection
- Better memory management
- Allow GPU memory growth by default

## Common Migration Issues

### Issue 1: Attribute Errors

**Error:**
```
AttributeError: module 'tensorflow' has no attribute 'placeholder'
```

**Cause:** TF 1.x code in TF 2.x environment

**Solution:** Update to TF 2.x patterns shown above

### Issue 2: Session Errors

**Error:**
```
AttributeError: module 'tensorflow' has no attribute 'Session'
```

**Cause:** Trying to use TF 1.x session

**Solution:** Remove session code, use eager execution

### Issue 3: Shape Inference

**Error:**
```
ValueError: Input 0 of layer is incompatible with the layer
```

**Cause:** Shape mismatches, different shape handling in TF 2.x

**Solution:**
- Use `model.build(input_shape=...)` to define shapes
- Verify input dimensions match expected format

## Best Practices for TensorFlow 2.x

### 1. Use Keras API

```python
# Good
model = tf.keras.Model(...)
layer = tf.keras.layers.Dense(128)

# Avoid
# TF 1.x style low-level operations
```

### 2. Leverage High-Level Training

```python
# Good
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(train_data, epochs=10)

# Acceptable for custom training
with tf.GradientTape() as tape:
    # custom training loop
```

### 3. Use Built-in Callbacks

```python
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('model.ckpt'),
    tf.keras.callbacks.TensorBoard(log_dir='logs'),
    tf.keras.callbacks.EarlyStopping(patience=10)
]

model.fit(train_data, epochs=100, callbacks=callbacks)
```

### 4. Enable Mixed Precision (Optional)

```python
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

### 5. Use @tf.function for Performance

```python
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

## Resources

### Official Documentation

- [TensorFlow 2.x Migration Guide](https://www.tensorflow.org/guide/migrate)
- [Keras API Documentation](https://keras.io/api/)
- [TensorFlow 2.x Tutorials](https://www.tensorflow.org/tutorials)

### Key Concepts

- [Eager Execution](https://www.tensorflow.org/guide/eager)
- [Keras Functional API](https://www.tensorflow.org/guide/keras/functional)
- [Training and Evaluation](https://www.tensorflow.org/guide/keras/train_and_evaluate)
- [Save and Load Models](https://www.tensorflow.org/guide/keras/save_and_serialize)

## Version Compatibility

### Tested Versions

- TensorFlow: 2.10, 2.11, 2.12+
- Python: 3.7, 3.8, 3.9, 3.10
- NumPy: 1.21+
- scikit-learn: 1.0+

### Installation

```bash
# Recommended installation
pip install tensorflow>=2.10.0
pip install numpy>=1.21.0
pip install scikit-learn>=1.0.0
pip install matplotlib>=3.3.0
```

## Summary

### Key Takeaways

1. **Modern Keras API**: Uses `tf.keras` for all layers and models
2. **Eager Execution**: No more sessions or placeholders
3. **Model Subclassing**: `tf.keras.Model` for custom architectures
4. **Simplified Training**: `compile()` and `fit()` methods
5. **Better Debugging**: Python-native debugging support
6. **Performance**: Comparable to TF 1.x with better usability

### Migration Checklist

- [x] Replaced `tf.layers` with `tf.keras.layers`
- [x] Removed sessions and placeholders
- [x] Updated to `tf.keras.Model` subclassing
- [x] Modified image processing to use eager execution
- [x] Updated checkpoint saving/loading
- [x] Ensured batch normalization has training flag
- [x] Updated dropout to use rate (not keep_prob)
- [x] Tested with TensorFlow 2.10+
- [x] Verified GPU compatibility
- [x] Updated documentation

The migration is complete and the code is fully compatible with TensorFlow 2.x!
