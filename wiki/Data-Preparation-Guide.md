# Data Preparation Guide

## Overview

Proper data preparation is crucial for training an effective image classification model. This guide covers all aspects of preparing your dataset for the ConvNet AlexNet implementation.

## Dataset Organization

### Directory Structure

Organize your images in the following structure:

```
ConvNet/
├── dataset/              # Training images
│   ├── Cat/
│   │   ├── cat001.jpg
│   │   ├── cat002.jpg
│   │   └── ...
│   ├── Tree/
│   │   ├── tree001.jpg
│   │   ├── tree002.jpg
│   │   └── ...
│   ├── Horse/
│   │   ├── horse001.jpg
│   │   └── ...
│   └── Dog/
│       ├── dog001.jpg
│       └── ...
└── test_dataset/         # Test images (same structure)
    ├── Cat/
    ├── Tree/
    ├── Horse/
    └── Dog/
```

### Important Notes

1. **Folder names** must match the class names: `Cat`, `Tree`, `Horse`, `Dog`
2. **Case-sensitive**: Use exact capitalization as shown
3. **One class per folder**: Each image should be in its corresponding class folder
4. **Separate train/test**: Keep training and test datasets in different directories

## Image Requirements

### File Format
- **Supported formats**: JPEG only (`.jpg` or `.jpeg`)
- **Color space**: RGB (3 channels)
- **Not supported**: PNG, GIF, BMP, grayscale images

### Image Dimensions
- **Target size**: 224×224 pixels
- **Actual size**: Any size (will be automatically resized)
- **Aspect ratio**: Any (will be cropped or padded)

### Image Quality
- **Minimum resolution**: 100×100 pixels (recommended)
- **Maximum resolution**: No strict limit (will be resized)
- **File size**: No strict limit, but smaller is faster to process
- **Compression**: Any JPEG compression level

## Preprocessing Commands

### 1. Preprocessing Training Data

Convert training images to compressed pickle format:

```bash
python my_alexnet_cnn.py preprocessing_training -f images_dataset.pkl
```

**Parameters:**
- `-f`, `--file`: Output filename for preprocessed data (default: `images_dataset.pkl`)
- `-s`, `--shuffle`: Shuffle the dataset after preprocessing (recommended)

**What happens:**
1. Scans all subdirectories in `dataset/`
2. Reads each JPEG image
3. Converts to float32 format
4. Resizes to 224×224 (crop or pad)
5. Converts labels to one-hot encoding
6. Saves to compressed pickle file

### 2. Shuffling Training Data

**Always shuffle your training data** for better convergence:

```bash
python my_alexnet_cnn.py preprocessing_training -f images_dataset.pkl --shuffle
```

**Output:** Creates `images_shuffled.pkl` with randomly shuffled data

**Why shuffle?**
- Prevents learning order-dependent patterns
- Improves gradient descent convergence
- Reduces overfitting
- Standard practice in deep learning

### 3. Preprocessing Test Data

Convert test images to compressed pickle format:

```bash
python my_alexnet_cnn.py preprocessing_test -t images_test_dataset.pkl
```

**Parameters:**
- `-t`, `--test`: Output filename for preprocessed test data (default: `images_test_dataset.pkl`)

**Note:** Test data should NOT be shuffled to maintain consistent evaluation

## Data Pipeline

### Step-by-Step Process

1. **Image Loading**
   ```python
   img_bytes = tf.io.read_file(img_path)
   img_u8 = tf.image.decode_jpeg(img_bytes, channels=3)
   ```

2. **Type Conversion**
   ```python
   image = tf.image.convert_image_dtype(img_u8, tf.float32)
   ```
   - Converts from uint8 [0, 255] to float32 [0, 1]
   - Normalization happens automatically

3. **Resizing**
   ```python
   img_resized = tf.image.resize_with_crop_or_pad(image, 224, 224)
   ```
   - If image is larger: crops from center
   - If image is smaller: pads with zeros
   - Maintains aspect ratio in center

4. **Reshaping**
   ```python
   img_flattened = tf.reshape(img_resized, shape=[224 * 224, 3])
   ```
   - Flattens spatial dimensions
   - Keeps channel dimension separate

5. **Label Encoding**
   ```python
   label = np.eye(num_labels)[class_id]
   ```
   - Converts class ID to one-hot vector
   - Example: Cat (0) → [1, 0, 0, 0]

6. **Compression**
   ```python
   with gzip.open(file_path, 'wb') as file:
       pickle.dump((img, label), file)
   ```
   - Uses gzip compression
   - Reduces file size by ~90%
   - Faster loading during training

## Class Mapping

The preprocessing automatically maps folder names to class IDs:

| Folder Name | Class ID | One-Hot Vector |
|-------------|----------|----------------|
| Cat         | 0        | [1, 0, 0, 0]   |
| Tree        | 1        | [0, 1, 0, 0]   |
| Horse       | 2        | [0, 0, 1, 0]   |
| Dog         | 3        | [0, 0, 0, 1]   |

**Important:** Class mapping is determined by the order in `Dataset.py:LABELS_DICT`

## Dataset Size Recommendations

### Minimum Dataset Size
- **Per class**: 100 images minimum
- **Total**: 400 images (100 per class)
- **Note**: May not achieve good accuracy with minimum size

### Recommended Dataset Size
- **Per class**: 500-1000 images
- **Total**: 2000-4000 images
- **Balance**: Similar number of images per class

### Large Dataset
- **Per class**: 5000+ images
- **Total**: 20000+ images
- **Best results**: More data generally improves performance

### Train/Test Split
- **Training**: 80-90% of total data
- **Testing**: 10-20% of total data
- **Example**: 800 train + 200 test per class

## Data Augmentation Considerations

### Currently Implemented
- **Crop/Pad**: Automatic resizing to 224×224

### Not Currently Implemented (Future Enhancement)
- Random horizontal flips
- Random rotations (±15 degrees)
- Random brightness/contrast adjustment
- Random crops (instead of center crop)
- Color jittering

### How to Add Augmentation

Modify `Dataset.py:convertDataset()` to add augmentation:

```python
# Random flip example
if random.random() > 0.5:
    image = tf.image.flip_left_right(image)

# Random brightness example
image = tf.image.random_brightness(image, max_delta=0.2)
```

## Data Quality Checklist

Before preprocessing, ensure:

- [ ] All images are JPEG format
- [ ] Images are in correct class folders
- [ ] No corrupted images
- [ ] No mislabeled images
- [ ] Reasonable class balance
- [ ] Train and test sets are separate
- [ ] Test set is representative of real-world data
- [ ] Sufficient number of images per class

## Common Data Issues

### Issue 1: Corrupted Images
**Symptoms**: Preprocessing crashes with decode error

**Solutions:**
- Remove corrupted images
- Re-download or re-export images
- Use image validation script to find corrupted files

**Validation script:**
```python
import os
from PIL import Image

for root, dirs, files in os.walk('dataset/'):
    for file in files:
        if file.endswith(('.jpg', '.jpeg')):
            try:
                img = Image.open(os.path.join(root, file))
                img.verify()
            except:
                print(f"Corrupted: {os.path.join(root, file)}")
```

### Issue 2: Imbalanced Classes
**Symptoms**: Model performs poorly on minority classes

**Solutions:**
- Collect more data for minority classes
- Use data augmentation on minority classes
- Apply class weights during training (requires code modification)
- Undersample majority classes

### Issue 3: Low Image Quality
**Symptoms**: Poor model performance despite sufficient data

**Solutions:**
- Use higher resolution images
- Ensure images are clear and well-lit
- Remove blurry or low-quality images
- Use better data sources

### Issue 4: Wrong File Format
**Symptoms**: Images not loaded during preprocessing

**Solutions:**
- Convert PNG/other formats to JPEG
- Ensure file extensions are `.jpg` or `.jpeg`
- Check for hidden extensions (e.g., `.jpg.png`)

**Batch conversion (ImageMagick):**
```bash
mogrify -format jpg *.png
```

## Preprocessing Performance

### Processing Time
- **Small dataset** (<1000 images): 1-5 minutes
- **Medium dataset** (1000-10000 images): 5-30 minutes
- **Large dataset** (>10000 images): 30+ minutes

### Output File Sizes
- **Uncompressed**: ~50-100 MB per 1000 images
- **Compressed (gzip)**: ~5-10 MB per 1000 images
- **Compression ratio**: ~90% size reduction

### Memory Usage
- **Peak memory**: ~2-4 GB for large datasets
- **Recommendation**: 8 GB+ RAM for large datasets
- **Disk space**: Ensure 2-3× dataset size available

## Advanced Preprocessing

### Custom Dataset Path

Modify `my_alexnet_cnn.py` to use custom paths:

```python
TRAIN_IMAGE_DIR = '/path/to/your/dataset'
TEST_IMAGE_DIR = '/path/to/your/test_dataset'
```

### Adding New Classes

To support more than 4 classes:

1. Update `Dataset.py:LABELS_DICT`:
```python
LABELS_DICT = {
    'Cat': 0,
    'Tree': 1,
    'Horse': 2,
    'Dog': 3,
    'NewClass': 4,  # Add new class
}
```

2. Update `my_alexnet_cnn.py:n_classes`:
```python
n_classes = 5  # Update class count
```

3. Create corresponding folder in `dataset/` and `test_dataset/`

### Grayscale Images

Current implementation requires RGB. To support grayscale:

```python
# In Dataset.py, change:
img_u8 = tf.image.decode_jpeg(img_bytes, channels=3)

# To:
img_u8 = tf.image.decode_jpeg(img_bytes, channels=1)
img_u8 = tf.image.grayscale_to_rgb(img_u8)
```

## Best Practices

1. **Keep original images**: Don't delete raw images after preprocessing
2. **Version your datasets**: Use different filenames for different versions
3. **Document preprocessing**: Note any special steps taken
4. **Validate preprocessed data**: Load and inspect a few samples
5. **Separate test set**: Never train on test data
6. **Consistent preprocessing**: Use same process for train and test
7. **Backup preprocessed files**: Avoid reprocessing if possible

## Complete Preprocessing Workflow

```bash
# Step 1: Organize images in correct directory structure
# (Manual step)

# Step 2: Validate image quality
# (Optional: Run validation script)

# Step 3: Preprocess training data
python my_alexnet_cnn.py preprocessing_training -f images_dataset.pkl

# Step 4: Shuffle training data
python my_alexnet_cnn.py preprocessing_training -f images_dataset.pkl --shuffle

# Step 5: Preprocess test data
python my_alexnet_cnn.py preprocessing_test -t images_test_dataset.pkl

# Step 6: Verify preprocessed files exist
ls -lh *.pkl

# Step 7: Ready to train!
python my_alexnet_cnn.py train
```
