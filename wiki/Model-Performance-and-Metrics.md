# Model Performance and Metrics

## Overview

Understanding and evaluating model performance is crucial for assessing the quality of your trained network. This guide explains all metrics, visualizations, and evaluation techniques available in the ConvNet implementation.

## Performance Metrics

### 1. Accuracy

**Definition**: Percentage of correctly classified images

**Formula**: `Accuracy = (Correct Predictions) / (Total Predictions)`

**Range**: 0% to 100%

**Interpretation:**
- **25%**: Random guessing (4 classes)
- **50%**: Poor performance
- **70-80%**: Good performance
- **85-95%**: Excellent performance
- **>95%**: Outstanding (check for overfitting)

**Location in output:**
- Training logs during training
- Classification report after prediction

### 2. Precision

**Definition**: Of all images predicted as class X, what percentage were actually class X?

**Formula**: `Precision = True Positives / (True Positives + False Positives)`

**Range**: 0.0 to 1.0

**Interpretation:**
- **High precision**: Few false positives, reliable when predicting this class
- **Low precision**: Many false positives, unreliable predictions

**Example:**
```
Precision for Cat = 0.85
→ 85% of images classified as "Cat" are actually cats
→ 15% are other classes misclassified as cats
```

### 3. Recall (Sensitivity)

**Definition**: Of all actual class X images, what percentage were correctly identified?

**Formula**: `Recall = True Positives / (True Positives + False Negatives)`

**Range**: 0.0 to 1.0

**Interpretation:**
- **High recall**: Few false negatives, catches most instances of this class
- **Low recall**: Many false negatives, misses many instances

**Example:**
```
Recall for Cat = 0.90
→ 90% of cat images were correctly identified as cats
→ 10% of cats were misclassified as other animals
```

### 4. F1-Score

**Definition**: Harmonic mean of precision and recall

**Formula**: `F1 = 2 × (Precision × Recall) / (Precision + Recall)`

**Range**: 0.0 to 1.0

**Interpretation:**
- **High F1**: Good balance between precision and recall
- **Low F1**: Poor performance in either precision or recall

**Use case**: Single metric to evaluate overall class performance

### 5. Support

**Definition**: Number of actual occurrences of each class in the test set

**Use**: Shows dataset balance

**Example:**
```
Support for Cat = 250
→ Test set contains 250 cat images
```

## Classification Report

After running predictions, you'll see a classification report:

```
              precision    recall  f1-score   support

     class 0       0.88      0.92      0.90       250
     class 1       0.85      0.87      0.86       245
     class 2       0.91      0.88      0.89       253
     class 3       0.89      0.86      0.87       252

    accuracy                           0.88      1000
   macro avg       0.88      0.88      0.88      1000
weighted avg       0.88      0.88      0.88      1000
```

**Reading the report:**
- **class 0-3**: Cat, Tree, Horse, Dog respectively
- **precision**: How reliable predictions are for each class
- **recall**: How well the model finds each class
- **f1-score**: Overall performance for each class
- **support**: Number of test images per class
- **accuracy**: Overall accuracy across all classes
- **macro avg**: Simple average across classes
- **weighted avg**: Average weighted by class frequency

## Loss Metrics

### Categorical Cross-Entropy Loss

**Definition**: Measures difference between predicted and true probability distributions

**Formula**: `Loss = -Σ(y_true × log(y_pred))`

**Range**: 0 to ∞ (lower is better)

**Interpretation:**
- **0**: Perfect predictions (impossible in practice)
- **0-0.5**: Excellent
- **0.5-1.0**: Good
- **1.0-2.0**: Moderate
- **>2.0**: Poor

**Behavior during training:**
- Should decrease over epochs
- May fluctuate batch-to-batch
- Validation loss should track training loss

## ROC Curve

### What is ROC?

ROC (Receiver Operating Characteristic) curve visualizes the trade-off between:
- **True Positive Rate (Recall)**: How many positives we catch
- **False Positive Rate**: How many negatives we misclassify

### Interpretation

**AUC (Area Under Curve):**
- **0.5**: Random guessing (worst)
- **0.7-0.8**: Fair
- **0.8-0.9**: Good
- **0.9-1.0**: Excellent
- **1.0**: Perfect (suspect overfitting)

### Viewing ROC Curve

The ROC curve is automatically displayed after training completes.

**Multi-class ROC:**
- One curve per class
- Shows how well each class is distinguished from others

## Confusion Matrix

While not explicitly shown, you can compute it from predictions:

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
```

**Example confusion matrix:**
```
          Predicted
          Cat Tree Horse Dog
Actual Cat  230   5    10   5
      Tree   3  215     2  25
     Horse   8   4   225  16
       Dog   4  18    15 215
```

**Reading the matrix:**
- Diagonal: Correct predictions
- Off-diagonal: Misclassifications
- Row i, column j: Class i images predicted as class j

**Insights:**
- Check which classes are confused with each other
- Identify systematic errors
- Guide data collection or model improvements

## TensorBoard Metrics

### Launching TensorBoard

```bash
tensorboard --logdir=ckpt_dir/
```

Access at: http://localhost:6006

### Available Visualizations

1. **Scalars**
   - Training loss over time
   - Training accuracy over time
   - Validation metrics (if logged)

2. **Graphs**
   - Network architecture visualization
   - Computational graph

3. **Distributions**
   - Weight distributions per layer
   - Activation distributions

4. **Histograms**
   - Weight histograms over time
   - Gradient histograms

## Interpreting Training Curves

### Healthy Training

**Loss curve:**
- Steady decrease
- May have small fluctuations
- Eventually plateaus

**Accuracy curve:**
- Steady increase
- Approaches asymptote
- Stabilizes at high value

### Overfitting

**Signs:**
- Training loss much lower than validation loss
- Training accuracy much higher than validation accuracy
- Validation metrics plateau or degrade while training improves

**Solutions:**
- Increase dropout
- Add data augmentation
- Use more training data
- Reduce model complexity
- Stop training earlier

### Underfitting

**Signs:**
- Both training and validation metrics poor
- Loss plateaus at high value
- Model doesn't improve with more training

**Solutions:**
- Train longer
- Increase model capacity
- Reduce regularization
- Check data quality
- Increase learning rate

### Unstable Training

**Signs:**
- Loss spikes or oscillates
- Metrics fluctuate wildly
- NaN or infinite values

**Solutions:**
- Reduce learning rate
- Reduce batch size
- Check for data issues
- Use gradient clipping
- Normalize inputs

## Evaluation Best Practices

### 1. Train/Validation/Test Split

**Recommended split:**
- Training: 70-80%
- Validation: 10-15% (for hyperparameter tuning)
- Test: 10-20% (for final evaluation)

**Important:**
- Never train on validation or test data
- Never tune hyperparameters on test data
- Test set should only be used once, at the end

### 2. Cross-Validation (Optional)

For small datasets, use k-fold cross-validation:

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True)
for train_idx, val_idx in kfold.split(X):
    # Train on train_idx
    # Validate on val_idx
    # Average results across folds
```

### 3. Class Balance

Check class distribution in train/test sets:

```python
import numpy as np
from collections import Counter

# Count classes in training data
class_counts = Counter(y_train)
print("Training distribution:", class_counts)

# Ensure test set has similar distribution
```

### 4. Stratified Sampling

Ensure each split has similar class distribution:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)
```

## Performance Benchmarking

### Baseline Comparisons

**Random guessing:**
- Accuracy: 25% (1/4 classes)
- Baseline to beat

**Simple models:**
- Logistic regression: 40-60%
- Decision tree: 50-70%

**Your model:**
- AlexNet: Should achieve 70-95%

### Performance Expectations

| Dataset Size | Expected Accuracy | Training Time (GPU) |
|--------------|-------------------|---------------------|
| Small (<1K)  | 60-75%            | 10-20 min           |
| Medium (1-10K) | 75-85%          | 30-60 min           |
| Large (>10K) | 85-95%            | 1-3 hours           |

**Note:** Actual performance depends on:
- Data quality
- Class separability
- Image diversity
- Proper hyperparameter tuning

## Error Analysis

### Systematic Error Analysis

1. **Collect misclassified examples**
   ```python
   misclassified_idx = np.where(y_pred != y_true)[0]
   ```

2. **Analyze patterns**
   - Which classes are most confused?
   - Are errors random or systematic?
   - Common visual characteristics?

3. **Visualize mistakes**
   - Display misclassified images
   - Compare with correctly classified
   - Look for data quality issues

4. **Take action**
   - Collect more data for problematic classes
   - Fix mislabeled training data
   - Add specific data augmentation
   - Adjust class weights

### Common Error Patterns

**Confusion between similar classes:**
- Dog vs. Cat: Expected, visually similar
- Tree vs. background: Need better data

**Asymmetric errors:**
- Class A → Class B often, but not reverse
- Indicates class imbalance or poor representation

**Low-confidence predictions:**
- Probabilities close to 25% (random)
- Image quality issues or out-of-distribution

## Model Confidence

### Prediction Probabilities

```python
predictions = model.predict(images)
# predictions[i] = [0.8, 0.1, 0.05, 0.05]
#                   ^    ^    ^     ^
#                  Cat Tree Horse  Dog
```

**Interpreting probabilities:**
- **>0.9**: Very confident
- **0.7-0.9**: Confident
- **0.5-0.7**: Moderate confidence
- **<0.5**: Low confidence (below random for 4 classes)

### Confidence Calibration

**Well-calibrated model:**
- Predicted probability matches actual accuracy
- 80% confidence → 80% correct

**Poorly-calibrated model:**
- Overconfident: High probability, often wrong
- Underconfident: Low probability, often correct

## Production Metrics

### Inference Speed

**Measure prediction time:**
```python
import time

start = time.time()
predictions = model.predict(test_images)
elapsed = time.time() - start

print(f"Throughput: {len(test_images) / elapsed} images/sec")
```

**Expected performance:**
- CPU: 10-50 images/sec
- GPU: 100-1000+ images/sec

### Model Size

**Checkpoint size:**
- Typical: 50-200 MB
- Consider compression for deployment

### Memory Usage

**GPU memory:**
- Training: 2-4 GB
- Inference: 500 MB - 2 GB

## Reporting Results

### Academic/Research Format

```
Model: AlexNet (TensorFlow 2.x)
Dataset: 4-class image classification (Cat/Tree/Horse/Dog)
Training samples: 3200
Test samples: 800

Results:
- Overall Accuracy: 88.5% ± 1.2%
- Macro-averaged F1: 0.88
- Training time: 45 minutes (NVIDIA RTX 3080)

Per-class Performance:
- Cat: P=0.88, R=0.92, F1=0.90
- Tree: P=0.85, R=0.87, F1=0.86
- Horse: P=0.91, R=0.88, F1=0.89
- Dog: P=0.89, R=0.86, F1=0.87
```

### Business Format

```
Model Performance Summary
- Overall accuracy: 88.5%
- Correctly classifies 8-9 out of 10 images
- Processing speed: 200 images/second
- Model ready for deployment

Recommendations:
- Deploy with confidence threshold of 0.7
- Review low-confidence predictions manually
- Expected error rate: 11.5% (1 in 9 images)
```

## Next Steps After Evaluation

1. **If performance is good:**
   - Save final model
   - Document configuration
   - Prepare for deployment

2. **If performance is poor:**
   - Analyze errors (see Error Analysis)
   - Try hyperparameter tuning
   - Collect more/better data
   - Consider data augmentation

3. **If overfitting:**
   - Increase regularization
   - Add dropout
   - Use data augmentation
   - Collect more training data

4. **If underfitting:**
   - Train longer
   - Increase model capacity
   - Reduce regularization
   - Check data quality

## Summary Checklist

- [ ] Run predictions on test set
- [ ] Review classification report
- [ ] Check precision/recall per class
- [ ] Analyze ROC curve
- [ ] Identify confused classes
- [ ] Perform error analysis
- [ ] Monitor confidence calibration
- [ ] Measure inference speed
- [ ] Compare to baseline
- [ ] Document results
- [ ] Decide on next steps
