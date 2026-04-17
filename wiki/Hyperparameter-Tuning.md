# Hyperparameter Tuning

## Overview

Hyperparameter tuning is the process of finding the optimal configuration for your neural network to achieve the best performance. This guide covers all tunable parameters in the ConvNet AlexNet implementation.

## Key Hyperparameters

### 1. Learning Rate

**Location**: Command-line argument or code

**Default**: 0.001

**Range**: 0.00001 to 0.1

**Impact**:
- **Too high**: Training unstable, loss oscillates or diverges
- **Too low**: Training very slow, may get stuck in local minima
- **Just right**: Steady decrease in loss, good convergence

**Tuning strategy:**
```bash
# Start with default
python my_alexnet_cnn.py train --learning-rate 0.001

# If unstable, try lower
python my_alexnet_cnn.py train --learning-rate 0.0001

# If too slow, try higher
python my_alexnet_cnn.py train --learning-rate 0.01
```

**Recommended values:**
- Small dataset: 0.001 - 0.01
- Large dataset: 0.0001 - 0.001
- Fine-tuning: 0.00001 - 0.0001

### 2. Batch Size

**Location**: `my_alexnet_cnn.py:22`

**Default**: 64

**Range**: 8 to 256 (depends on GPU memory)

**Impact**:
- **Larger batches**: Faster training, more stable gradients, needs more memory
- **Smaller batches**: Better generalization, less memory, noisier gradients

**Tuning strategy:**
```python
# In my_alexnet_cnn.py
BATCH_SIZE = 32  # Reduce if memory errors
BATCH_SIZE = 128 # Increase for faster training
```

**Recommended values:**
- CPU training: 8 - 32
- GPU with 4 GB VRAM: 32 - 64
- GPU with 8+ GB VRAM: 64 - 128

### 3. Number of Epochs

**Location**: Command-line argument

**Default**: 100

**Range**: 10 to 500+

**Impact**:
- **Too few**: Underfitting, model hasn't learned enough
- **Too many**: Overfitting, model memorizes training data

**Tuning strategy:**
- Start with 50-100 epochs
- Monitor validation accuracy
- Stop when validation accuracy plateaus
- Use early stopping if available

**Recommended values:**
- Quick test: 10 - 20
- Small dataset: 50 - 100
- Large dataset: 100 - 200

### 4. Dropout Rates

**Location**: `my_alexnet_cnn.py:28-29`

**Defaults**:
- Input dropout: 0.8 (keep 80%)
- Hidden dropout: 0.5 (keep 50%)

**Range**: 0.1 to 0.9 (keep rate)

**Impact**:
- **Higher dropout**: Less overfitting, may underfit if too high
- **Lower dropout**: Better training accuracy, may overfit

**Tuning strategy:**
```python
# In my_alexnet_cnn.py
input_dropout = 0.9   # Less regularization
hidden_dropout = 0.6  # Less regularization

# Or
input_dropout = 0.7   # More regularization
hidden_dropout = 0.4  # More regularization
```

**Recommended values:**
- Small dataset: 0.3 - 0.5 (keep rate)
- Large dataset: 0.5 - 0.8 (keep rate)
- Overfitting: Decrease keep rate
- Underfitting: Increase keep rate

### 5. Optimizer Parameters (Adam)

**Location**: Code where optimizer is created

**Default epsilon**: 0.1 (unusually high)

**Standard Adam parameters:**
- Learning rate: 0.001
- Beta1: 0.9 (momentum)
- Beta2: 0.999 (RMSprop)
- Epsilon: 1e-7 (numerical stability)

**Tuning strategy:**
```python
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)
```

### 6. Weight Initialization

**Location**: `my_alexnet_cnn.py:30`

**Default**: 0.1 (standard deviation)

**Alternatives:**
- **He initialization** (for ReLU): `math.sqrt(2/n_input)`
- **Xavier initialization**: `math.sqrt(1/n_input)`
- **Fixed small value**: 0.01 - 0.1

**Tuning strategy:**
```python
# Current
std_dev = 0.1

# He initialization (better for ReLU)
std_dev = math.sqrt(2/n_input)

# Smaller initialization
std_dev = 0.01
```

## Hyperparameter Tuning Workflow

### Step 1: Baseline

Start with default parameters:
```bash
python my_alexnet_cnn.py train
```

Record baseline performance:
- Training accuracy: ____%
- Validation accuracy: ____%
- Training time: ____

### Step 2: Learning Rate Tuning

Test different learning rates:
```bash
python my_alexnet_cnn.py train --learning-rate 0.0001
python my_alexnet_cnn.py train --learning-rate 0.001
python my_alexnet_cnn.py train --learning-rate 0.01
```

Choose the learning rate that:
- Converges fastest
- Achieves best validation accuracy
- Has stable training (no oscillations)

### Step 3: Batch Size Tuning

Try different batch sizes (modify code):
```python
BATCH_SIZE = 32
BATCH_SIZE = 64
BATCH_SIZE = 128
```

Consider:
- Memory constraints
- Training speed
- Validation accuracy

### Step 4: Regularization Tuning

If overfitting (high train, low validation accuracy):
- Increase dropout (decrease keep rate)
- Add data augmentation
- Use more training data

If underfitting (low train and validation accuracy):
- Decrease dropout (increase keep rate)
- Train longer
- Increase model capacity

### Step 5: Fine-Tuning

Once you have good parameters:
- Train for more epochs
- Use lower learning rate (0.1× current)
- Monitor validation closely

## Grid Search Strategy

### Simple Grid Search

Test combinations systematically:

| Learning Rate | Batch Size | Dropout (hidden) | Val Accuracy |
|---------------|------------|------------------|--------------|
| 0.001         | 64         | 0.5              | ?            |
| 0.001         | 64         | 0.6              | ?            |
| 0.001         | 32         | 0.5              | ?            |
| 0.0001        | 64         | 0.5              | ?            |
| ...           | ...        | ...              | ...          |

### Focused Grid Search

After finding promising region, search nearby:

If 0.001/64/0.5 works well, try:
- Learning rates: 0.0005, 0.001, 0.002
- Batch sizes: 48, 64, 80
- Dropout: 0.4, 0.5, 0.6

## Random Search Strategy

Randomly sample hyperparameter space:

```python
import random

learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
batch_sizes = [16, 32, 64, 128]
dropout_rates = [0.3, 0.4, 0.5, 0.6, 0.7]

for trial in range(10):
    lr = random.choice(learning_rates)
    bs = random.choice(batch_sizes)
    dr = random.choice(dropout_rates)

    # Update code and train
    # Record results
```

## Advanced Tuning Techniques

### Learning Rate Scheduling

**Step Decay:**
```python
# Reduce learning rate by 10× every 30 epochs
initial_lr = 0.001
epoch = current_epoch
lr = initial_lr * (0.1 ** (epoch // 30))
```

**Exponential Decay:**
```python
# Gradually reduce learning rate
initial_lr = 0.001
decay_rate = 0.95
epoch = current_epoch
lr = initial_lr * (decay_rate ** epoch)
```

**Cosine Annealing:**
```python
# Cyclical learning rate
import math
initial_lr = 0.001
epoch = current_epoch
max_epochs = 100
lr = initial_lr * 0.5 * (1 + math.cos(math.pi * epoch / max_epochs))
```

### Adaptive Dropout

Adjust dropout during training:
```python
# Start with high dropout, reduce as training progresses
epoch = current_epoch
max_epochs = 100
dropout_rate = 0.5 + 0.3 * (epoch / max_epochs)
```

### Warmup Strategy

Start with low learning rate, increase gradually:
```python
# First 5 epochs: warmup
warmup_epochs = 5
target_lr = 0.001

if epoch < warmup_epochs:
    lr = target_lr * (epoch / warmup_epochs)
else:
    lr = target_lr
```

## Monitoring Metrics

### Key Metrics to Track

1. **Training Loss**: Should decrease steadily
2. **Training Accuracy**: Should increase steadily
3. **Validation Loss**: Should decrease (may fluctuate)
4. **Validation Accuracy**: Best metric for tuning
5. **Training Time**: Consider efficiency

### Identifying Issues

**Overfitting Signs:**
- Training accuracy much higher than validation
- Validation accuracy plateaus or decreases
- Training loss continues to decrease

**Underfitting Signs:**
- Both training and validation accuracy low
- Loss plateaus at high value
- Model doesn't improve with more training

**Unstable Training Signs:**
- Loss oscillates or spikes
- Accuracy fluctuates wildly
- Gradients explode (NaN loss)

## Hyperparameter Importance Ranking

### Most Important (Tune First)
1. **Learning rate**: Biggest impact on convergence
2. **Batch size**: Affects speed and generalization
3. **Number of epochs**: Needs to be sufficient

### Moderately Important (Tune Second)
4. **Dropout rates**: Important for regularization
5. **Optimizer choice**: Adam usually good default

### Less Important (Tune Last)
6. **Weight initialization**: Good defaults usually work
7. **Epsilon**: Rarely needs tuning
8. **Activation functions**: ReLU is standard

## Practical Tuning Examples

### Example 1: Small Dataset (<1000 images)

**Problem**: Overfitting

**Solution:**
```bash
# Increase regularization
# In code: hidden_dropout = 0.3 (keep only 30%)
# Use lower learning rate for stability
python my_alexnet_cnn.py train \
  --learning-rate 0.0001 \
  --max_epochs 50
```

### Example 2: Large Dataset (>10000 images)

**Problem**: Slow training

**Solution:**
```bash
# In code: BATCH_SIZE = 128
# Use higher learning rate
python my_alexnet_cnn.py train \
  --learning-rate 0.01 \
  --max_epochs 100
```

### Example 3: Unstable Training

**Problem**: Loss oscillates

**Solution:**
```bash
# Reduce learning rate significantly
python my_alexnet_cnn.py train \
  --learning-rate 0.0001 \
  --max_epochs 100

# Also consider reducing batch size in code
# BATCH_SIZE = 32
```

### Example 4: Poor Convergence

**Problem**: Accuracy plateaus at 40%

**Solutions to try:**
1. Check data quality and labels
2. Increase model capacity (more layers/neurons)
3. Train much longer
4. Reduce regularization
5. Try different learning rate

## Automated Hyperparameter Tuning

### Using Optuna (Optional Enhancement)

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    dropout = trial.suggest_uniform('dropout', 0.3, 0.7)

    # Train model with these parameters
    # Return validation accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print('Best parameters:', study.best_params)
```

## Documentation Template

Keep track of experiments:

```
Experiment #1
Date: 2024-XX-XX
Parameters:
  - Learning rate: 0.001
  - Batch size: 64
  - Epochs: 100
  - Dropout (input): 0.8
  - Dropout (hidden): 0.5
Results:
  - Training accuracy: 92%
  - Validation accuracy: 85%
  - Training time: 45 min
Notes:
  - Good baseline performance
  - Some overfitting observed
```

## Best Practices

1. **Change one thing at a time**: Easier to understand impact
2. **Keep detailed records**: Track all experiments
3. **Use validation set**: Never tune on test set
4. **Be patient**: Good tuning takes time
5. **Start simple**: Default parameters first
6. **Monitor trends**: Look for patterns across experiments
7. **Consider resources**: Balance accuracy vs. training time
8. **Reproducibility**: Set random seeds

## Common Pitfalls

1. **Overfitting to validation set**: Don't tune excessively
2. **Ignoring training time**: Sometimes good enough is enough
3. **Not enough trials**: Need sufficient exploration
4. **Forgetting to shuffle**: Always shuffle training data
5. **Using test set for tuning**: Leads to overoptimistic results

## Summary Checklist

- [ ] Start with baseline (default parameters)
- [ ] Tune learning rate first
- [ ] Adjust batch size based on resources
- [ ] Monitor for overfitting/underfitting
- [ ] Adjust regularization (dropout) accordingly
- [ ] Train for sufficient epochs
- [ ] Document all experiments
- [ ] Select best configuration based on validation
- [ ] Final evaluation on test set only
