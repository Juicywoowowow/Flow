# Flow Optimization Guide

A practical guide to tuning neural networks with Flow.

---

## Optimizer Selection

### Quick Reference

| Use Case | Recommended Optimizer | Typical Config |
|----------|----------------------|----------------|
| General purpose | Adam | LR=0.001, β1=0.9, β2=0.999 |
| Large batch training | Lion | LR=0.0001, β1=0.9, β2=0.99 |
| Memory constrained | AdaFactor | Default settings |
| Fine-tuning | AdamW | LR=0.0001, WeightDecay=0.01 |
| Simple/stable | SGD + Momentum | LR=0.01, Momentum=0.9 |
| Convex problems | Adagrad | LR=0.01 |

### Adam vs Lion vs SGD

```
Training Speed:     Lion > Adam > SGD
Memory Usage:       SGD < Lion < Adam
Generalization:     SGD ≈ Lion > Adam
Hyperparameter Sensitivity: SGD > Adam > Lion
```

---

## Learning Rate

The most important hyperparameter. Too high = divergence. Too low = slow convergence.

### Finding the Right LR

**1. Start with standard values:**
```go
// Adam family
LR: 0.001  // Start here

// SGD
LR: 0.01   // Start here

// Lion
LR: 0.0001 // 3-10x smaller than Adam
```

**2. Learning Rate Range Test:**
Train for a few epochs with increasing LR. Plot loss vs LR. Choose LR where loss decreases fastest.

**3. Use LR Schedulers:**

```go
// Cosine annealing - smooth decay
flow.CosineAnnealing(flow.CosineAnnealingConfig{
    TMax:   100,    // Epoch cycle length
    EtaMin: 0.0001, // Minimum LR
    EtaMax: 0.01,   // Maximum LR
})

// Warmup - for large batch training
flow.Warmup(flow.WarmupConfig{
    WarmupEpochs: 5,
    TargetLR:     0.001,
    InitialLR:    0.0001,
})

// Step decay - classic approach
flow.StepDecay(flow.StepDecayConfig{
    StepSize: 30,  // Decay every 30 epochs
    Gamma:    0.1, // Multiply LR by 0.1
})
```

---

## Batch Size

Affects training stability and generalization.

### Guidelines

| Batch Size | Characteristics |
|------------|-----------------|
| 16-32 | Good generalization, noisy gradients |
| 64-128 | Balanced choice for most problems |
| 256-512 | Faster iteration, may need LR warmup |
| 1024+ | Requires careful tuning, use Lion optimizer |

### Gradient Accumulation

Train with larger effective batch sizes without memory increase:

```go
flow.TrainConfig{
    BatchSize:                 32,
    GradientAccumulationSteps: 4,  // Effective batch = 32 * 4 = 128
}
```

---

## Weight Initialization

Proper initialization prevents vanishing/exploding gradients.

### Recommendations

| Activation | Initializer | Gain |
|------------|-------------|------|
| ReLU, LeakyReLU | HeNormal | 1.0 |
| Tanh, Sigmoid | XavierNormal | 1.0 |
| GELU, Swish | HeNormal | 1.0 |
| Linear (output) | XavierUniform | 1.0 |

```go
// For ReLU networks
flow.Dense(128).
    WithActivation(flow.ReLU()).
    WithInitializer(flow.HeNormal(1.0)).
    WithBiasInitializer(flow.Zeros()).  // Bias always zeros
    WithBias(true)
```

---

## Regularization

Prevent overfitting.

### L2 Regularization (Weight Decay)

```go
// Via optimizer (decoupled - preferred)
flow.AdamW(flow.AdamWConfig{
    WeightDecay: 0.01,  // Typical: 0.01 - 0.1
})

// Via regularizer (classic)
CompileConfig{
    Regularizer: flow.L2(0.0001),
}
```

### Dropout

```go
flow.Dense(256).WithActivation(flow.ReLU())...
flow.Dropout(0.5).Build()  // 50% dropout after dense
flow.Dense(128)...
```

**Typical dropout rates:**
- After dense layers: 0.3 - 0.5
- After conv layers: 0.1 - 0.3
- Before output: 0.2

### Early Stopping

```go
flow.EarlyStopping(flow.EarlyStoppingConfig{
    Monitor:  "val_loss",
    Patience: 10,        // Wait 10 epochs for improvement
    MinDelta: 0.001,     // Minimum improvement to count
    Mode:     "min",
})
```

---

## Normalization

Stabilize and accelerate training.

### When to Use What

| Layer Type | Normalization |
|------------|---------------|
| Dense (feedforward) | BatchNorm or LayerNorm |
| Conv2D | BatchNorm |
| Transformer | LayerNorm or RMSNorm |
| RNN/LSTM | LayerNorm |
| Small batch | LayerNorm or GroupNorm |
| Large batch | BatchNorm |

```go
// After activation (modern approach)
flow.Dense(128).WithActivation(flow.ReLU())...
flow.LayerNorm(1e-6).Build()

// Before activation (classic BatchNorm)
flow.Dense(128).WithActivation(flow.Linear())...
flow.BatchNorm(1e-5, 0.1).Build()
// Then apply activation separately
```

---

## Gradient Clipping

Prevent exploding gradients.

```go
CompileConfig{
    GradientClip: flow.GradientClipConfig{
        Mode:    "norm",
        MaxNorm: 1.0,  // Clip if total norm > 1.0
    },
}
```

**When to use:**
- Always for RNNs/LSTMs
- When training is unstable
- With high learning rates
- For Transformers

---

## Common Configurations

### Classification (Dense)

```go
// 2-layer MLP for classification
Dense(256, ReLU, HeNormal)
Dropout(0.3)
Dense(128, ReLU, HeNormal)
Dropout(0.2)
Dense(num_classes, Softmax, XavierNormal)

// Optimizer
Adam(LR=0.001, WeightDecay=0.0001)

// Loss
CrossEntropy(LabelSmoothing=0.1)
```

### CNN for Images

```go
// LeNet-style CNN
Conv2D(32, 3x3, same, ReLU)
MaxPool2D(2x2)
Conv2D(64, 3x3, same, ReLU)
MaxPool2D(2x2)
Flatten
Dense(128, ReLU)
Dropout(0.5)
Dense(num_classes, Softmax)

// Optimizer  
Lion(LR=0.0001, WeightDecay=0.01)
// or
Adam(LR=0.001)
```

### Regression

```go
Dense(64, Tanh, XavierNormal)
Dense(32, ReLU, HeNormal)
Dense(1, Linear, XavierUniform)

// Optimizer
RMSprop(LR=0.01)

// Loss
MSE or Huber
```

---

## Debugging Checklist

When training doesn't converge:

1. **Loss = NaN?**
   - Reduce learning rate by 10x
   - Add gradient clipping
   - Check for division by zero in data

2. **Loss not decreasing?**
   - Increase learning rate
   - Check data normalization
   - Verify labels are correct

3. **Overfitting?**
   - Add dropout
   - Add weight decay
   - Get more data
   - Use early stopping

4. **Underfitting?**
   - Increase model capacity
   - Train longer
   - Reduce regularization

5. **Training unstable?**
   - Use batch normalization
   - Add gradient clipping
   - Reduce learning rate
   - Use warmup schedule
