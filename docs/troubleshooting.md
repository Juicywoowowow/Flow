# Flow Troubleshooting

Solutions to common problems when using Flow.

---

## Build Errors

### "DenseLayer requires activation - use WithActivation()"

Every Dense layer needs an explicit activation:

```go
// ❌ Wrong
flow.Dense(128).Build()

// ✅ Correct
flow.Dense(128).
    WithActivation(flow.ReLU()).
    WithInitializer(flow.HeNormal(1.0)).
    WithBiasInitializer(flow.Zeros()).
    WithBias(true).
    Build()
```

### "DenseLayer requires initializer - use WithInitializer()"

Weight initialization is mandatory:

```go
// ✅ Add an initializer
flow.Dense(128).
    WithActivation(flow.ReLU()).
    WithInitializer(flow.HeNormal(1.0)).  // Required!
    ...
```

### "network must be built before compiling"

Call `Build()` before `Compile()`:

```go
// Build first
net, err := flow.NewNetwork(config).
    AddLayer(...).
    Build([]int{inputDim})  // ← Must call this

// Then compile
err = net.Compile(compileConfig)
```

### "network must be compiled before training"

Call `Compile()` before `Train()`:

```go
err = net.Compile(flow.CompileConfig{
    Optimizer:   flow.Adam(...),
    Loss:        flow.CrossEntropy(...),
    Regularizer: flow.NoReg(),
    GradientClip: flow.GradientClipConfig{Mode: "none"},
})

// Now you can train
result, err := net.Train(inputs, targets, trainConfig, nil)
```

---

## Runtime Errors

### "panic: runtime error: index out of range"

Usually a shape mismatch. Check:

1. **Input shape matches network:**
```go
// If you built with []int{784}
net.Build([]int{784})

// Your input must have 784 features per sample
inputs := [][]float64{
    make([]float64, 784),  // ✅ Correct
    make([]float64, 100),  // ❌ Wrong!
}
```

2. **For Conv2D, input shape must be [H, W, C]:**
```go
net.Build([]int{28, 28, 1})  // 28x28 grayscale

// Flatten your images to H*W*C = 784 for the API
inputs[i] = flattenedImage  // len = 784
```

3. **Target shape matches output:**
```go
// If your last layer is Dense(10)
targets := [][]float64{
    make([]float64, 10),  // ✅ One-hot encoded
}
```

### "inputs and targets must have same length"

Number of samples must match:

```go
inputs := make([][]float64, 1000)   // 1000 samples
targets := make([][]float64, 1000)  // Must also be 1000
```

---

## Training Issues

### Loss is NaN

**Causes:**
1. Learning rate too high
2. Exploding gradients
3. Division by zero in data

**Solutions:**

```go
// 1. Reduce learning rate
flow.Adam(flow.AdamConfig{LR: 0.0001})  // 10x smaller

// 2. Add gradient clipping
GradientClip: flow.GradientClipConfig{
    Mode:    "norm",
    MaxNorm: 1.0,
}

// 3. Check your data
for _, row := range inputs {
    for _, val := range row {
        if math.IsNaN(val) || math.IsInf(val, 0) {
            panic("Bad data!")
        }
    }
}
```

### Loss Not Decreasing

**Causes:**
1. Learning rate too low
2. Wrong loss function
3. Data not normalized

**Solutions:**

```go
// 1. Increase learning rate
LR: 0.01  // Try 10x higher

// 2. Check loss function matches task
// Classification → CrossEntropy (with Softmax output)
// Regression → MSE or Huber (with Linear output)

// 3. Normalize inputs
mean := computeMean(inputs)
std := computeStd(inputs)
for i := range inputs {
    for j := range inputs[i] {
        inputs[i][j] = (inputs[i][j] - mean) / std
    }
}
```

### Training Too Slow

**Solutions:**

```go
// 1. Increase batch size
BatchSize: 64  // or 128, 256

// 2. Use faster optimizer
flow.Lion(...)  // Faster than Adam for large batches

// 3. Reduce model size (if overfitting anyway)
Dense(64)  // instead of Dense(256)
```

### Overfitting (Training loss << Validation loss)

**Solutions:**

```go
// 1. Add dropout
flow.Dropout(0.5).Build()

// 2. Add weight decay
flow.AdamW(flow.AdamWConfig{WeightDecay: 0.01})

// 3. Use early stopping
flow.EarlyStopping(flow.EarlyStoppingConfig{
    Monitor:  "val_loss",
    Patience: 10,
})

// 4. Add data augmentation (implement yourself)
// 5. Get more training data
```

### Underfitting (Both losses high)

**Solutions:**

```go
// 1. Increase model capacity
Dense(512)  // More units
// Or add more layers

// 2. Train longer
Epochs: 200

// 3. Reduce regularization
WeightDecay: 0.0
// Remove dropout layers

// 4. Increase learning rate
LR: 0.01
```

---

## Memory Issues

### Out of Memory

**Solutions:**

```go
// 1. Reduce batch size
BatchSize: 16

// 2. Use gradient accumulation
GradientAccumulationSteps: 4  // Effective batch = 16 * 4 = 64

// 3. Use memory-efficient optimizer
flow.Lion(...)  // Less memory than Adam
// or
flow.AdaFactor(...)

// 4. Reduce model size
Dense(128)  // instead of Dense(1024)
```

---

## Common Mistakes

### Forgetting to Handle Errors

```go
// ❌ Bad - ignoring errors
net, _ := flow.NewNetwork(config).Build(shape)
net.Compile(compileConfig)  // Might fail silently

// ✅ Good - always check
net, err := flow.NewNetwork(config).Build(shape)
if err != nil {
    log.Fatalf("Build failed: %v", err)
}

err = net.Compile(compileConfig)
if err != nil {
    log.Fatalf("Compile failed: %v", err)
}
```

### Wrong Activation for Task

```go
// ❌ Classification with Linear output
Dense(10).WithActivation(flow.Linear())
// Probabilities won't sum to 1!

// ✅ Classification with Softmax
Dense(10).WithActivation(flow.Softmax())

// ❌ Binary classification with Softmax
Dense(1).WithActivation(flow.Softmax())
// Softmax needs multiple classes!

// ✅ Binary classification with Sigmoid
Dense(1).WithActivation(flow.Sigmoid())

// ❌ Regression with Sigmoid
Dense(1).WithActivation(flow.Sigmoid())
// Output stuck in [0, 1]!

// ✅ Regression with Linear
Dense(1).WithActivation(flow.Linear())
```

### Not Setting Regularizer

```go
// ❌ Missing regularizer
CompileConfig{
    Optimizer: flow.Adam(...),
    Loss:      flow.CrossEntropy(...),
    // Error: Regularizer is required
}

// ✅ Explicit regularizer (even if none)
CompileConfig{
    Optimizer:   flow.Adam(...),
    Loss:        flow.CrossEntropy(...),
    Regularizer: flow.NoReg(),  // Explicitly no regularization
    GradientClip: flow.GradientClipConfig{Mode: "none"},
}
```

### Not Setting GradientClip

```go
// ✅ Always set Mode, even if "none"
GradientClip: flow.GradientClipConfig{
    Mode:     "none",  // Required
    MaxNorm:  0.0,
    MaxValue: 0.0,
}
```

---

## Getting Help

1. **Read error messages carefully** - Flow provides specific guidance
2. **Check the examples** - Working code in `examples/`
3. **Print network summary** - `net.Summary()` shows architecture
4. **Start simple** - Get XOR working, then add complexity
5. **Compare with examples** - Diff your code against working examples

---

## Error Message Reference

| Error | Meaning | Solution |
|-------|---------|----------|
| "requires activation" | Layer missing activation | Add `WithActivation()` |
| "requires initializer" | Layer missing initializer | Add `WithInitializer()` |
| "requires bias initializer" | Bias enabled but no init | Add `WithBiasInitializer()` |
| "must be built" | Build not called | Call `Build(inputShape)` |
| "must be compiled" | Compile not called | Call `Compile(config)` |
| "no training data" | Empty input slice | Check your data loading |
| "shape mismatch" | Wrong tensor dimensions | Check input/output shapes |
| "Regularizer is required" | Missing regularizer | Add `Regularizer: flow.NoReg()` |
| "GradientClip.Mode required" | Missing clip mode | Add `Mode: "none"` |
