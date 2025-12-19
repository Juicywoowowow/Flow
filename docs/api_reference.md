# Flow API Reference

Complete reference for all Flow types, functions, and configurations.

---

## Network

### NewNetwork

Creates a new network builder.

```go
func NewNetwork(config NetworkConfig) *NetworkBuilder

type NetworkConfig struct {
    Seed    int64  // Random seed for reproducibility
    Verbose bool   // Enable verbose output
}
```

### NetworkBuilder Methods

```go
func (n *NetworkBuilder) AddLayer(layer Layer) *NetworkBuilder
func (n *NetworkBuilder) Build(inputShape []int) (*Network, error)
```

### Network Methods

```go
func (n *Network) Compile(config CompileConfig) error
func (n *Network) Train(inputs, targets [][]float64, config TrainConfig, callbacks []Callback) (*TrainResult, error)
func (n *Network) Predict(inputs [][]float64) ([][]float64, error)
func (n *Network) Evaluate(inputs, targets [][]float64) (map[string]float64, error)
func (n *Network) Save(path string) error
func (n *Network) Load(path string) error
func (n *Network) Summary() string
```

---

## Layers

### Dense

Fully connected layer.

```go
flow.Dense(units int).
    WithActivation(act Activation).        // Required
    WithInitializer(init Initializer).     // Required
    WithBiasInitializer(init Initializer). // Required if WithBias(true)
    WithBias(useBias bool).                // Required
    Build()
```

### Conv2D

2D Convolution layer.

```go
flow.Conv2D(filters int, kernelSize [2]int).
    WithStride(h, w int).           // Default: 1, 1
    WithPadding(padding string).    // "valid" or "same"
    WithActivation(act Activation).
    WithInitializer(init Initializer).
    WithBiasInitializer(init Initializer).
    WithBias(useBias bool).
    Build()
```

### MaxPool2D / AvgPool2D

Pooling layers.

```go
flow.MaxPool2D(poolSize [2]int).
    WithStride(h, w int).
    WithPadding(padding string).
    Build()

flow.AvgPool2D(poolSize [2]int).
    WithStride(h, w int).
    WithPadding(padding string).
    Build()
```

### Dropout

```go
flow.Dropout(rate float64).
    WithSeed(seed int64).
    Build()
```

### Flatten

```go
flow.Flatten().Build()
```

### Normalization Layers

```go
flow.BatchNorm(epsilon, momentum float64).Build()
flow.LayerNorm(epsilon float64).Build()
flow.RMSNorm(epsilon float64).Build()
flow.GroupNorm(numGroups int, epsilon float64).Build()
```

### Recurrent Layers

#### LSTM

Long Short-Term Memory layer with forget, input, candidate, and output gates.

```go
flow.LSTM(units int).
    WithReturnSequences(bool).           // Return full sequence or last output
    WithDropout(rate float64).           // Dropout rate
    WithInitializer(init Initializer).   // Input weights initializer
    WithRecurrentInitializer(init Initializer). // Recurrent weights initializer
    WithBiasInitializer(init Initializer).      // Bias initializer
    Build()
```

**Input shape:** `[seqLen, features]`  
**Output shape:** `[seqLen, units]` if returnSequences, else `[units]`

#### GRU

Gated Recurrent Unit layer with reset and update gates.

```go
flow.GRU(units int).
    WithReturnSequences(bool).
    WithDropout(rate float64).
    WithInitializer(init Initializer).
    WithRecurrentInitializer(init Initializer).
    WithBiasInitializer(init Initializer).
    Build()
```

**Input shape:** `[seqLen, features]`  
**Output shape:** `[seqLen, units]` if returnSequences, else `[units]`

#### SimpleRNN

Basic recurrent layer.

```go
flow.SimpleRNN(units int).
    WithReturnSequences(bool).
    WithActivation(act Activation).      // Default: Tanh
    WithInitializer(init Initializer).
    WithRecurrentInitializer(init Initializer).
    WithBiasInitializer(init Initializer).
    Build()
```

### Attention Layers

#### MultiHeadAttention

Scaled dot-product multi-head self-attention as described in "Attention Is All You Need".

```go
flow.MultiHeadAttention(numHeads int, keyDim int).
    WithValueDim(dim int).               // Value dimension (default: keyDim)
    WithDropout(rate float64).           // Attention dropout
    WithBias(useBias bool).              // Use bias in projections
    WithInitializer(init Initializer).   // Weight initializer
    WithBiasInitializer(init Initializer).
    Build()
```

**Input shape:** `[seqLen, embedDim]`  
**Output shape:** `[seqLen, embedDim]`

#### SelfAttention

Single-head self-attention layer.

```go
flow.SelfAttention(embedDim int).
    WithBias(useBias bool).
    WithInitializer(init Initializer).
    WithBiasInitializer(init Initializer).
    Build()
```

**Input shape:** `[seqLen, embedDim]`  
**Output shape:** `[seqLen, embedDim]`

---

## Activations

| Function | Signature | Description |
|----------|-----------|-------------|
| `ReLU()` | `max(0, x)` | Rectified Linear Unit |
| `LeakyReLU(negSlope)` | `x if x > 0 else negSlope * x` | Leaky ReLU |
| `ELU(alpha)` | `x if x > 0 else alpha * (exp(x) - 1)` | Exponential LU |
| `Sigmoid()` | `1 / (1 + exp(-x))` | Logistic |
| `Tanh()` | `tanh(x)` | Hyperbolic tangent |
| `Softmax()` | `exp(x) / sum(exp(x))` | Normalized exponential |
| `Swish()` | `x * sigmoid(x)` | Self-gated |
| `GELU()` | `x * Φ(x)` | Gaussian Error LU |
| `Linear()` | `x` | Identity |

---

## Optimizers

### SGD

```go
flow.SGD(flow.SGDConfig{
    LR:          float64,  // Learning rate
    Momentum:    float64,  // Momentum factor
    Dampening:   float64,  // Dampening for momentum
    WeightDecay: float64,  // L2 penalty
    Nesterov:    bool,     // Use Nesterov momentum
})
```

### Adam

```go
flow.Adam(flow.AdamConfig{
    LR:          float64,  // Learning rate
    Beta1:       float64,  // First moment decay (0.9)
    Beta2:       float64,  // Second moment decay (0.999)
    Epsilon:     float64,  // Numerical stability (1e-8)
    WeightDecay: float64,  // L2 penalty
    AMSGrad:     bool,     // Use AMSGrad variant
})
```

### AdamW

Decoupled weight decay.

```go
flow.AdamW(flow.AdamWConfig{
    LR:          float64,
    Beta1:       float64,
    Beta2:       float64,
    Epsilon:     float64,
    WeightDecay: float64,
})
```

### RMSprop

```go
flow.RMSprop(flow.RMSpropConfig{
    LR:          float64,
    Alpha:       float64,  // Smoothing constant (0.99)
    Epsilon:     float64,
    WeightDecay: float64,
    Momentum:    float64,
    Centered:    bool,     // Compute centered RMSprop
})
```

### Adagrad

```go
flow.Adagrad(flow.AdagradConfig{
    LR:          float64,
    LRDecay:     float64,
    WeightDecay: float64,
    Epsilon:     float64,
})
```

### Lion

Google's memory-efficient optimizer.

```go
flow.Lion(flow.LionConfig{
    LR:          float64,  // Learning rate (try 0.0001)
    Beta1:       float64,  // Update interpolation (0.9)
    Beta2:       float64,  // Momentum decay (0.99)
    WeightDecay: float64,  // Weight decay
})
```

### AdaFactor

Memory-efficient for large models.

```go
flow.AdaFactor(flow.AdaFactorConfig{
    LR:             float64,
    Beta2Decay:     float64,
    Epsilon1:       float64,
    Epsilon2:       float64,
    ClipThreshold:  float64,
    WeightDecay:    float64,
    ScaleParameter: bool,
    RelativeStep:   bool,
})
```

---

## Loss Functions

### MSE

```go
flow.MSE(flow.MSEConfig{
    Reduction: string,  // "mean" or "sum"
})
```

### MAE

```go
flow.MAE(flow.MAEConfig{
    Reduction: string,
})
```

### Huber

```go
flow.Huber(flow.HuberConfig{
    Delta:     float64,  // Threshold for L1 vs L2
    Reduction: string,
})
```

### CrossEntropy

```go
flow.CrossEntropy(flow.CrossEntropyConfig{
    LabelSmoothing: float64,  // 0.0 to 1.0
})
```

### BinaryCrossEntropy

```go
flow.BinaryCrossEntropy(flow.BinaryCrossEntropyConfig{
    Reduction: string,
})
```

### KLDivergence

```go
flow.KLDivergence(flow.KLDivConfig{
    Reduction: string,
})
```

---

## Initializers

| Initializer | Parameters | Formula |
|-------------|------------|---------|
| `HeNormal(gain)` | gain | `N(0, gain * sqrt(2/fan_in))` |
| `HeUniform(gain)` | gain | `U(-limit, limit)` where `limit = gain * sqrt(6/fan_in)` |
| `XavierNormal(gain)` | gain | `N(0, gain * sqrt(2/(fan_in+fan_out)))` |
| `XavierUniform(gain)` | gain | `U(-limit, limit)` |
| `LeCunNormal(gain)` | gain | `N(0, gain * sqrt(1/fan_in))` |
| `LeCunUniform(gain)` | gain | `U(-limit, limit)` |
| `Zeros()` | - | All zeros |
| `Ones()` | - | All ones |
| `Constant(value)` | value | All same value |
| `RandomNormal(mean, std)` | mean, std | `N(mean, std)` |
| `RandomUniform(min, max)` | min, max | `U(min, max)` |

---

## Regularizers

```go
flow.L1(lambda float64)      // L1 regularization
flow.L2(lambda float64)      // L2 regularization
flow.ElasticNet(l1, l2, ratio float64)  // Combined
flow.NoReg()                 // No regularization
```

---

## Learning Rate Schedulers

```go
flow.StepDecay(flow.StepDecayConfig{StepSize: int, Gamma: float64})
flow.ExponentialDecay(flow.ExponentialDecayConfig{Gamma: float64})
flow.CosineAnnealing(flow.CosineAnnealingConfig{TMax: int, EtaMin: float64, EtaMax: float64})
flow.WarmRestarts(flow.WarmRestartsConfig{T0: int, TMult: int, EtaMin: float64, EtaMax: float64})
flow.LinearDecay(flow.LinearDecayConfig{StartLR: float64, EndLR: float64, TotalEpochs: int})
flow.PolynomialDecay(flow.PolynomialDecayConfig{...})
flow.Warmup(flow.WarmupConfig{WarmupEpochs: int, TargetLR: float64, InitialLR: float64})
flow.ConstantLR()
```

---

## Callbacks

```go
flow.EarlyStopping(flow.EarlyStoppingConfig{
    Monitor:     string,   // Metric to watch ("val_loss")
    MinDelta:    float64,  // Minimum improvement
    Patience:    int,      // Epochs to wait
    Mode:        string,   // "min" or "max"
    RestoreBest: bool,     // Restore best weights
})

flow.PrintProgress(flow.PrintProgressConfig{
    PrintEvery: int,  // Print every N epochs
})

flow.History()  // Records training history

flow.LRSchedulerCallback_(flow.LRSchedulerConfig{
    Scheduler: Scheduler,
    InitialLR: float64,
})

flow.GradientClipping(flow.GradientClippingConfig{
    MaxNorm:   float64,
    ClipValue: float64,
    Mode:      string,  // "norm" or "value"
})
```

---

## Metrics

```go
flow.Accuracy()
flow.Precision(flow.PrecisionConfig{Threshold: float64})
flow.Recall(flow.RecallConfig{Threshold: float64})
flow.F1Score(flow.F1Config{Threshold: float64})
flow.MeanSquaredError()
flow.MeanAbsoluteError()
flow.TopKAccuracy(flow.TopKConfig{K: int})
```

---

## Configuration Structs

### TrainConfig

```go
type TrainConfig struct {
    Epochs                    int
    BatchSize                 int
    Shuffle                   bool
    ValidationSplit           float64
    Verbose                   int
    GradientAccumulationSteps int
}
```

### CompileConfig

```go
type CompileConfig struct {
    Optimizer    Optimizer
    Loss         Loss
    Metrics      []Metric
    Regularizer  Regularizer
    GradientClip GradientClipConfig
}
```

### GradientClipConfig

```go
type GradientClipConfig struct {
    Mode     string   // "norm", "value", or "none"
    MaxNorm  float64
    MaxValue float64
}
```
