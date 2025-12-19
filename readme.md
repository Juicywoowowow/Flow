# Flow

A power-user focused neural network library for Go. No hidden defaults, no magic - every hyperparameter must be explicitly configured.

> ⚠️ **Not recommended for beginners.** Flow requires explicit configuration of all hyperparameters. If you want a simpler API, look elsewhere.

## Philosophy

- **Maximum Control**: Every hyperparameter exposed, no magic defaults
- **Explicit Over Implicit**: Users must configure everything intentionally
- **Composable**: Layers, optimizers, and callbacks can be mixed freely
- **Performance-Oriented**: Efficient tensor operations, minimal allocations

## Installation

```bash
go get github.com/yourusername/flow
```

## Quick Start

```go
package main

import (
    "log"
    flow "flow/src"
)

func main() {
    // Build network - ALL parameters explicit
    net, err := flow.NewNetwork(flow.NetworkConfig{
        Seed:    42,
        Verbose: true,
    }).
        AddLayer(flow.Dense(128).
            WithActivation(flow.ReLU()).
            WithInitializer(flow.HeNormal(1.0)).
            WithBiasInitializer(flow.Zeros()).
            WithBias(true).
            Build()).
        AddLayer(flow.Dense(10).
            WithActivation(flow.Softmax()).
            WithInitializer(flow.XavierNormal(1.0)).
            WithBiasInitializer(flow.Zeros()).
            WithBias(true).
            Build()).
        Build([]int{784})

    if err != nil {
        log.Fatal(err)
    }

    // Compile - every parameter required
    err = net.Compile(flow.CompileConfig{
        Optimizer: flow.Adam(flow.AdamConfig{
            LR:          0.001,
            Beta1:       0.9,
            Beta2:       0.999,
            Epsilon:     1e-8,
            WeightDecay: 0.0,
            AMSGrad:     false,
        }),
        Loss: flow.CrossEntropy(flow.CrossEntropyConfig{
            LabelSmoothing: 0.0,
        }),
        Metrics:     []flow.Metric{flow.Accuracy()},
        Regularizer: flow.NoReg(),
        GradientClip: flow.GradientClipConfig{
            Mode:     "none",
            MaxNorm:  0.0,
            MaxValue: 0.0,
        },
    })

    // Train
    result, err := net.Train(inputs, targets, flow.TrainConfig{
        Epochs:          100,
        BatchSize:       32,
        Shuffle:         true,
        ValidationSplit: 0.2,
        Verbose:         1,
    }, nil)
}
```

## API Reference

### Layers

| Layer | Builder | Description |
|-------|---------|-------------|
| `Dense(units)` | `.WithActivation()`, `.WithInitializer()`, `.WithBiasInitializer()`, `.WithBias()` | Fully connected |
| `Conv2D(filters, kernelSize)` | `.WithStride()`, `.WithPadding()`, `.WithActivation()`, `.WithInitializer()`, `.WithBias()` | 2D Convolution |
| `MaxPool2D(poolSize)` | `.WithStride()`, `.WithPadding()` | Max pooling |
| `AvgPool2D(poolSize)` | `.WithStride()`, `.WithPadding()` | Average pooling |
| `Embedding(vocabSize, embedDim)` | `.WithInitializer()`, `.WithPaddingIdx()` | Token → Vector lookup |
| `PositionalEncoding(maxLen, embedDim)` | `.WithDropout()`, `.WithLearned()` | Sinusoidal/learned position info |
| `Residual(layers...)` | `.WithProjection()` | Skip connection wrapper |
| `LSTM(units)` | `.WithReturnSequences()`, `.WithDropout()`, `.WithInitializer()`, `.WithRecurrentInitializer()`, `.WithBiasInitializer()` | Long Short-Term Memory |
| `GRU(units)` | `.WithReturnSequences()`, `.WithDropout()`, `.WithInitializer()`, `.WithRecurrentInitializer()`, `.WithBiasInitializer()` | Gated Recurrent Unit |
| `SimpleRNN(units)` | `.WithReturnSequences()`, `.WithActivation()`, `.WithInitializer()`, `.WithRecurrentInitializer()`, `.WithBiasInitializer()` | Basic RNN |
| `MultiHeadAttention(numHeads, keyDim)` | `.WithValueDim()`, `.WithDropout()`, `.WithBias()`, `.WithInitializer()`, `.WithBiasInitializer()` | Multi-Head Self-Attention |
| `SelfAttention(embedDim)` | `.WithBias()`, `.WithInitializer()`, `.WithBiasInitializer()` | Single-Head Self-Attention |
| `Dropout(rate)` | `.WithSeed()` | Random dropout |
| `Flatten()` | - | Flatten to 1D |
| `BatchNorm(epsilon, momentum)` | - | Batch normalization |
| `LayerNorm(epsilon)` | - | Layer normalization (for Transformers) |
| `RMSNorm(epsilon)` | - | RMS normalization (LLaMA-style) |
| `GroupNorm(numGroups, epsilon)` | - | Group normalization |

### Activations

| Function | Description |
|----------|-------------|
| `ReLU()` | Rectified Linear Unit |
| `LeakyReLU(negSlope)` | Leaky ReLU |
| `ELU(alpha)` | Exponential Linear Unit |
| `Sigmoid()` | Logistic sigmoid |
| `Tanh()` | Hyperbolic tangent |
| `Softmax()` | Softmax (multi-class output) |
| `Swish()` | x * sigmoid(x) |
| `GELU()` | Gaussian Error Linear Unit |
| `Linear()` | Identity (no-op) |

### Optimizers

| Optimizer | Config Required |
|-----------|-----------------|
| `SGD(SGDConfig)` | `LR`, `Momentum`, `Dampening`, `WeightDecay`, `Nesterov` |
| `Adam(AdamConfig)` | `LR`, `Beta1`, `Beta2`, `Epsilon`, `WeightDecay`, `AMSGrad` |
| `AdamW(AdamWConfig)` | `LR`, `Beta1`, `Beta2`, `Epsilon`, `WeightDecay` |
| `RMSprop(RMSpropConfig)` | `LR`, `Alpha`, `Epsilon`, `WeightDecay`, `Momentum`, `Centered` |
| `Adagrad(AdagradConfig)` | `LR`, `LRDecay`, `WeightDecay`, `Epsilon` |
| `Lion(LionConfig)` | `LR`, `Beta1`, `Beta2`, `WeightDecay` – Google's memory-efficient optimizer |
| `AdaFactor(AdaFactorConfig)` | `LR`, `Beta2Decay`, `Epsilon1`, `Epsilon2`, `ClipThreshold`, `WeightDecay` |

### Loss Functions

| Loss | Config Required |
|------|-----------------|
| `MSE(MSEConfig)` | `Reduction` ("mean" or "sum") |
| `MAE(MAEConfig)` | `Reduction` |
| `Huber(HuberConfig)` | `Delta`, `Reduction` |
| `CrossEntropy(CrossEntropyConfig)` | `LabelSmoothing` |
| `BinaryCrossEntropy(BinaryCrossEntropyConfig)` | `Reduction` |
| `KLDivergence(KLDivConfig)` | `Reduction` |

### Initializers

| Initializer | Parameters |
|-------------|------------|
| `HeNormal(gain)` | gain (float64) |
| `HeUniform(gain)` | gain (float64) |
| `XavierNormal(gain)` | gain (float64) |
| `XavierUniform(gain)` | gain (float64) |
| `LeCunNormal(gain)` | gain (float64) |
| `LeCunUniform(gain)` | gain (float64) |
| `Zeros()` | - |
| `Ones()` | - |
| `Constant(value)` | value (float64) |
| `RandomNormal(mean, std)` | mean, std (float64) |
| `RandomUniform(min, max)` | min, max (float64) |

### Regularizers

| Regularizer | Parameters |
|-------------|------------|
| `L1(lambda)` | lambda (float64) |
| `L2(lambda)` | lambda (float64) |
| `ElasticNet(l1Lambda, l2Lambda, l1Ratio)` | lambdas and ratio |
| `NoReg()` | - |

### Learning Rate Schedulers

| Scheduler | Config |
|-----------|--------|
| `StepDecay(StepDecayConfig)` | `StepSize`, `Gamma` |
| `ExponentialDecay(ExponentialDecayConfig)` | `Gamma` |
| `CosineAnnealing(CosineAnnealingConfig)` | `TMax`, `EtaMin`, `EtaMax` |
| `WarmRestarts(WarmRestartsConfig)` | `T0`, `TMult`, `EtaMin`, `EtaMax` |
| `LinearDecay(LinearDecayConfig)` | `StartLR`, `EndLR`, `TotalEpochs` |
| `PolynomialDecay(PolynomialDecayConfig)` | `StartLR`, `EndLR`, `Power`, `TotalEpochs` |
| `Warmup(WarmupConfig)` | `WarmupEpochs`, `TargetLR`, `InitialLR` |
| `ConstantLR()` | - |

### Callbacks

| Callback | Purpose |
|----------|---------|
| `EarlyStopping(EarlyStoppingConfig)` | Stop when metric stops improving |
| `PrintProgress(PrintProgressConfig)` | Print training progress |
| `History()` | Record training history |
| `LRSchedulerCallback_(LRSchedulerConfig)` | Apply LR schedule |
| `GradientClipping(GradientClippingConfig)` | Clip gradients |

### Metrics

| Metric | Description |
|--------|-------------|
| `Accuracy()` | Classification accuracy |
| `Precision(PrecisionConfig)` | Precision (binary) |
| `Recall(RecallConfig)` | Recall (binary) |
| `F1Score(F1Config)` | F1 score |
| `MeanSquaredError()` | MSE |
| `MeanAbsoluteError()` | MAE |
| `TopKAccuracy(TopKConfig)` | Top-K accuracy |

## Examples

See the `examples/` directory:

- `xor.go` - Classic XOR problem
- `mnist.go` - Digit classification with dropout and early stopping
- `regression.go` - Function approximation with LR scheduling
- `cnn.go` - CNN with Conv2D, MaxPool2D, and Lion optimizer
- `sequence.go` - LSTM/GRU sequence classification
- `transformer.go` - Multi-head attention character language model

## Building

```bash
make build      # Build library
make examples   # Build examples
make test       # Run tests
make clean      # Clean artifacts
```

## License

MIT
