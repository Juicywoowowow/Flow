# Flow Architecture

This document describes the internal architecture of Flow and how its components interact.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        User Code                             │
│   inputs [][]float64  →  Network  →  predictions [][]float64│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Network Layer                           │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────────┐  │
│  │ Builder │→ │ Compile  │→ │  Train  │→ │ Predict/Eval │  │
│  └─────────┘  └──────────┘  └─────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Component Layer                           │
│  ┌────────┐ ┌───────────┐ ┌──────┐ ┌─────────┐ ┌─────────┐ │
│  │ Layers │ │ Optimizer │ │ Loss │ │ Metrics │ │Callbacks│ │
│  └────────┘ └───────────┘ └──────┘ └─────────┘ └─────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Tensor Layer                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  tensor: data []float64, shape []int, grad []float64   │ │
│  │  Operations: matmul, add, mul, activation functions    │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Component Hierarchy

### 1. Tensor Layer (`tensor.go`)

The foundation. Internal only, never exposed to users.

```go
type tensor struct {
    data   []float64  // Flat data storage
    shape  []int      // Dimensions [batch, height, width, channels]
    stride []int      // For efficient indexing
    grad   []float64  // Gradient storage for backprop
}
```

**Key Operations:**
- `matmul(a, b, out)` - Matrix multiplication
- `matmulTransA/TransB` - Transposed multiplication for backprop
- `addVec`, `mulScalar` - Element-wise operations
- `l2Norm`, `clip` - Gradient utilities

### 2. Layer Interface (`layer.go`, `conv.go`, `normalization.go`)

All layers implement this interface:

```go
type Layer interface {
    forward(input *tensor, training bool) (*tensor, error)
    backward(gradOutput *tensor) (*tensor, error)
    parameters() []*tensor
    gradients() []*tensor
    build(inputShape []int, rng *rand.Rand) error
    outputShape() []int
    name() string
}
```

**Layer Types:**
| File | Layers |
|------|--------|
| `layer.go` | Dense, Dropout, Flatten, BatchNorm |
| `conv.go` | Conv2D, MaxPool2D, AvgPool2D |
| `normalization.go` | LayerNorm, RMSNorm, GroupNorm |
| `recurrent.go` | LSTM, GRU, SimpleRNN |
| `attention.go` | MultiHeadAttention, SelfAttention |

### 3. Optimizer Interface (`optimizer.go`)

```go
type Optimizer interface {
    init(params []*tensor)
    step(params []*tensor, grads []*tensor)
    name() string
}
```

**Implementations:**
- SGD (with momentum, Nesterov)
- Adam, AdamW
- RMSprop, Adagrad
- Lion (Google 2023)
- AdaFactor

### 4. Network (`network.go`)

Orchestrates everything:

```go
type Network struct {
    layers      []Layer
    optimizer   Optimizer
    loss        Loss
    metrics     []Metric
    regularizer Regularizer
    gradClip    GradientClipConfig
    compiled    bool
    built       bool
    rng         *rand.Rand
    inputShape  []int
}
```

---

## Data Flow

### Forward Pass

```
Input [][]float64
       │
       ▼
  ┌─────────┐
  │ Reshape │ → tensor [batch, ...]
  └─────────┘
       │
       ▼
  ┌─────────┐     ┌─────────┐     ┌─────────┐
  │ Layer 1 │ ──▶ │ Layer 2 │ ──▶ │ Layer N │
  │ forward │     │ forward │     │ forward │
  └─────────┘     └─────────┘     └─────────┘
       │
       ▼
  ┌──────────┐
  │   Loss   │ → scalar loss value
  └──────────┘
```

### Backward Pass

```
  ┌──────────────┐
  │ Loss.gradient│ → gradOutput tensor
  └──────────────┘
         │
         ▼
  ┌─────────┐     ┌─────────┐     ┌─────────┐
  │ Layer N │ ◀── │ Layer 2 │ ◀── │ Layer 1 │
  │backward │     │backward │     │backward │
  └─────────┘     └─────────┘     └─────────┘
         │
         │ (gradients accumulated in each layer)
         ▼
  ┌───────────────┐
  │  Regularizer  │ → adds regularization gradients
  └───────────────┘
         │
         ▼
  ┌───────────────┐
  │ Gradient Clip │ → clips by norm or value
  └───────────────┘
         │
         ▼
  ┌───────────────┐
  │   Optimizer   │ → updates parameters
  └───────────────┘
```

---

## File Structure

```
Flow/
├── src/
│   ├── flow.go          # Package entry, version info
│   ├── tensor.go        # Internal tensor type & ops
│   ├── layer.go         # Dense, Dropout, Flatten, BatchNorm
│   ├── conv.go          # Conv2D, MaxPool2D, AvgPool2D
│   ├── normalization.go # LayerNorm, RMSNorm, GroupNorm
│   ├── recurrent.go     # LSTM, GRU, SimpleRNN
│   ├── attention.go     # MultiHeadAttention, SelfAttention
│   ├── activation.go    # Activation functions
│   ├── initializer.go   # Weight initializers
│   ├── loss.go          # Loss functions
│   ├── optimizer.go     # Optimizers
│   ├── regularizer.go   # L1, L2, ElasticNet
│   ├── scheduler.go     # LR schedulers
│   ├── callback.go      # Training callbacks
│   ├── metrics.go       # Evaluation metrics
│   ├── config.go        # Configuration structs
│   ├── utils.go         # Utility functions
│   └── network.go       # Network builder & training
├── examples/
│   ├── xor.go           # Classic XOR problem
│   ├── mnist.go         # Digit classification
│   ├── regression.go    # Function approximation
│   ├── cnn.go           # CNN with Conv2D
│   ├── sequence.go      # LSTM/GRU sequence classification
│   └── transformer.go   # Multi-head attention language model
├── docs/
│   ├── philosophy.md
│   ├── architecture.md
│   ├── api_reference.md
│   ├── optimization_guide.md
│   └── troubleshooting.md
├── readme.md
└── makefile
```

---

## Memory Layout

### Tensor Storage

Tensors use row-major (C-style) layout:

```
Shape: [2, 3, 4]  (2 batches, 3 rows, 4 cols)

Logical view:
  Batch 0:        Batch 1:
  [[a b c d]      [[m n o p]
   [e f g h]       [q r s t]
   [i j k l]]      [u v w x]]

Physical storage:
  [a b c d e f g h i j k l m n o p q r s t u v w x]
   └──── batch 0 ─────────┘ └──── batch 1 ──────────┘
```

### Stride Calculation

```go
stride[i] = product(shape[i+1:])

For shape [2, 3, 4]:
  stride = [12, 4, 1]
  
Index [b, r, c] → offset = b*12 + r*4 + c*1
```

---

## Thread Safety

Flow is **NOT** thread-safe by design:

- Training modifies network state
- Tensors are mutated in place for performance
- Use separate Network instances for parallel inference

For parallel training, use multiple processes or implement data parallelism at the batch level.
