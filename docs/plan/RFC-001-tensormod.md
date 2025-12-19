# RFC-001: TensorMod Layer

**Status:** Draft  
**Author:** Flow Team  
**Created:** 2024-12-19  

## Summary

TensorMod is a power-user layer that allows arbitrary tensor transformations within a neural network. Users provide custom forward and backward functions to modify tensor data during training and inference.

## Motivation

Current Flow layers cover common operations, but researchers and power users often need:
- Custom normalization schemes not available in standard layers
- Novel activation functions during experimentation
- Data masking or gating mechanisms
- Debugging/inspection points in the network
- Rapid prototyping without modifying Flow source

Previously rejected due to safety concerns, this RFC proposes TensorMod with strict guardrails.

## API Design

### Basic Usage (Shape-Preserving)

```go
flow.TensorMod(
    // Forward function: transform input data
    func(data []float64, shape []int) []float64 {
        result := make([]float64, len(data))
        for i, v := range data {
            result[i] = v * 2.0  // Example: double all values
        }
        return result
    },
    // Backward function: compute gradient
    func(gradOut []float64, shape []int) []float64 {
        result := make([]float64, len(gradOut))
        for i, v := range gradOut {
            result[i] = v * 2.0  // Gradient of f(x) = 2x is 2
        }
        return result
    },
).Build()
```

### Shape-Changing Usage

```go
flow.TensorMod(forwardFn, backwardFn).
    WithOutputShape([]int{64, 32}).  // Declare new shape
    WithName("my_reshape").          // Optional name for debugging
    Build()
```

### Builder Methods

| Method | Required | Description |
|--------|----------|-------------|
| `TensorMod(forward, backward)` | Yes | Core transform functions |
| `.WithOutputShape([]int)` | If shape changes | Declare output dimensions |
| `.WithName(string)` | No | Name for error messages |
| `.WithValidation(level)` | No | Strict (default), Standard, Unsafe |
| `.Build()` | Yes | Returns Layer interface |

## Function Signatures

### Forward Function

```go
type TensorModForward func(data []float64, shape []int) []float64
```

- `data`: Flattened input tensor values (read-only, do not modify)
- `shape`: Input tensor dimensions (e.g., [batch, features])
- Returns: New data slice (must match declared output shape)

### Backward Function

```go
type TensorModBackward func(gradOutput []float64, shape []int) []float64
```

- `gradOutput`: Gradient flowing back from next layer
- `shape`: Shape of gradOutput tensor
- Returns: Gradient with respect to input (must match input shape)

## Validation Levels

### Strict (Default)

Every forward/backward call performs:
- Output length matches declared shape
- NaN detection (scan all values)
- Inf detection (scan all values)
- Nil/empty slice check

Performance impact: ~5-10% overhead

### Standard

First batch only:
- Shape validation
- NaN/Inf spot check (first 100 values)

Performance impact: ~1% overhead

### Unsafe

No validation. User assumes full responsibility.

Performance impact: 0% overhead

```go
.WithValidation(flow.ValidationStrict)   // default
.WithValidation(flow.ValidationStandard)
.WithValidation(flow.ValidationUnsafe)
```

## Error Handling

All errors follow the concise format defined in RFC-002. Example:

```
flow: TensorMod shape mismatch at layer 3 "custom_norm"
  location: Dense(128) -> ReLU -> [TensorMod] -> Dense(64)
  input:    [32, 128] size=4096 addr=0xc0001a2000
  output:   [32, 64]  size=2048 addr=0xc0001b4000 (expected [32, 128])
  cause:    output shape changed without .WithOutputShape() declaration
```

## Implementation Details

### Struct Definition

```go
type TensorModLayer struct {
    forward     TensorModForward
    backward    TensorModBackward
    outputShape []int
    name        string
    validation  ValidationLevel
    
    // Cached for backward pass
    inputCache  *tensor
    inputShape  []int
    built       bool
}
```

### Layer Interface Implementation

```go
func (t *TensorModLayer) build(inputShape []int, rng *rand.Rand) error
func (t *TensorModLayer) forward(input *tensor, training bool) (*tensor, error)
func (t *TensorModLayer) backward(gradOutput *tensor) (*tensor, error)
func (t *TensorModLayer) parameters() []*tensor    // Returns nil (no learnable params)
func (t *TensorModLayer) gradients() []*tensor     // Returns nil
func (t *TensorModLayer) outputShape() []int
func (t *TensorModLayer) name() string
```

### Panic Recovery

User functions are wrapped in recover() to catch panics:

```go
func (t *TensorModLayer) forward(input *tensor, training bool) (*tensor, error) {
    defer func() {
        if r := recover(); r != nil {
            // Format error with context and re-panic with better message
        }
    }()
    
    result := t.forward(input.data, input.shape)
    // ... validation ...
}
```

## Security Considerations

1. **No arbitrary code execution** - Functions are Go code compiled with the application
2. **Memory safety** - Go's slice bounds checking prevents buffer overflows
3. **No file/network access** - Transform functions should be pure (documented, not enforced)

## Rejected Alternatives

### Forward-Only Mode

Allowing forward function without backward would break training silently. Rejected.

### Auto-Gradient Estimation

Finite differences could estimate gradients, but:
- Expensive (2N forward passes per backward)
- Numerically unstable
- Hides user errors

Rejected in favor of requiring explicit backward function.

### Expression DSL

A domain-specific language for safe expressions was considered but adds complexity. May revisit in future RFC.

## Migration Path

This is a new feature. No migration required.

## Decisions

### 1. Utility Functions - APPROVED

Common utility functions will be provided to reduce boilerplate:

```go
// Clip values to a range
flow.ClipValues(min, max float64) TensorModForward

// Mask values based on condition
flow.MaskWhere(condition func(float64) bool, maskValue float64) TensorModForward

// Scale by constant
flow.Scale(factor float64) TensorModForward

// Apply element-wise function
flow.ElementWise(fn func(float64) float64) TensorModForward
```

Usage:

```go
flow.TensorMod(
    flow.ClipValues(-1.0, 1.0),
    flow.ClipValuesGrad(),  // Corresponding gradient
).Build()
```

### 2. Intermediate Caching - DEFERRED

Caching intermediate values between forward/backward passes adds complexity. Deferred to future RFC. Users can use closures if needed:

```go
var cached []float64
flow.TensorMod(
    func(data []float64, shape []int) []float64 {
        cached = computeIntermediate(data)  // User manages caching
        return transform(data, cached)
    },
    func(grad []float64, shape []int) []float64 {
        return gradTransform(grad, cached)
    },
).Build()
```

### 3. TensorModInspect - APPROVED

A forward-only inspection layer for debugging. Automatically disabled during training to prevent gradient issues.

```go
flow.TensorModInspect(func(data []float64, shape []int) {
    fmt.Printf("Layer output: shape=%v min=%.4f max=%.4f\n", 
        shape, minVal(data), maxVal(data))
}).WithName("debug_point").Build()
```

Behavior:
- During `training=true`: passes data through unchanged, does NOT call inspect function
- During `training=false` (inference): calls inspect function, then passes data through
- No backward function required (gradient passes through as identity)

Implementation:

```go
type TensorModInspectLayer struct {
    inspect    func([]float64, []int)
    name       string
    inputShape []int
    built      bool
}

func (t *TensorModInspectLayer) forward(input *tensor, training bool) (*tensor, error) {
    if !training {
        t.inspect(input.data, input.shape)
    }
    return input.clone(), nil
}

func (t *TensorModInspectLayer) backward(gradOutput *tensor) (*tensor, error) {
    return gradOutput.clone(), nil  // Identity gradient
}
```

## Future Considerations

- Expression DSL for safe tensor math
- Automatic gradient verification in debug builds
- TensorMod with learnable parameters

## References

- PyTorch Lambda layers
- Keras Lambda layers  
- JAX custom_vjp for custom gradients

