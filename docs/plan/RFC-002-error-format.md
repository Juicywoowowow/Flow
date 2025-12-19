# RFC-002: Error Message Format

**Status:** Draft  
**Author:** Flow Team  
**Created:** 2024-12-19  

## Summary

Defines a concise, informative error message format for Flow. All errors include location, address, and cause without excessive decoration.

## Motivation

Current error messages are inconsistent and lack context. Users need:
- Immediate understanding of what failed
- Exact location in the network
- Memory addresses for debugging
- Tensor state at time of failure

Errors must be concise and actionable, not verbose walls of text.

## Error Format Specification

### Standard Format

```
flow: <component> <error_type> at layer <index> "<name>"
  location: <network_path>
  <context_lines>
  cause: <reason>
```

### Components

| Field | Description | Example |
|-------|-------------|---------|
| component | Which module failed | TensorMod, Dense, Conv2D |
| error_type | What kind of failure | shape mismatch, NaN detected, dimension error |
| index | Layer position (0-indexed) | layer 3 |
| name | Layer name if provided | "custom_norm" |
| location | Network path with marker | Dense(128) -> [TensorMod*] -> Dense(64) |
| context | Relevant tensor info | input: [32, 128] size=4096 |
| cause | Why it failed | output length 2048 does not match shape |

## Error Examples

### Shape Mismatch

```
flow: TensorMod shape mismatch at layer 3 "custom_norm"
  location: Dense(128) -> ReLU -> [TensorMod] -> Dense(64)
  input:    [32, 128] size=4096 addr=0xc0001a2000
  output:   [32, 64]  size=2048 addr=0xc0001b4000 (expected [32, 128])
  cause:    output shape changed without .WithOutputShape() declaration
```

### NaN Detection

```
flow: TensorMod NaN detected at layer 5 "normalize"
  location: Conv2D(32) -> MaxPool -> [TensorMod] -> Flatten
  output:   [16, 32, 32] size=16384 addr=0xc0002b8000
  corrupt:  3 NaN values at indices [1024, 2048, 4096]
  range:    min=-12.45 max=NaN
  cause:    division by zero or log of negative number in transform function
```

### Inf Detection

```
flow: Dense overflow at layer 2 "hidden"
  location: Input -> [Dense] -> ReLU
  output:   [32, 256] size=8192 addr=0xc0001f4000
  corrupt:  12 Inf values at indices [0, 256, 512, ...]
  range:    min=-Inf max=Inf
  cause:    weights exploded - consider gradient clipping or lower learning rate
```

### Dimension Mismatch

```
flow: Dense input dimension mismatch at layer 4 "output"
  location: Flatten -> Dense(128) -> [Dense] 
  expected: 128 input features
  received: 256 input features (shape [32, 256])
  cause:    previous layer output shape does not match Dense fan_in
```

### Build Error

```
flow: MultiHeadAttention build failed at layer 2
  location: Embedding -> [MultiHeadAttention]
  input:    [16] (1D)
  expected: [seqLen, embedDim] (2D)
  cause:    attention requires 2D input shape, got 1D
```

### Training Error

```
flow: training failed at epoch 45, batch 12
  layer:    3 "attention" (MultiHeadAttention)
  phase:    backward pass
  gradient: [32, 16, 64] size=32768 addr=0xc0003a2000
  corrupt:  NaN in gradient tensor
  cause:    exploding gradients - enable gradient clipping
```

## Tensor Info Format

When displaying tensor information:

```
<name>: [<shape>] size=<total_elements> addr=<memory_address>
```

With optional corruption info:

```
output: [32, 64] size=2048 addr=0xc0001b4000
corrupt: 3 NaN, 0 Inf at indices [45, 892, 1203]
range:   min=-2.45 max=NaN
```

## Network Path Format

Show the network flow with the failing layer marked:

```
Input -> Dense(128) -> ReLU -> [TensorMod] -> Dense(64) -> Output
                               ^^^^^^^^^^^
```

For complex networks, abbreviate:

```
... -> Layer5 -> [Layer6] -> Layer7 -> ...
```

## Implementation

### Error Structure

```go
type FlowError struct {
    Component   string      // "TensorMod", "Dense", etc.
    ErrorType   string      // "shape mismatch", "NaN detected"
    LayerIndex  int         // 0-indexed position
    LayerName   string      // user-provided name or ""
    Phase       string      // "forward", "backward", "build"
    
    Location    string      // formatted network path
    
    InputInfo   *TensorInfo // nil if not relevant
    OutputInfo  *TensorInfo // nil if not relevant
    ExpectedInfo string     // what was expected
    
    Cause       string      // human-readable cause
}

type TensorInfo struct {
    Shape    []int
    Size     int
    Address  string
    NaNCount int
    InfCount int
    MinValue float64
    MaxValue float64
    BadIndices []int  // first 10 corrupted indices
}
```

### Error Function

```go
func (e *FlowError) Error() string {
    var b strings.Builder
    
    // Line 1: Component and error type
    fmt.Fprintf(&b, "flow: %s %s at layer %d", e.Component, e.ErrorType, e.LayerIndex)
    if e.LayerName != "" {
        fmt.Fprintf(&b, " %q", e.LayerName)
    }
    b.WriteString("\n")
    
    // Line 2: Location
    fmt.Fprintf(&b, "  location: %s\n", e.Location)
    
    // Tensor info lines
    if e.InputInfo != nil {
        fmt.Fprintf(&b, "  input:    %s\n", e.InputInfo.Format())
    }
    if e.OutputInfo != nil {
        fmt.Fprintf(&b, "  output:   %s\n", e.OutputInfo.Format())
    }
    if e.ExpectedInfo != "" {
        fmt.Fprintf(&b, "  expected: %s\n", e.ExpectedInfo)
    }
    
    // Cause
    fmt.Fprintf(&b, "  cause:    %s", e.Cause)
    
    return b.String()
}
```

### Helper Functions

```go
// FormatTensorInfo creates the compact tensor description
func (t *TensorInfo) Format() string {
    s := fmt.Sprintf("%v size=%d addr=%s", t.Shape, t.Size, t.Address)
    if t.NaNCount > 0 || t.InfCount > 0 {
        s += fmt.Sprintf(" (corrupt: %d NaN, %d Inf)", t.NaNCount, t.InfCount)
    }
    return s
}

// BuildNetworkPath creates the location string
func BuildNetworkPath(layers []Layer, failedIndex int) string {
    // ... builds "Dense(128) -> [TensorMod] -> Dense(64)"
}

// ScanTensor checks for NaN/Inf and collects stats
func ScanTensor(t *tensor) *TensorInfo {
    // ... scans data, returns TensorInfo
}
```

## Guidelines

### Do

- Keep error messages under 6 lines for common errors
- Always include layer index and type
- Show memory addresses for debugging
- State the cause clearly in plain language

### Do Not

- Use decorative banners (===, ***, etc.)
- Use emojis
- Include full stack traces (unless panic)
- Show entire tensor contents (only samples/indices)

## Migration

Existing error messages will be updated to follow this format. This is a breaking change for users parsing error strings (not recommended practice).

## References

- Go error handling conventions
- Rust error messages (concise with context)
- Compiler error message best practices
