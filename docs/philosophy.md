# Flow Philosophy

> *"Explicit is better than implicit. Simple is better than complex."*

## Core Principles

Flow is built on a fundamental belief: **power users deserve complete control**. Unlike libraries that hide complexity behind "smart defaults," Flow exposes every lever and dial, trusting you to make informed decisions.

### 1. No Magic Defaults

```go
// ❌ Other libraries might do this:
optimizer := SomeLib.Adam()  // Hidden: LR=0.001, Beta1=0.9, Beta2=0.999...

// ✅ Flow requires explicit configuration:
optimizer := flow.Adam(flow.AdamConfig{
    LR:          0.001,
    Beta1:       0.9,
    Beta2:       0.999,
    Epsilon:     1e-8,
    WeightDecay: 0.0,
    AMSGrad:     false,
})
```

**Why?** Because defaults that "just work" eventually don't. When your model fails to converge, you need to know exactly what values are being used. Flow makes this explicit from day one.

### 2. Errors Over Panics

Flow returns errors instead of panicking. This is Go idiomatic and allows you to handle failures gracefully:

```go
net, err := flow.NewNetwork(config).
    AddLayer(layer).
    Build(inputShape)

if err != nil {
    log.Printf("Network build failed: %v", err)
    return
}
```

**Why?** Production systems need graceful degradation. A panic in a training loop can lose hours of progress.

### 3. Composition Over Inheritance

Every component in Flow is composable. Layers, optimizers, callbacks, and metrics can be mixed freely:

```go
// Mix any optimizer with any scheduler
optimizer := flow.Lion(lionConfig)
scheduler := flow.CosineAnnealing(cosineConfig)

// Mix any layers
layers := []flow.Layer{
    flow.Conv2D(...).Build(),
    flow.LayerNorm(...).Build(),  // Can use LayerNorm after Conv2D
    flow.Dense(...).Build(),
}
```

### 4. Performance Is Non-Negotiable

Flow is designed for speed:

- **Zero allocations in hot paths** where possible
- **No unnecessary abstractions** - data flows directly through operations
- **Minimal interface indirection** - concrete types where performance matters

### 5. Internal Tensors, External Safety

Tensors are internal to Flow. Users work with `[][]float64` for inputs/outputs:

```go
// Users provide data as simple slices
inputs := [][]float64{{1, 2, 3}, {4, 5, 6}}
targets := [][]float64{{0, 1}, {1, 0}}

// Flow handles the rest internally
result, err := net.Train(inputs, targets, config, callbacks)
```

**Why?** Exposing raw tensors invites footguns. Incorrect reshapes, memory aliasing, and gradient corruption are too easy. Flow protects you from yourself.

---

## Who Should Use Flow

### ✅ Good Fit
- Researchers who need fine-grained control
- Engineers who understand their hyperparameters
- Production systems requiring deterministic behavior
- Projects where "it just works" isn't good enough

### ❌ Poor Fit
- Beginners learning neural networks
- Rapid prototyping where speed > correctness
- Projects that prefer convention over configuration

---

## Design Decisions

### Builder Pattern

Flow uses the builder pattern for complex objects:

```go
flow.Dense(128).
    WithActivation(flow.ReLU()).
    WithInitializer(flow.HeNormal(1.0)).
    WithBiasInitializer(flow.Zeros()).
    WithBias(true).
    Build()
```

**Rationale:**
1. Clear what's being configured
2. Compile-time safety (can't forget required params)
3. Fluent, readable code
4. Easy to extend without breaking changes

### Config Structs

All configuration uses explicit structs:

```go
type AdamConfig struct {
    LR          float64
    Beta1       float64
    Beta2       float64
    Epsilon     float64
    WeightDecay float64
    AMSGrad     bool
}
```

**Rationale:**
1. Self-documenting - field names are the documentation
2. IDE autocomplete works
3. No positional parameter confusion
4. Easy to serialize/deserialize

### Validation at Build Time

Errors are caught when building, not during training:

```go
// This fails immediately with a clear error
net, err := flow.NewNetwork(config).
    AddLayer(flow.Dense(128).Build()).  // Error: missing activation
    Build(inputShape)
// err: "flow: DenseLayer requires activation - use WithActivation()"
```

---

## The Flow Way

1. **Read the error messages** - they tell you exactly what's missing
2. **Configure everything** - if you don't know a value, research it
3. **Start simple** - get a basic network working first
4. **Add complexity incrementally** - one layer/feature at a time
5. **Trust the types** - if it compiles, structure is correct

---

*Flow: Because your neural network deserves better than magic.*
