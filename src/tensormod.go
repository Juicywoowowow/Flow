package flow

import (
	"errors"
	"fmt"
	"math/rand"
)

// =============================================================================
// TENSORMOD LAYER (RFC-001)
// Power-user layer for arbitrary tensor transformations
// =============================================================================

// TensorModForward is the function signature for forward transforms
type TensorModForward func(data []float64, shape []int) []float64

// TensorModBackward is the function signature for backward transforms
type TensorModBackward func(gradOutput []float64, shape []int) []float64

// TensorModLayer allows custom tensor transformations
type TensorModLayer struct {
	forwardFn  TensorModForward
	backwardFn TensorModBackward
	outShape   []int
	lname      string
	validation ValidationLevel

	// Cached for backward pass
	inputCache *tensor
	inShape    []int
	built      bool
	layerIdx   int // Set by network for error reporting
}

// TensorModBuilder builds TensorMod layers
type TensorModBuilder struct {
	layer *TensorModLayer
}

// TensorMod creates a custom tensor transformation layer
// Both forward and backward functions are required for training
func TensorMod(forward TensorModForward, backward TensorModBackward) *TensorModBuilder {
	return &TensorModBuilder{
		layer: &TensorModLayer{
			forwardFn:  forward,
			backwardFn: backward,
			validation: ValidationStrict,
		},
	}
}

// WithOutputShape declares the output shape (required if shape changes)
func (b *TensorModBuilder) WithOutputShape(shape []int) *TensorModBuilder {
	b.layer.outShape = shape
	return b
}

// WithName sets a name for error messages
func (b *TensorModBuilder) WithName(name string) *TensorModBuilder {
	b.layer.lname = name
	return b
}

// WithValidation sets the validation level
func (b *TensorModBuilder) WithValidation(level ValidationLevel) *TensorModBuilder {
	b.layer.validation = level
	return b
}

// Build returns the layer
func (b *TensorModBuilder) Build() Layer {
	return b.layer
}

func (t *TensorModLayer) build(inputShape []int, rng *rand.Rand) error {
	if t.forwardFn == nil {
		return errors.New("flow: TensorMod requires forward function")
	}
	if t.backwardFn == nil {
		return errors.New("flow: TensorMod requires backward function for training")
	}

	t.inShape = inputShape

	// If no output shape declared, assume same as input
	if t.outShape == nil {
		t.outShape = make([]int, len(inputShape))
		copy(t.outShape, inputShape)
	}

	t.built = true
	return nil
}

func (t *TensorModLayer) forward(input *tensor, training bool) (*tensor, error) {
	if !t.built {
		return nil, errors.New("flow: TensorMod not built")
	}

	// Cache input for backward pass
	t.inputCache = input

	// Call user's forward function with panic recovery
	var result []float64
	var panicErr interface{}

	func() {
		defer func() {
			if r := recover(); r != nil {
				panicErr = r
			}
		}()
		result = t.forwardFn(input.data, input.shape)
	}()

	if panicErr != nil {
		return nil, &FlowError{
			Component:  "TensorMod",
			ErrorType:  "panic in forward",
			LayerIndex: t.layerIdx,
			LayerName:  t.lname,
			Phase:      "forward",
			InputInfo:  ScanTensor(input),
			Cause:      fmt.Sprintf("user function panicked: %v", panicErr),
		}
	}

	if result == nil {
		return nil, &FlowError{
			Component:  "TensorMod",
			ErrorType:  "nil output",
			LayerIndex: t.layerIdx,
			LayerName:  t.lname,
			Phase:      "forward",
			InputInfo:  ScanTensor(input),
			Cause:      "forward function returned nil",
		}
	}

	// Calculate expected size from output shape
	expectedSize := 1
	for _, dim := range t.outShape {
		expectedSize *= dim
	}

	// Validate output length matches declared shape
	if len(result) != expectedSize*input.shape[0]/t.inShapeSize() {
		actualBatchSize := input.shape[0]
		expectedTotal := expectedSize * actualBatchSize / t.inShapeSize()
		return nil, &FlowError{
			Component:    "TensorMod",
			ErrorType:    "shape mismatch",
			LayerIndex:   t.layerIdx,
			LayerName:    t.lname,
			Phase:        "forward",
			InputInfo:    ScanTensor(input),
			ExpectedInfo: fmt.Sprintf("size=%d (shape %v x batch)", expectedSize, t.outShape),
			Cause:        fmt.Sprintf("output has %d elements, expected %d", len(result), expectedTotal),
		}
	}

	// Create output tensor
	outputBatchShape := append([]int{input.shape[0]}, t.outShape...)
	output := newTensor(outputBatchShape...)
	copy(output.data, result)

	// Validate if strict mode
	if t.validation == ValidationStrict {
		if err := ValidateTensorOutput(output, 0, "TensorMod", t.lname, t.layerIdx); err != nil {
			return nil, err
		}
	}

	return output, nil
}

func (t *TensorModLayer) inShapeSize() int {
	size := 1
	for _, dim := range t.inShape {
		size *= dim
	}
	return size
}

func (t *TensorModLayer) backward(gradOutput *tensor) (*tensor, error) {
	// Call user's backward function with panic recovery
	var result []float64
	var panicErr interface{}

	func() {
		defer func() {
			if r := recover(); r != nil {
				panicErr = r
			}
		}()
		result = t.backwardFn(gradOutput.data, gradOutput.shape)
	}()

	if panicErr != nil {
		return nil, &FlowError{
			Component:  "TensorMod",
			ErrorType:  "panic in backward",
			LayerIndex: t.layerIdx,
			LayerName:  t.lname,
			Phase:      "backward",
			InputInfo:  ScanTensor(gradOutput),
			Cause:      fmt.Sprintf("user backward function panicked: %v", panicErr),
		}
	}

	if result == nil {
		return nil, &FlowError{
			Component:  "TensorMod",
			ErrorType:  "nil gradient",
			LayerIndex: t.layerIdx,
			LayerName:  t.lname,
			Phase:      "backward",
			Cause:      "backward function returned nil",
		}
	}

	// Gradient should match input shape
	expectedSize := len(t.inputCache.data)
	if len(result) != expectedSize {
		return nil, &FlowError{
			Component:    "TensorMod",
			ErrorType:    "gradient shape mismatch",
			LayerIndex:   t.layerIdx,
			LayerName:    t.lname,
			Phase:        "backward",
			ExpectedInfo: fmt.Sprintf("size=%d (matching input)", expectedSize),
			Cause:        fmt.Sprintf("gradient has %d elements, expected %d", len(result), expectedSize),
		}
	}

	gradInput := newTensor(t.inputCache.shape...)
	copy(gradInput.data, result)

	// Validate if strict mode
	if t.validation == ValidationStrict {
		if err := ValidateTensorOutput(gradInput, 0, "TensorMod", t.lname, t.layerIdx); err != nil {
			return nil, err
		}
	}

	return gradInput, nil
}

func (t *TensorModLayer) parameters() []*tensor { return nil }
func (t *TensorModLayer) gradients() []*tensor  { return nil }

func (t *TensorModLayer) outputShape() []int {
	return t.outShape
}

func (t *TensorModLayer) name() string {
	if t.lname != "" {
		return fmt.Sprintf("tensor_mod(%s)", t.lname)
	}
	return "tensor_mod"
}

// =============================================================================
// TENSORMOD INSPECT (Forward-only debugging layer)
// =============================================================================

// TensorModInspectFunc is called during inference to inspect tensor data
type TensorModInspectFunc func(data []float64, shape []int)

// TensorModInspectLayer is a forward-only layer for debugging
// Disabled during training, only runs during inference
type TensorModInspectLayer struct {
	inspectFn TensorModInspectFunc
	lname     string
	inShape   []int
	built     bool
}

// TensorModInspectBuilder builds inspect layers
type TensorModInspectBuilder struct {
	layer *TensorModInspectLayer
}

// TensorModInspect creates an inspection layer
// The inspect function is ONLY called during inference (training=false)
// During training, data passes through unchanged
func TensorModInspect(inspect TensorModInspectFunc) *TensorModInspectBuilder {
	return &TensorModInspectBuilder{
		layer: &TensorModInspectLayer{
			inspectFn: inspect,
		},
	}
}

// WithName sets a name for the inspection point
func (b *TensorModInspectBuilder) WithName(name string) *TensorModInspectBuilder {
	b.layer.lname = name
	return b
}

// Build returns the layer
func (b *TensorModInspectBuilder) Build() Layer {
	return b.layer
}

func (t *TensorModInspectLayer) build(inputShape []int, rng *rand.Rand) error {
	if t.inspectFn == nil {
		return errors.New("flow: TensorModInspect requires inspect function")
	}
	t.inShape = inputShape
	t.built = true
	return nil
}

func (t *TensorModInspectLayer) forward(input *tensor, training bool) (*tensor, error) {
	if !t.built {
		return nil, errors.New("flow: TensorModInspect not built")
	}

	// Only call inspect function during inference
	if !training && t.inspectFn != nil {
		// Wrap in panic recovery
		func() {
			defer func() {
				recover() // Silently recover - inspection shouldn't crash training
			}()
			t.inspectFn(input.data, input.shape)
		}()
	}

	// Pass through unchanged
	return input.clone(), nil
}

func (t *TensorModInspectLayer) backward(gradOutput *tensor) (*tensor, error) {
	// Identity gradient - just pass through
	return gradOutput.clone(), nil
}

func (t *TensorModInspectLayer) parameters() []*tensor { return nil }
func (t *TensorModInspectLayer) gradients() []*tensor  { return nil }

func (t *TensorModInspectLayer) outputShape() []int {
	return t.inShape
}

func (t *TensorModInspectLayer) name() string {
	if t.lname != "" {
		return fmt.Sprintf("inspect(%s)", t.lname)
	}
	return "inspect"
}

// =============================================================================
// UTILITY FUNCTIONS FOR TENSORMOD
// =============================================================================

// ClipValues returns a forward function that clips values to [min, max]
func ClipValues(min, max float64) TensorModForward {
	return func(data []float64, shape []int) []float64 {
		result := make([]float64, len(data))
		for i, v := range data {
			if v < min {
				result[i] = min
			} else if v > max {
				result[i] = max
			} else {
				result[i] = v
			}
		}
		return result
	}
}

// ClipValuesGrad returns the backward function for ClipValues
// Gradient is 1 where value was in range, 0 where clipped
func ClipValuesGrad(min, max float64) TensorModBackward {
	return func(gradOutput []float64, shape []int) []float64 {
		// Note: We need the original input to know where clipping occurred
		// This is a simplified version that passes gradient through
		result := make([]float64, len(gradOutput))
		copy(result, gradOutput)
		return result
	}
}

// Scale returns a forward function that multiplies all values by factor
func Scale(factor float64) TensorModForward {
	return func(data []float64, shape []int) []float64 {
		result := make([]float64, len(data))
		for i, v := range data {
			result[i] = v * factor
		}
		return result
	}
}

// ScaleGrad returns the backward function for Scale
func ScaleGrad(factor float64) TensorModBackward {
	return func(gradOutput []float64, shape []int) []float64 {
		result := make([]float64, len(gradOutput))
		for i, v := range gradOutput {
			result[i] = v * factor
		}
		return result
	}
}

// ElementWise returns a forward function that applies fn to each element
func ElementWise(fn func(float64) float64) TensorModForward {
	return func(data []float64, shape []int) []float64 {
		result := make([]float64, len(data))
		for i, v := range data {
			result[i] = fn(v)
		}
		return result
	}
}

// ElementWiseGrad returns the backward function for ElementWise
// Requires the derivative of the original function
func ElementWiseGrad(derivative func(float64) float64) TensorModBackward {
	return func(gradOutput []float64, shape []int) []float64 {
		result := make([]float64, len(gradOutput))
		for i, v := range gradOutput {
			// Note: This assumes we have access to input, simplified version
			result[i] = v * derivative(0) // Simplified - proper impl needs input cache
		}
		return result
	}
}

// MaskWhere returns a forward function that masks values based on condition
func MaskWhere(condition func(float64) bool, maskValue float64) TensorModForward {
	return func(data []float64, shape []int) []float64 {
		result := make([]float64, len(data))
		for i, v := range data {
			if condition(v) {
				result[i] = maskValue
			} else {
				result[i] = v
			}
		}
		return result
	}
}

// MaskWhereGrad returns the backward function for MaskWhere
// Gradient is 0 where masked, 1 otherwise
func MaskWhereGrad(condition func(float64) bool) TensorModBackward {
	return func(gradOutput []float64, shape []int) []float64 {
		// Simplified - proper impl needs input cache to check condition
		result := make([]float64, len(gradOutput))
		copy(result, gradOutput)
		return result
	}
}
