package flow

import (
	"fmt"
	"math"
	"strings"
)

// =============================================================================
// FLOW ERROR TYPES (RFC-002)
// Concise, informative error messages with location and context
// =============================================================================

// ValidationLevel controls how much validation TensorMod performs
type ValidationLevel int

const (
	ValidationStrict   ValidationLevel = iota // All checks, every call
	ValidationStandard                        // Shape + spot checks first batch
	ValidationUnsafe                          // No validation
)

// TensorInfo captures tensor state for error reporting
type TensorInfo struct {
	Shape      []int
	Size       int
	Address    string
	NaNCount   int
	InfCount   int
	MinValue   float64
	MaxValue   float64
	BadIndices []int // First 10 corrupted indices
}

// Format returns a compact string representation
func (t *TensorInfo) Format() string {
	s := fmt.Sprintf("%v size=%d addr=%s", t.Shape, t.Size, t.Address)
	if t.NaNCount > 0 || t.InfCount > 0 {
		s += fmt.Sprintf(" (corrupt: %d NaN, %d Inf)", t.NaNCount, t.InfCount)
	}
	return s
}

// FormatWithRange includes min/max range
func (t *TensorInfo) FormatWithRange() string {
	s := t.Format()
	if t.NaNCount == 0 && t.InfCount == 0 {
		s += fmt.Sprintf(" range=[%.4f, %.4f]", t.MinValue, t.MaxValue)
	}
	return s
}

// FlowError is the standard error type for Flow
type FlowError struct {
	Component    string      // "TensorMod", "Dense", etc.
	ErrorType    string      // "shape mismatch", "NaN detected"
	LayerIndex   int         // 0-indexed position
	LayerName    string      // user-provided name or ""
	Phase        string      // "forward", "backward", "build"
	Location     string      // formatted network path
	InputInfo    *TensorInfo // nil if not relevant
	OutputInfo   *TensorInfo // nil if not relevant
	ExpectedInfo string      // what was expected
	Cause        string      // human-readable cause
}

// Error implements the error interface
func (e *FlowError) Error() string {
	var b strings.Builder

	// Line 1: Component and error type
	fmt.Fprintf(&b, "flow: %s %s at layer %d", e.Component, e.ErrorType, e.LayerIndex)
	if e.LayerName != "" {
		fmt.Fprintf(&b, " %q", e.LayerName)
	}
	b.WriteString("\n")

	// Line 2: Location
	if e.Location != "" {
		fmt.Fprintf(&b, "  location: %s\n", e.Location)
	}

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

// ScanTensor checks for NaN/Inf and collects stats
func ScanTensor(t *tensor) *TensorInfo {
	if t == nil {
		return nil
	}

	info := &TensorInfo{
		Shape:      t.shape,
		Size:       len(t.data),
		Address:    fmt.Sprintf("%p", t),
		MinValue:   math.Inf(1),
		MaxValue:   math.Inf(-1),
		BadIndices: make([]int, 0, 10),
	}

	for i, v := range t.data {
		if math.IsNaN(v) {
			info.NaNCount++
			if len(info.BadIndices) < 10 {
				info.BadIndices = append(info.BadIndices, i)
			}
		} else if math.IsInf(v, 0) {
			info.InfCount++
			if len(info.BadIndices) < 10 {
				info.BadIndices = append(info.BadIndices, i)
			}
		} else {
			if v < info.MinValue {
				info.MinValue = v
			}
			if v > info.MaxValue {
				info.MaxValue = v
			}
		}
	}

	// Handle empty or all-corrupt tensors
	if math.IsInf(info.MinValue, 1) {
		info.MinValue = 0
	}
	if math.IsInf(info.MaxValue, -1) {
		info.MaxValue = 0
	}

	return info
}

// ValidateTensorOutput checks tensor for common issues
func ValidateTensorOutput(t *tensor, expectedSize int, component, layerName string, layerIndex int) error {
	if t == nil {
		return &FlowError{
			Component:  component,
			ErrorType:  "nil output",
			LayerIndex: layerIndex,
			LayerName:  layerName,
			Phase:      "forward",
			Cause:      "transform function returned nil",
		}
	}

	if len(t.data) == 0 {
		return &FlowError{
			Component:  component,
			ErrorType:  "empty output",
			LayerIndex: layerIndex,
			LayerName:  layerName,
			Phase:      "forward",
			OutputInfo: ScanTensor(t),
			Cause:      "transform function returned empty data slice",
		}
	}

	if expectedSize > 0 && len(t.data) != expectedSize {
		return &FlowError{
			Component:    component,
			ErrorType:    "size mismatch",
			LayerIndex:   layerIndex,
			LayerName:    layerName,
			Phase:        "forward",
			OutputInfo:   ScanTensor(t),
			ExpectedInfo: fmt.Sprintf("size=%d", expectedSize),
			Cause:        fmt.Sprintf("output has %d elements, expected %d", len(t.data), expectedSize),
		}
	}

	info := ScanTensor(t)
	if info.NaNCount > 0 {
		return &FlowError{
			Component:  component,
			ErrorType:  "NaN detected",
			LayerIndex: layerIndex,
			LayerName:  layerName,
			Phase:      "forward",
			OutputInfo: info,
			Cause:      fmt.Sprintf("%d NaN values at indices %v", info.NaNCount, info.BadIndices),
		}
	}

	if info.InfCount > 0 {
		return &FlowError{
			Component:  component,
			ErrorType:  "Inf detected",
			LayerIndex: layerIndex,
			LayerName:  layerName,
			Phase:      "forward",
			OutputInfo: info,
			Cause:      fmt.Sprintf("%d Inf values at indices %v - likely overflow", info.InfCount, info.BadIndices),
		}
	}

	return nil
}
