package flow

import (
	"errors"
	"fmt"
)

// =============================================================================
// LAYER FREEZING & FINE-TUNING SUPPORT
// Allows selective freezing of layers to prevent weight updates during training
// =============================================================================

// frozenLayers tracks which layer indices are frozen
// This is stored at the network level rather than modifying the Layer interface
type frozenState struct {
	frozen map[int]bool
}

// LayerFreezeInfo contains information about a layer's freeze status
type LayerFreezeInfo struct {
	Index      int
	Name       string
	Frozen     bool
	Parameters int
}

// Freeze freezes the specified layer indices, preventing their weights from updating
// during training. Frozen layers still participate in forward and backward passes,
// but their gradients are not applied.
func (n *Network) Freeze(indices ...int) error {
	if n.frozenState == nil {
		n.frozenState = &frozenState{frozen: make(map[int]bool)}
	}

	for _, idx := range indices {
		if idx < 0 || idx >= len(n.layers) {
			return errorf("layer index %d out of range [0, %d)", idx, len(n.layers))
		}
		n.frozenState.frozen[idx] = true
	}
	return nil
}

// Unfreeze unfreezes the specified layer indices, allowing their weights to update
func (n *Network) Unfreeze(indices ...int) error {
	if n.frozenState == nil {
		return nil
	}

	for _, idx := range indices {
		if idx < 0 || idx >= len(n.layers) {
			return errorf("layer index %d out of range [0, %d)", idx, len(n.layers))
		}
		delete(n.frozenState.frozen, idx)
	}
	return nil
}

// FreezeTo freezes all layers from index 0 to endIndex (exclusive)
// Useful for transfer learning where early layers learn general features
func (n *Network) FreezeTo(endIndex int) error {
	if endIndex < 0 || endIndex > len(n.layers) {
		return errorf("end index %d out of range [0, %d]", endIndex, len(n.layers))
	}

	indices := make([]int, endIndex)
	for i := 0; i < endIndex; i++ {
		indices[i] = i
	}
	return n.Freeze(indices...)
}

// FreezeFrom freezes all layers from startIndex to the end
func (n *Network) FreezeFrom(startIndex int) error {
	if startIndex < 0 || startIndex >= len(n.layers) {
		return errorf("start index %d out of range [0, %d)", startIndex, len(n.layers))
	}

	indices := make([]int, len(n.layers)-startIndex)
	for i := startIndex; i < len(n.layers); i++ {
		indices[i-startIndex] = i
	}
	return n.Freeze(indices...)
}

// FreezeAll freezes all layers in the network
func (n *Network) FreezeAll() error {
	indices := make([]int, len(n.layers))
	for i := range n.layers {
		indices[i] = i
	}
	return n.Freeze(indices...)
}

// UnfreezeAll unfreezes all layers in the network
func (n *Network) UnfreezeAll() {
	if n.frozenState != nil {
		n.frozenState.frozen = make(map[int]bool)
	}
}

// IsFrozen returns whether a specific layer is frozen
func (n *Network) IsFrozen(index int) bool {
	if n.frozenState == nil {
		return false
	}
	return n.frozenState.frozen[index]
}

// FrozenLayers returns the indices of all frozen layers
func (n *Network) FrozenLayers() []int {
	if n.frozenState == nil {
		return nil
	}

	result := make([]int, 0, len(n.frozenState.frozen))
	for idx := range n.frozenState.frozen {
		result = append(result, idx)
	}
	return result
}

// LayerInfo returns information about all layers including their freeze status
func (n *Network) LayerInfo() []LayerFreezeInfo {
	result := make([]LayerFreezeInfo, len(n.layers))
	for i, layer := range n.layers {
		params := layer.parameters()
		paramCount := 0
		for _, p := range params {
			paramCount += p.size()
		}
		result[i] = LayerFreezeInfo{
			Index:      i,
			Name:       layer.name(),
			Frozen:     n.IsFrozen(i),
			Parameters: paramCount,
		}
	}
	return result
}

// TrainableParameters returns the total count of trainable (non-frozen) parameters
func (n *Network) TrainableParameters() int {
	total := 0
	for i, layer := range n.layers {
		if !n.IsFrozen(i) {
			for _, p := range layer.parameters() {
				total += p.size()
			}
		}
	}
	return total
}

// TotalParameters returns the total count of all parameters
func (n *Network) TotalParameters() int {
	total := 0
	for _, layer := range n.layers {
		for _, p := range layer.parameters() {
			total += p.size()
		}
	}
	return total
}

// FreezeByName freezes all layers with the given name
// Useful for freezing all layers of a specific type (e.g., "conv2d", "dense")
func (n *Network) FreezeByName(name string) error {
	found := false
	for i, layer := range n.layers {
		if layer.name() == name {
			if err := n.Freeze(i); err != nil {
				return err
			}
			found = true
		}
	}
	if !found {
		return errorf("no layer found with name '%s'", name)
	}
	return nil
}

// UnfreezeByName unfreezes all layers with the given name
func (n *Network) UnfreezeByName(name string) error {
	found := false
	for i, layer := range n.layers {
		if layer.name() == name {
			if err := n.Unfreeze(i); err != nil {
				return err
			}
			found = true
		}
	}
	if !found {
		return errorf("no layer found with name '%s'", name)
	}
	return nil
}

// getTrainableParamsAndGrads returns only the parameters and gradients
// from non-frozen layers
func (n *Network) getTrainableParamsAndGrads() ([]*tensor, []*tensor) {
	var params []*tensor
	var grads []*tensor

	for i, layer := range n.layers {
		if !n.IsFrozen(i) {
			params = append(params, layer.parameters()...)
			grads = append(grads, layer.gradients()...)
		}
	}

	return params, grads
}

// FreezeExcept freezes all layers except those at the specified indices
func (n *Network) FreezeExcept(indices ...int) error {
	// Create a set of exceptions
	exceptions := make(map[int]bool)
	for _, idx := range indices {
		if idx < 0 || idx >= len(n.layers) {
			return errorf("layer index %d out of range [0, %d)", idx, len(n.layers))
		}
		exceptions[idx] = true
	}

	// Freeze all layers not in the exception set
	for i := range n.layers {
		if !exceptions[i] {
			if err := n.Freeze(i); err != nil {
				return err
			}
		}
	}
	return nil
}

// FreezeSummary returns a human-readable summary of frozen/unfrozen layers
func (n *Network) FreezeSummary() string {
	result := "Layer Freeze Status\n"
	result += "===================\n"

	trainableParams := 0
	frozenParams := 0

	for i, layer := range n.layers {
		params := layer.parameters()
		paramCount := 0
		for _, p := range params {
			paramCount += p.size()
		}

		status := "trainable"
		if n.IsFrozen(i) {
			status = "FROZEN"
			frozenParams += paramCount
		} else {
			trainableParams += paramCount
		}

		result += fmt.Sprintf("Layer %d: %-15s %d params [%s]\n", i, layer.name(), paramCount, status)
	}

	result += "===================\n"
	result += fmt.Sprintf("Trainable params: %d\n", trainableParams)
	result += fmt.Sprintf("Frozen params:    %d\n", frozenParams)
	result += fmt.Sprintf("Total params:     %d\n", trainableParams+frozenParams)

	return result
}

// validateFreezeState checks that frozen state is valid
func (n *Network) validateFreezeState() error {
	if n.frozenState == nil {
		return nil
	}

	for idx := range n.frozenState.frozen {
		if idx < 0 || idx >= len(n.layers) {
			return errors.New("flow: invalid frozen layer index detected")
		}
	}
	return nil
}

// zeroFrozenGradients zeroes out gradients for all frozen layers
// This is called before the optimizer step to prevent frozen layer updates
func (n *Network) zeroFrozenGradients() {
	for i, layer := range n.layers {
		if n.IsFrozen(i) {
			for _, g := range layer.gradients() {
				if g != nil {
					g.zeroGrad()
				}
			}
		}
	}
}
