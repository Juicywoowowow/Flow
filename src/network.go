package flow

import (
	"encoding/json"
	"errors"
	"math"
	"math/rand"
	"os"
)

// Network is the main neural network container
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

// NetworkBuilder for fluent API
type NetworkBuilder struct {
	network *Network
	err     error
}

// NewNetwork creates a new network builder
func NewNetwork(config NetworkConfig) *NetworkBuilder {
	return &NetworkBuilder{
		network: &Network{
			layers: make([]Layer, 0),
			rng:    rand.New(rand.NewSource(config.Seed)),
		},
	}
}

// AddLayer adds a layer to the network
func (n *NetworkBuilder) AddLayer(layer Layer) *NetworkBuilder {
	if n.err != nil {
		return n
	}
	n.network.layers = append(n.network.layers, layer)
	return n
}

// Build finalizes the network structure
func (n *NetworkBuilder) Build(inputShape []int) (*Network, error) {
	if n.err != nil {
		return nil, n.err
	}
	if len(n.network.layers) == 0 {
		return nil, errors.New("flow: network must have at least one layer")
	}
	if len(inputShape) == 0 {
		return nil, errors.New("flow: inputShape must be specified")
	}

	n.network.inputShape = inputShape

	// Build each layer
	currentShape := inputShape
	for i, layer := range n.network.layers {
		err := layer.build(currentShape, n.network.rng)
		if err != nil {
			return nil, errorf("layer %d (%s): %v", i, layer.name(), err)
		}
		outShape := layer.outputShape()
		if outShape != nil {
			currentShape = outShape
		}
	}

	n.network.built = true
	return n.network, nil
}

// Compile configures optimizer, loss, and metrics
func (n *Network) Compile(config CompileConfig) error {
	if !n.built {
		return errors.New("flow: network must be built before compiling")
	}
	if err := ValidateCompileConfig(config); err != nil {
		return err
	}

	n.optimizer = config.Optimizer
	n.loss = config.Loss
	n.metrics = config.Metrics
	n.regularizer = config.Regularizer
	n.gradClip = config.GradientClip
	n.compiled = true

	return nil
}

// TrainResult holds training output
type TrainResult struct {
	History      map[string][]float64
	FinalLoss    float64
	FinalMetrics map[string]float64
}

// Train trains the network
func (n *Network) Train(inputs [][]float64, targets [][]float64, config TrainConfig, callbacks []Callback) (*TrainResult, error) {
	if !n.compiled {
		return nil, errors.New("flow: network must be compiled before training")
	}
	if err := ValidateTrainConfig(config); err != nil {
		return nil, err
	}

	numSamples := len(inputs)
	if numSamples == 0 {
		return nil, errors.New("flow: no training data provided")
	}
	if len(inputs) != len(targets) {
		return nil, errors.New("flow: inputs and targets must have same length")
	}

	inputDim := len(inputs[0])
	targetDim := len(targets[0])

	// Convert to internal tensors - use inputShape for proper dimensions
	var inputTensor *tensor
	if len(n.inputShape) == 1 {
		// 1D input (dense network)
		inputTensor = newTensor(numSamples, inputDim)
	} else if len(n.inputShape) == 2 {
		// 2D input (sequence: seqLen, features)
		inputTensor = newTensor(numSamples, n.inputShape[0], n.inputShape[1])
	} else if len(n.inputShape) == 3 {
		// 3D input (image: H, W, C)
		inputTensor = newTensor(numSamples, n.inputShape[0], n.inputShape[1], n.inputShape[2])
	} else {
		inputTensor = newTensor(numSamples, inputDim)
	}

	targetTensor := newTensor(numSamples, targetDim)
	for i, row := range inputs {
		for j, val := range row {
			inputTensor.data[i*inputDim+j] = val
		}
	}
	for i, row := range targets {
		for j, val := range row {
			targetTensor.data[i*targetDim+j] = val
		}
	}

	// Split validation data
	var trainX, trainY, valX, valY *tensor
	if config.ValidationSplit > 0 {
		trainX, trainY, valX, valY = splitData(inputTensor, targetTensor, config.ValidationSplit)
	} else {
		trainX, trainY = inputTensor, targetTensor
	}

	result := &TrainResult{
		History:      make(map[string][]float64),
		FinalMetrics: make(map[string]float64),
	}

	logs := make(map[string]float64)

	// Training callbacks
	for _, cb := range callbacks {
		cb.onTrainBegin(logs)
	}

	trainSize := trainX.shape[0]
	numBatches := (trainSize + config.BatchSize - 1) / config.BatchSize

	// Get all parameters
	var params []*tensor
	var grads []*tensor
	for _, layer := range n.layers {
		params = append(params, layer.parameters()...)
		grads = append(grads, layer.gradients()...)
	}

	for epoch := 0; epoch < config.Epochs; epoch++ {
		for _, cb := range callbacks {
			cb.onEpochBegin(epoch, logs)
		}

		if config.Shuffle {
			shuffleData(trainX, trainY, n.rng)
		}

		epochLoss := 0.0
		for _, m := range n.metrics {
			m.reset()
		}

		for batch := 0; batch < numBatches; batch++ {
			for _, cb := range callbacks {
				cb.onBatchBegin(batch, logs)
			}

			start := batch * config.BatchSize
			batchX := getBatch(trainX, start, config.BatchSize)
			batchY := getBatch(trainY, start, config.BatchSize)

			// Forward pass
			output := batchX
			var err error
			for _, layer := range n.layers {
				output, err = layer.forward(output, true)
				if err != nil {
					return nil, err
				}
			}

			// Compute loss
			batchLoss := n.loss.compute(output, batchY)

			// Add regularization loss
			for _, layer := range n.layers {
				for _, p := range layer.parameters() {
					batchLoss += n.regularizer.loss(p)
				}
			}

			epochLoss += batchLoss

			// Update metrics
			for _, m := range n.metrics {
				m.update(output, batchY)
			}

			// Backward pass
			gradOutput := newTensor(output.shape...)
			n.loss.gradient(output, batchY, gradOutput)

			for i := len(n.layers) - 1; i >= 0; i-- {
				gradOutput, err = n.layers[i].backward(gradOutput)
				if err != nil {
					return nil, err
				}
			}

			// Apply regularization gradient
			for _, layer := range n.layers {
				layerGrads := layer.gradients()
				layerParams := layer.parameters()
				for j := range layerParams {
					if layerGrads[j] != nil {
						n.regularizer.gradient(layerParams[j], layerGrads[j])
					}
				}
			}

			// Gradient clipping
			if n.gradClip.Mode == "norm" {
				totalNorm := 0.0
				for _, g := range grads {
					if g != nil {
						norm := l2Norm(g)
						totalNorm += norm * norm
					}
				}
				totalNorm = math.Sqrt(totalNorm)
				if totalNorm > n.gradClip.MaxNorm {
					scale := n.gradClip.MaxNorm / totalNorm
					for _, g := range grads {
						if g != nil {
							mulScalar(g, scale)
						}
					}
				}
			} else if n.gradClip.Mode == "value" {
				for _, g := range grads {
					if g != nil {
						clip(g, -n.gradClip.MaxValue, n.gradClip.MaxValue)
					}
				}
			}

			// Optimizer step
			n.optimizer.step(params, grads)

			for _, cb := range callbacks {
				cb.onBatchEnd(batch, logs)
			}
		}

		// Epoch logs
		logs["loss"] = epochLoss / float64(numBatches)
		for _, m := range n.metrics {
			logs[m.name()] = m.result()
		}

		// Validation
		if valX != nil {
			valOutput := valX
			var err error
			for _, layer := range n.layers {
				valOutput, err = layer.forward(valOutput, false)
				if err != nil {
					return nil, err
				}
			}
			logs["val_loss"] = n.loss.compute(valOutput, valY)
			for _, m := range n.metrics {
				m.reset()
				m.update(valOutput, valY)
				logs["val_"+m.name()] = m.result()
			}
		}

		// Save to history
		for k, v := range logs {
			result.History[k] = append(result.History[k], v)
		}

		// Epoch end callbacks
		stopTraining := false
		for _, cb := range callbacks {
			if cb.onEpochEnd(epoch, logs) {
				stopTraining = true
			}
		}

		if stopTraining {
			break
		}
	}

	for _, cb := range callbacks {
		cb.onTrainEnd(logs)
	}

	result.FinalLoss = logs["loss"]
	for _, m := range n.metrics {
		result.FinalMetrics[m.name()] = logs[m.name()]
	}

	return result, nil
}

// Predict runs inference on inputs
func (n *Network) Predict(inputs [][]float64) ([][]float64, error) {
	if !n.compiled {
		return nil, errors.New("flow: network must be compiled before prediction")
	}

	numSamples := len(inputs)
	inputDim := len(inputs[0])

	// Convert to internal tensors - use inputShape for proper dimensions
	var inputTensor *tensor
	if len(n.inputShape) == 1 {
		inputTensor = newTensor(numSamples, inputDim)
	} else if len(n.inputShape) == 2 {
		inputTensor = newTensor(numSamples, n.inputShape[0], n.inputShape[1])
	} else if len(n.inputShape) == 3 {
		inputTensor = newTensor(numSamples, n.inputShape[0], n.inputShape[1], n.inputShape[2])
	} else {
		inputTensor = newTensor(numSamples, inputDim)
	}

	for i, row := range inputs {
		for j, val := range row {
			inputTensor.data[i*inputDim+j] = val
		}
	}

	output := inputTensor
	var err error
	for _, layer := range n.layers {
		output, err = layer.forward(output, false)
		if err != nil {
			return nil, err
		}
	}

	// Convert back to [][]float64
	outputDim := output.shape[1]
	result := make([][]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		result[i] = make([]float64, outputDim)
		for j := 0; j < outputDim; j++ {
			result[i][j] = output.data[i*outputDim+j]
		}
	}

	return result, nil
}

// Evaluate runs evaluation on test data
func (n *Network) Evaluate(inputs [][]float64, targets [][]float64) (map[string]float64, error) {
	if !n.compiled {
		return nil, errors.New("flow: network must be compiled before evaluation")
	}

	numSamples := len(inputs)
	inputDim := len(inputs[0])
	targetDim := len(targets[0])

	var inputTensor *tensor
	if len(n.inputShape) == 1 {
		inputTensor = newTensor(numSamples, inputDim)
	} else if len(n.inputShape) == 2 {
		inputTensor = newTensor(numSamples, n.inputShape[0], n.inputShape[1])
	} else if len(n.inputShape) == 3 {
		inputTensor = newTensor(numSamples, n.inputShape[0], n.inputShape[1], n.inputShape[2])
	} else {
		inputTensor = newTensor(numSamples, inputDim)
	}

	targetTensor := newTensor(numSamples, targetDim)

	for i, row := range inputs {
		for j, val := range row {
			inputTensor.data[i*inputDim+j] = val
		}
	}
	for i, row := range targets {
		for j, val := range row {
			targetTensor.data[i*targetDim+j] = val
		}
	}

	output := inputTensor
	var err error
	for _, layer := range n.layers {
		output, err = layer.forward(output, false)
		if err != nil {
			return nil, err
		}
	}

	results := make(map[string]float64)
	results["loss"] = n.loss.compute(output, targetTensor)

	for _, m := range n.metrics {
		m.reset()
		m.update(output, targetTensor)
		results[m.name()] = m.result()
	}

	return results, nil
}

// ModelState for serialization
type ModelState struct {
	Weights [][][]float64 `json:"weights"`
	Shapes  [][]int       `json:"shapes"`
}

// Save saves model weights to file
func (n *Network) Save(path string) error {
	state := ModelState{
		Weights: make([][][]float64, 0),
		Shapes:  make([][]int, 0),
	}

	for _, layer := range n.layers {
		for _, p := range layer.parameters() {
			data := make([]float64, len(p.data))
			copy(data, p.data)
			state.Weights = append(state.Weights, [][]float64{data})
			state.Shapes = append(state.Shapes, p.shape)
		}
	}

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	return encoder.Encode(state)
}

// Load loads model weights from file
func (n *Network) Load(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	var state ModelState
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&state); err != nil {
		return err
	}

	idx := 0
	for _, layer := range n.layers {
		for _, p := range layer.parameters() {
			if idx >= len(state.Weights) {
				return errors.New("flow: weight count mismatch")
			}
			copy(p.data, state.Weights[idx][0])
			idx++
		}
	}

	return nil
}

// Summary prints network architecture
func (n *Network) Summary() string {
	result := "Flow Network Summary\n"
	result += "====================\n"

	totalParams := 0
	for i, layer := range n.layers {
		params := layer.parameters()
		layerParams := 0
		for _, p := range params {
			layerParams += p.size()
		}
		totalParams += layerParams
		result += errorf("Layer %d: %s - %d params\n", i+1, layer.name(), layerParams).Error()[6:]
	}
	result += "====================\n"
	result += errorf("Total parameters: %d\n", totalParams).Error()[6:]

	return result
}
