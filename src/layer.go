package flow

import (
	"errors"
	"math/rand"
)

// Layer is the base interface for all layers
type Layer interface {
	forward(input *tensor, training bool) (*tensor, error)
	backward(gradOutput *tensor) (*tensor, error)
	parameters() []*tensor
	gradients() []*tensor
	build(inputShape []int, rng *rand.Rand) error
	outputShape() []int
	name() string
}

// DenseLayer - fully connected layer
type DenseLayer struct {
	units       int
	activation  Activation
	initializer Initializer
	biasInit    Initializer
	useBias     bool
	weights     *tensor
	bias        *tensor
	input       *tensor
	preAct      *tensor
	output      *tensor
	gradW       *tensor
	gradB       *tensor
	inputShape  []int
	built       bool
}

// DenseBuilder for fluent API
type DenseBuilder struct {
	layer *DenseLayer
}

func Dense(units int) *DenseBuilder {
	return &DenseBuilder{
		layer: &DenseLayer{
			units: units,
		},
	}
}

func (b *DenseBuilder) WithActivation(act Activation) *DenseBuilder {
	b.layer.activation = act
	return b
}

func (b *DenseBuilder) WithInitializer(init Initializer) *DenseBuilder {
	b.layer.initializer = init
	return b
}

func (b *DenseBuilder) WithBiasInitializer(init Initializer) *DenseBuilder {
	b.layer.biasInit = init
	return b
}

func (b *DenseBuilder) WithBias(useBias bool) *DenseBuilder {
	b.layer.useBias = useBias
	return b
}

func (b *DenseBuilder) Build() Layer {
	return b.layer
}

func (d *DenseLayer) build(inputShape []int, rng *rand.Rand) error {
	if len(inputShape) == 0 {
		return errors.New("flow: DenseLayer requires non-empty input shape")
	}
	if d.initializer == nil {
		return errors.New("flow: DenseLayer requires initializer - use WithInitializer()")
	}
	if d.activation == nil {
		return errors.New("flow: DenseLayer requires activation - use WithActivation()")
	}
	if d.useBias && d.biasInit == nil {
		return errors.New("flow: DenseLayer with bias requires bias initializer - use WithBiasInitializer()")
	}

	fanIn := inputShape[len(inputShape)-1]
	d.inputShape = inputShape

	d.weights = newTensor(fanIn, d.units)
	d.initializer.initialize(d.weights, fanIn, d.units, rng)

	d.gradW = newTensor(fanIn, d.units)

	if d.useBias {
		d.bias = newTensor(d.units)
		d.biasInit.initialize(d.bias, fanIn, d.units, rng)
		d.gradB = newTensor(d.units)
	}

	d.built = true
	return nil
}

func (d *DenseLayer) forward(input *tensor, training bool) (*tensor, error) {
	if !d.built {
		return nil, errors.New("flow: layer not built - call Build() first")
	}
	batchSize := input.shape[0]
	inputDim := input.shape[1]

	if inputDim != d.weights.shape[0] {
		return nil, errors.New("flow: input dimension mismatch")
	}

	d.input = input
	d.preAct = newTensor(batchSize, d.units)
	d.output = newTensor(batchSize, d.units)

	// Y = X @ W
	matmul(input, d.weights, d.preAct)

	// Y = Y + b
	if d.useBias {
		addVec(d.preAct, d.bias)
	}

	// Y = activation(Y)
	d.activation.forward(d.preAct, d.output)

	return d.output, nil
}

func (d *DenseLayer) backward(gradOutput *tensor) (*tensor, error) {
	if d.input == nil {
		return nil, errors.New("flow: backward called before forward")
	}

	batchSize := d.input.shape[0]

	// Gradient through activation
	gradPreAct := newTensor(gradOutput.shape...)
	d.activation.backward(d.preAct, gradOutput, gradPreAct)

	// Gradient w.r.t. weights: dL/dW = X^T @ dL/dY
	d.gradW.zeroGrad()
	matmulTransA(d.input, gradPreAct, d.gradW)
	// Average over batch
	mulScalar(d.gradW, 1.0/float64(batchSize))

	// Gradient w.r.t. bias: dL/db = sum(dL/dY, axis=0)
	if d.useBias {
		d.gradB.zeroGrad()
		sumAxis0(gradPreAct, d.gradB)
		mulScalar(d.gradB, 1.0/float64(batchSize))
	}

	// Gradient w.r.t. input: dL/dX = dL/dY @ W^T
	gradInput := newTensor(d.input.shape...)
	matmulTransB(gradPreAct, d.weights, gradInput)

	return gradInput, nil
}

func (d *DenseLayer) parameters() []*tensor {
	if d.useBias {
		return []*tensor{d.weights, d.bias}
	}
	return []*tensor{d.weights}
}

func (d *DenseLayer) gradients() []*tensor {
	if d.useBias {
		return []*tensor{d.gradW, d.gradB}
	}
	return []*tensor{d.gradW}
}

func (d *DenseLayer) outputShape() []int {
	return []int{d.units}
}

func (d *DenseLayer) name() string { return "dense" }

// DropoutLayer - randomly zeros elements during training
type DropoutLayer struct {
	rate  float64
	seed  int64
	mask  *tensor
	rng   *rand.Rand
	built bool
}

type DropoutBuilder struct {
	layer *DropoutLayer
}

func Dropout(rate float64) *DropoutBuilder {
	return &DropoutBuilder{
		layer: &DropoutLayer{
			rate: rate,
		},
	}
}

func (b *DropoutBuilder) WithSeed(seed int64) *DropoutBuilder {
	b.layer.seed = seed
	return b
}

func (b *DropoutBuilder) Build() Layer {
	return b.layer
}

func (d *DropoutLayer) build(inputShape []int, rng *rand.Rand) error {
	if d.rate < 0 || d.rate >= 1 {
		return errors.New("flow: dropout rate must be in [0, 1)")
	}
	d.rng = rng
	d.built = true
	return nil
}

func (d *DropoutLayer) forward(input *tensor, training bool) (*tensor, error) {
	if !training {
		return input.clone(), nil
	}

	output := newTensor(input.shape...)
	d.mask = newTensor(input.shape...)

	scale := 1.0 / (1.0 - d.rate)
	for i := range input.data {
		if d.rng.Float64() >= d.rate {
			d.mask.data[i] = scale
			output.data[i] = input.data[i] * scale
		} else {
			d.mask.data[i] = 0
			output.data[i] = 0
		}
	}
	return output, nil
}

func (d *DropoutLayer) backward(gradOutput *tensor) (*tensor, error) {
	gradInput := newTensor(gradOutput.shape...)
	elemMul(gradOutput, d.mask, gradInput)
	return gradInput, nil
}

func (d *DropoutLayer) parameters() []*tensor { return nil }
func (d *DropoutLayer) gradients() []*tensor  { return nil }
func (d *DropoutLayer) outputShape() []int    { return nil }
func (d *DropoutLayer) name() string          { return "dropout" }

// FlattenLayer - flattens input to 1D (per sample)
type FlattenLayer struct {
	inputShape []int
	built      bool
}

type FlattenBuilder struct {
	layer *FlattenLayer
}

func Flatten() *FlattenBuilder {
	return &FlattenBuilder{
		layer: &FlattenLayer{},
	}
}

func (b *FlattenBuilder) Build() Layer {
	return b.layer
}

func (f *FlattenLayer) build(inputShape []int, rng *rand.Rand) error {
	f.inputShape = inputShape
	f.built = true
	return nil
}

func (f *FlattenLayer) forward(input *tensor, training bool) (*tensor, error) {
	batchSize := input.shape[0]
	flatSize := 1
	for _, s := range input.shape[1:] {
		flatSize *= s
	}
	output := newTensor(batchSize, flatSize)
	copy(output.data, input.data)
	return output, nil
}

func (f *FlattenLayer) backward(gradOutput *tensor) (*tensor, error) {
	shape := append([]int{gradOutput.shape[0]}, f.inputShape...)
	gradInput := newTensor(shape...)
	copy(gradInput.data, gradOutput.data)
	return gradInput, nil
}

func (f *FlattenLayer) parameters() []*tensor { return nil }
func (f *FlattenLayer) gradients() []*tensor  { return nil }

func (f *FlattenLayer) outputShape() []int {
	flatSize := 1
	for _, s := range f.inputShape {
		flatSize *= s
	}
	return []int{flatSize}
}

func (f *FlattenLayer) name() string { return "flatten" }

// BatchNormLayer - batch normalization
type BatchNormLayer struct {
	epsilon     float64
	momentum    float64
	gamma       *tensor
	beta        *tensor
	runningMean *tensor
	runningVar  *tensor
	gradGamma   *tensor
	gradBeta    *tensor
	input       *tensor
	normalized  *tensor
	mean        *tensor
	variance    *tensor
	features    int
	built       bool
}

type BatchNormBuilder struct {
	layer *BatchNormLayer
}

func BatchNorm(epsilon, momentum float64) *BatchNormBuilder {
	return &BatchNormBuilder{
		layer: &BatchNormLayer{
			epsilon:  epsilon,
			momentum: momentum,
		},
	}
}

func (b *BatchNormBuilder) Build() Layer {
	return b.layer
}

func (bn *BatchNormLayer) build(inputShape []int, rng *rand.Rand) error {
	if len(inputShape) == 0 {
		return errors.New("flow: BatchNorm requires non-empty input shape")
	}
	bn.features = inputShape[len(inputShape)-1]

	bn.gamma = newTensor(bn.features)
	bn.gamma.fill(1.0)
	bn.beta = newTensor(bn.features)
	bn.beta.fill(0.0)

	bn.runningMean = newTensor(bn.features)
	bn.runningVar = newTensor(bn.features)
	bn.runningVar.fill(1.0)

	bn.gradGamma = newTensor(bn.features)
	bn.gradBeta = newTensor(bn.features)

	bn.built = true
	return nil
}

func (bn *BatchNormLayer) forward(input *tensor, training bool) (*tensor, error) {
	if !bn.built {
		return nil, errors.New("flow: layer not built")
	}

	batchSize := input.shape[0]
	features := input.shape[1]

	bn.input = input
	bn.normalized = newTensor(input.shape...)
	bn.mean = newTensor(features)
	bn.variance = newTensor(features)

	if training {
		// Compute batch mean
		for j := 0; j < features; j++ {
			sum := 0.0
			for i := 0; i < batchSize; i++ {
				sum += input.data[i*features+j]
			}
			bn.mean.data[j] = sum / float64(batchSize)
		}

		// Compute batch variance
		for j := 0; j < features; j++ {
			sum := 0.0
			for i := 0; i < batchSize; i++ {
				diff := input.data[i*features+j] - bn.mean.data[j]
				sum += diff * diff
			}
			bn.variance.data[j] = sum / float64(batchSize)
		}

		// Update running stats
		for j := 0; j < features; j++ {
			bn.runningMean.data[j] = bn.momentum*bn.runningMean.data[j] + (1-bn.momentum)*bn.mean.data[j]
			bn.runningVar.data[j] = bn.momentum*bn.runningVar.data[j] + (1-bn.momentum)*bn.variance.data[j]
		}
	} else {
		copy(bn.mean.data, bn.runningMean.data)
		copy(bn.variance.data, bn.runningVar.data)
	}

	// Normalize and scale
	output := newTensor(input.shape...)
	for i := 0; i < batchSize; i++ {
		for j := 0; j < features; j++ {
			idx := i*features + j
			xNorm := (input.data[idx] - bn.mean.data[j]) / (bn.variance.data[j] + bn.epsilon)
			bn.normalized.data[idx] = xNorm
			output.data[idx] = bn.gamma.data[j]*xNorm + bn.beta.data[j]
		}
	}

	return output, nil
}

func (bn *BatchNormLayer) backward(gradOutput *tensor) (*tensor, error) {
	batchSize := bn.input.shape[0]
	features := bn.input.shape[1]
	N := float64(batchSize)

	bn.gradGamma.zeroGrad()
	bn.gradBeta.zeroGrad()

	// Gradients w.r.t. gamma and beta
	for j := 0; j < features; j++ {
		for i := 0; i < batchSize; i++ {
			idx := i*features + j
			bn.gradGamma.data[j] += gradOutput.data[idx] * bn.normalized.data[idx]
			bn.gradBeta.data[j] += gradOutput.data[idx]
		}
	}

	// Gradient w.r.t. input
	gradInput := newTensor(bn.input.shape...)

	for j := 0; j < features; j++ {
		std := bn.variance.data[j] + bn.epsilon

		var dxNorm, dVar, dMean float64

		// Sum of dL/dy * gamma
		for i := 0; i < batchSize; i++ {
			dxNorm += gradOutput.data[i*features+j] * bn.gamma.data[j]
		}

		// Gradient w.r.t. variance
		for i := 0; i < batchSize; i++ {
			idx := i*features + j
			dVar += gradOutput.data[idx] * bn.gamma.data[j] * (bn.input.data[idx] - bn.mean.data[j]) * (-0.5) / (std * std)
		}

		// Gradient w.r.t. mean
		for i := 0; i < batchSize; i++ {
			idx := i*features + j
			dMean += gradOutput.data[idx] * bn.gamma.data[j] * (-1.0 / std)
			dMean += dVar * (-2.0 * (bn.input.data[idx] - bn.mean.data[j]) / N)
		}

		for i := 0; i < batchSize; i++ {
			idx := i*features + j
			gradInput.data[idx] = gradOutput.data[idx]*bn.gamma.data[j]/std +
				dVar*2*(bn.input.data[idx]-bn.mean.data[j])/N +
				dMean/N
		}
	}

	return gradInput, nil
}

func (bn *BatchNormLayer) parameters() []*tensor {
	return []*tensor{bn.gamma, bn.beta}
}

func (bn *BatchNormLayer) gradients() []*tensor {
	return []*tensor{bn.gradGamma, bn.gradBeta}
}

func (bn *BatchNormLayer) outputShape() []int { return []int{bn.features} }
func (bn *BatchNormLayer) name() string       { return "batch_norm" }
