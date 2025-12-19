package flow

import (
	"errors"
	"math"
	"math/rand"
)

// =============================================================================
// EMBEDDING LAYER
// Maps integer indices to dense vectors (lookup table)
// =============================================================================

type EmbeddingLayer struct {
	vocabSize   int
	embedDim    int
	initializer Initializer
	paddingIdx  int // Index to keep zero (no gradient), -1 if none
	seqLen      int // Stored from input shape

	weights  *tensor // [vocabSize, embedDim]
	gradW    *tensor
	inputIdx []int // Cached input indices for backward
	built    bool
}

type EmbeddingBuilder struct {
	layer *EmbeddingLayer
}

// Embedding creates an embedding layer
// Input: integer indices [batch, seqLen] (flattened as []float64)
// Output: dense vectors [batch, seqLen, embedDim]
func Embedding(vocabSize, embedDim int) *EmbeddingBuilder {
	return &EmbeddingBuilder{
		layer: &EmbeddingLayer{
			vocabSize:  vocabSize,
			embedDim:   embedDim,
			paddingIdx: -1,
		},
	}
}

func (b *EmbeddingBuilder) WithInitializer(init Initializer) *EmbeddingBuilder {
	b.layer.initializer = init
	return b
}

func (b *EmbeddingBuilder) WithPaddingIdx(idx int) *EmbeddingBuilder {
	b.layer.paddingIdx = idx
	return b
}

func (b *EmbeddingBuilder) Build() Layer {
	return b.layer
}

func (e *EmbeddingLayer) build(inputShape []int, rng *rand.Rand) error {
	if e.initializer == nil {
		// Default: normal distribution with small std
		e.initializer = RandomNormal(0, 0.02)
	}

	// Store sequence length from input shape
	if len(inputShape) >= 1 {
		e.seqLen = inputShape[0]
	}

	e.weights = newTensor(e.vocabSize, e.embedDim)
	e.initializer.initialize(e.weights, e.vocabSize, e.embedDim, rng)

	// Zero out padding index if specified
	if e.paddingIdx >= 0 && e.paddingIdx < e.vocabSize {
		for j := 0; j < e.embedDim; j++ {
			e.weights.data[e.paddingIdx*e.embedDim+j] = 0
		}
	}

	e.gradW = newTensor(e.vocabSize, e.embedDim)
	e.built = true
	return nil
}

func (e *EmbeddingLayer) forward(input *tensor, training bool) (*tensor, error) {
	if !e.built {
		return nil, errors.New("flow: Embedding layer not built")
	}

	// Input is assumed to be [batch, seqLen] with integer values as float64
	batchSize := input.shape[0]
	seqLen := input.shape[1]

	// Cache indices for backward pass
	e.inputIdx = make([]int, batchSize*seqLen)
	for i := range e.inputIdx {
		e.inputIdx[i] = int(input.data[i])
	}

	// Output: [batch, seqLen, embedDim]
	output := newTensor(batchSize, seqLen, e.embedDim)

	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			idx := e.inputIdx[b*seqLen+s]
			if idx < 0 || idx >= e.vocabSize {
				idx = 0 // Fallback to index 0 for out-of-range
			}
			for d := 0; d < e.embedDim; d++ {
				output.data[b*seqLen*e.embedDim+s*e.embedDim+d] = e.weights.data[idx*e.embedDim+d]
			}
		}
	}

	return output, nil
}

func (e *EmbeddingLayer) backward(gradOutput *tensor) (*tensor, error) {
	batchSize := gradOutput.shape[0]
	seqLen := gradOutput.shape[1]

	e.gradW.zeroGrad()

	// Sparse update: only update rows that were accessed
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			idx := e.inputIdx[b*seqLen+s]
			if idx < 0 || idx >= e.vocabSize {
				continue
			}
			// Skip padding index
			if idx == e.paddingIdx {
				continue
			}
			// Accumulate gradient for this index
			for d := 0; d < e.embedDim; d++ {
				e.gradW.data[idx*e.embedDim+d] += gradOutput.data[b*seqLen*e.embedDim+s*e.embedDim+d]
			}
		}
	}

	// Scale by batch size
	scaleFactor := 1.0 / float64(batchSize)
	mulScalar(e.gradW, scaleFactor)

	// Input gradient is not needed (indices are not differentiable)
	return nil, nil
}

func (e *EmbeddingLayer) parameters() []*tensor {
	return []*tensor{e.weights}
}

func (e *EmbeddingLayer) gradients() []*tensor {
	return []*tensor{e.gradW}
}

func (e *EmbeddingLayer) outputShape() []int {
	return []int{e.seqLen, e.embedDim}
}

func (e *EmbeddingLayer) name() string { return "embedding" }

// =============================================================================
// POSITIONAL ENCODING LAYER
// Adds position information to embeddings using sinusoidal functions
// =============================================================================

type PositionalEncodingLayer struct {
	maxLen     int
	embedDim   int
	dropout    float64
	encodings  *tensor // Precomputed [maxLen, embedDim]
	learned    bool    // If true, use learned embeddings instead of sinusoidal
	rng        *rand.Rand
	built      bool
	inputShape []int

	// For learned positional encoding
	weights *tensor
	gradW   *tensor
}

type PositionalEncodingBuilder struct {
	layer *PositionalEncodingLayer
}

// PositionalEncoding creates sinusoidal positional encoding
func PositionalEncoding(maxLen, embedDim int) *PositionalEncodingBuilder {
	return &PositionalEncodingBuilder{
		layer: &PositionalEncodingLayer{
			maxLen:   maxLen,
			embedDim: embedDim,
			learned:  false,
		},
	}
}

func (b *PositionalEncodingBuilder) WithDropout(rate float64) *PositionalEncodingBuilder {
	b.layer.dropout = rate
	return b
}

func (b *PositionalEncodingBuilder) WithLearned(learned bool) *PositionalEncodingBuilder {
	b.layer.learned = learned
	return b
}

func (b *PositionalEncodingBuilder) Build() Layer {
	return b.layer
}

func (pe *PositionalEncodingLayer) build(inputShape []int, rng *rand.Rand) error {
	pe.rng = rng
	pe.inputShape = inputShape

	if pe.learned {
		// Learned positional embeddings
		pe.weights = newTensor(pe.maxLen, pe.embedDim)
		init := RandomNormal(0, 0.02)
		init.initialize(pe.weights, pe.maxLen, pe.embedDim, rng)
		pe.gradW = newTensor(pe.maxLen, pe.embedDim)
	} else {
		// Sinusoidal positional encoding (fixed, no parameters)
		pe.encodings = newTensor(pe.maxLen, pe.embedDim)

		for pos := 0; pos < pe.maxLen; pos++ {
			for i := 0; i < pe.embedDim; i++ {
				// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
				// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
				angle := float64(pos) / math.Pow(10000.0, float64(2*(i/2))/float64(pe.embedDim))
				if i%2 == 0 {
					pe.encodings.data[pos*pe.embedDim+i] = math.Sin(angle)
				} else {
					pe.encodings.data[pos*pe.embedDim+i] = math.Cos(angle)
				}
			}
		}
	}

	pe.built = true
	return nil
}

func (pe *PositionalEncodingLayer) forward(input *tensor, training bool) (*tensor, error) {
	if !pe.built {
		return nil, errors.New("flow: PositionalEncoding not built")
	}

	// Input: [batch, seqLen, embedDim]
	batchSize := input.shape[0]
	seqLen := input.shape[1]
	embedDim := input.shape[2]

	if seqLen > pe.maxLen {
		return nil, errors.New("flow: sequence length exceeds maxLen for positional encoding")
	}

	output := newTensor(batchSize, seqLen, embedDim)

	// Add positional encoding to input
	var encoding *tensor
	if pe.learned {
		encoding = pe.weights
	} else {
		encoding = pe.encodings
	}

	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			for d := 0; d < embedDim; d++ {
				val := input.data[b*seqLen*embedDim+s*embedDim+d] + encoding.data[s*pe.embedDim+d]

				// Apply dropout during training
				if training && pe.dropout > 0 && pe.rng.Float64() < pe.dropout {
					val = 0
				} else if pe.dropout > 0 {
					val = val / (1 - pe.dropout) // Inverted dropout scaling
				}

				output.data[b*seqLen*embedDim+s*embedDim+d] = val
			}
		}
	}

	return output, nil
}

func (pe *PositionalEncodingLayer) backward(gradOutput *tensor) (*tensor, error) {
	// For sinusoidal (fixed) encoding, just pass gradients through
	// For learned encoding, accumulate gradients

	if pe.learned {
		batchSize := gradOutput.shape[0]
		seqLen := gradOutput.shape[1]
		embedDim := gradOutput.shape[2]

		pe.gradW.zeroGrad()

		for b := 0; b < batchSize; b++ {
			for s := 0; s < seqLen; s++ {
				for d := 0; d < embedDim; d++ {
					pe.gradW.data[s*pe.embedDim+d] += gradOutput.data[b*seqLen*embedDim+s*embedDim+d]
				}
			}
		}

		scaleFactor := 1.0 / float64(batchSize)
		mulScalar(pe.gradW, scaleFactor)
	}

	// Gradient passes through unchanged to input
	gradInput := gradOutput.clone()
	return gradInput, nil
}

func (pe *PositionalEncodingLayer) parameters() []*tensor {
	if pe.learned {
		return []*tensor{pe.weights}
	}
	return nil
}

func (pe *PositionalEncodingLayer) gradients() []*tensor {
	if pe.learned {
		return []*tensor{pe.gradW}
	}
	return nil
}

func (pe *PositionalEncodingLayer) outputShape() []int {
	return pe.inputShape // Preserves input shape
}

func (pe *PositionalEncodingLayer) name() string { return "positional_encoding" }

// =============================================================================
// RESIDUAL LAYER (Skip Connection)
// Wraps inner layers: output = input + F(input)
// =============================================================================

type ResidualLayer struct {
	innerLayers []Layer
	projection  Layer // Optional: used if input/output dims differ
	inputShape  []int
	built       bool
}

type ResidualBuilder struct {
	layer *ResidualLayer
}

// Residual creates a residual block that adds skip connection
// output = input + innerLayers(input)
func Residual(innerLayers ...Layer) *ResidualBuilder {
	return &ResidualBuilder{
		layer: &ResidualLayer{
			innerLayers: innerLayers,
		},
	}
}

// WithProjection adds a projection layer for dimension matching
func (b *ResidualBuilder) WithProjection(proj Layer) *ResidualBuilder {
	b.layer.projection = proj
	return b
}

func (b *ResidualBuilder) Build() Layer {
	return b.layer
}

func (r *ResidualLayer) build(inputShape []int, rng *rand.Rand) error {
	r.inputShape = inputShape

	// Build inner layers sequentially
	currentShape := inputShape
	for _, layer := range r.innerLayers {
		if err := layer.build(currentShape, rng); err != nil {
			return err
		}
		currentShape = layer.outputShape()
	}

	// Build projection if provided
	if r.projection != nil {
		if err := r.projection.build(inputShape, rng); err != nil {
			return err
		}
	}

	r.built = true
	return nil
}

func (r *ResidualLayer) forward(input *tensor, training bool) (*tensor, error) {
	if !r.built {
		return nil, errors.New("flow: Residual block not built")
	}

	// Save input for skip connection
	skipInput := input

	// Apply projection to skip if needed
	if r.projection != nil {
		var err error
		skipInput, err = r.projection.forward(input, training)
		if err != nil {
			return nil, err
		}
	}

	// Forward through inner layers
	current := input
	for _, layer := range r.innerLayers {
		var err error
		current, err = layer.forward(current, training)
		if err != nil {
			return nil, err
		}
	}

	// Add skip connection: output = F(x) + x
	output := newTensor(current.shape...)
	for i := range output.data {
		output.data[i] = current.data[i] + skipInput.data[i]
	}

	return output, nil
}

func (r *ResidualLayer) backward(gradOutput *tensor) (*tensor, error) {
	// Gradient flows through both paths
	// ∂L/∂x = ∂L/∂H * (∂F/∂x + 1)

	// Backward through inner layers (in reverse order)
	gradInner := gradOutput.clone()
	for i := len(r.innerLayers) - 1; i >= 0; i-- {
		var err error
		gradInner, err = r.innerLayers[i].backward(gradInner)
		if err != nil {
			return nil, err
		}
	}

	// Skip connection gradient
	gradSkip := gradOutput.clone()
	if r.projection != nil {
		var err error
		gradSkip, err = r.projection.backward(gradSkip)
		if err != nil {
			return nil, err
		}
	}

	// Sum gradients from both paths
	gradInput := newTensor(gradInner.shape...)
	for i := range gradInput.data {
		gradInput.data[i] = gradInner.data[i] + gradSkip.data[i]
	}

	return gradInput, nil
}

func (r *ResidualLayer) parameters() []*tensor {
	var params []*tensor
	for _, layer := range r.innerLayers {
		params = append(params, layer.parameters()...)
	}
	if r.projection != nil {
		params = append(params, r.projection.parameters()...)
	}
	return params
}

func (r *ResidualLayer) gradients() []*tensor {
	var grads []*tensor
	for _, layer := range r.innerLayers {
		grads = append(grads, layer.gradients()...)
	}
	if r.projection != nil {
		grads = append(grads, r.projection.gradients()...)
	}
	return grads
}

func (r *ResidualLayer) outputShape() []int {
	if len(r.innerLayers) > 0 {
		return r.innerLayers[len(r.innerLayers)-1].outputShape()
	}
	return r.inputShape
}

func (r *ResidualLayer) name() string { return "residual" }

// =============================================================================
// ADD LAYER
// Simple element-wise addition of multiple inputs (for explicit skip connections)
// =============================================================================

type AddLayer struct {
	inputShape []int
	built      bool
}

type AddBuilder struct {
	layer *AddLayer
}

// Add creates a layer that performs element-wise addition
// Useful for explicit residual connections in functional API
func Add() *AddBuilder {
	return &AddBuilder{
		layer: &AddLayer{},
	}
}

func (b *AddBuilder) Build() Layer {
	return b.layer
}

func (a *AddLayer) build(inputShape []int, rng *rand.Rand) error {
	a.inputShape = inputShape
	a.built = true
	return nil
}

func (a *AddLayer) forward(input *tensor, training bool) (*tensor, error) {
	// This layer expects to be used in a special way with multiple inputs
	// For now, it just passes through (identity) as we can't easily express
	// multi-input in our sequential API
	// The Residual layer above handles the actual skip-connection logic
	return input.clone(), nil
}

func (a *AddLayer) backward(gradOutput *tensor) (*tensor, error) {
	return gradOutput.clone(), nil
}

func (a *AddLayer) parameters() []*tensor { return nil }
func (a *AddLayer) gradients() []*tensor  { return nil }
func (a *AddLayer) outputShape() []int    { return a.inputShape }
func (a *AddLayer) name() string          { return "add" }
