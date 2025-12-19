package flow

import (
	"errors"
	"math"
	"math/rand"
)

// LayerNormLayer - Layer Normalization (for Transformers/RNNs)
// Normalizes across features, not batch
type LayerNormLayer struct {
	epsilon    float64
	gamma      *tensor
	beta       *tensor
	gradGamma  *tensor
	gradBeta   *tensor
	input      *tensor
	normalized *tensor
	features   int
	inputShape []int
	built      bool
}

type LayerNormBuilder struct {
	layer *LayerNormLayer
}

func LayerNorm(epsilon float64) *LayerNormBuilder {
	return &LayerNormBuilder{
		layer: &LayerNormLayer{
			epsilon: epsilon,
		},
	}
}

func (b *LayerNormBuilder) Build() Layer {
	return b.layer
}

func (ln *LayerNormLayer) build(inputShape []int, rng *rand.Rand) error {
	if len(inputShape) == 0 {
		return errors.New("flow: LayerNorm requires non-empty input shape")
	}
	ln.inputShape = inputShape
	ln.features = inputShape[len(inputShape)-1]

	ln.gamma = newTensor(ln.features)
	ln.gamma.fill(1.0)
	ln.beta = newTensor(ln.features)
	ln.beta.fill(0.0)

	ln.gradGamma = newTensor(ln.features)
	ln.gradBeta = newTensor(ln.features)

	ln.built = true
	return nil
}

func (ln *LayerNormLayer) forward(input *tensor, training bool) (*tensor, error) {
	if !ln.built {
		return nil, errors.New("flow: LayerNorm not built")
	}

	ln.input = input
	features := ln.features
	totalVectors := len(input.data) / features

	ln.normalized = newTensor(input.shape...)
	output := newTensor(input.shape...)

	for i := 0; i < totalVectors; i++ {
		// Compute mean for this vector
		mean := 0.0
		baseIdx := i * features
		for j := 0; j < features; j++ {
			mean += input.data[baseIdx+j]
		}
		mean /= float64(features)

		// Compute variance
		variance := 0.0
		for j := 0; j < features; j++ {
			diff := input.data[baseIdx+j] - mean
			variance += diff * diff
		}
		variance /= float64(features)

		// Normalize and scale
		std := math.Sqrt(variance + ln.epsilon)
		for j := 0; j < features; j++ {
			idx := baseIdx + j
			xNorm := (input.data[idx] - mean) / std
			ln.normalized.data[idx] = xNorm
			output.data[idx] = ln.gamma.data[j]*xNorm + ln.beta.data[j]
		}
	}

	return output, nil
}

func (ln *LayerNormLayer) backward(gradOutput *tensor) (*tensor, error) {
	features := ln.features
	totalVectors := len(ln.input.data) / features
	N := float64(features)

	ln.gradGamma.zeroGrad()
	ln.gradBeta.zeroGrad()

	// Gradients w.r.t. gamma and beta
	for i := 0; i < totalVectors; i++ {
		baseIdx := i * features
		for j := 0; j < features; j++ {
			idx := baseIdx + j
			ln.gradGamma.data[j] += gradOutput.data[idx] * ln.normalized.data[idx]
			ln.gradBeta.data[j] += gradOutput.data[idx]
		}
	}

	gradInput := newTensor(ln.input.shape...)

	for i := 0; i < totalVectors; i++ {
		baseIdx := i * features

		// Recompute mean and variance for backward
		mean := 0.0
		for j := 0; j < features; j++ {
			mean += ln.input.data[baseIdx+j]
		}
		mean /= N

		variance := 0.0
		for j := 0; j < features; j++ {
			diff := ln.input.data[baseIdx+j] - mean
			variance += diff * diff
		}
		variance /= N
		std := math.Sqrt(variance + ln.epsilon)

		// Compute intermediate gradients
		dxNorm := make([]float64, features)
		for j := 0; j < features; j++ {
			dxNorm[j] = gradOutput.data[baseIdx+j] * ln.gamma.data[j]
		}

		dVar := 0.0
		for j := 0; j < features; j++ {
			xMinusMean := ln.input.data[baseIdx+j] - mean
			dVar += dxNorm[j] * xMinusMean * (-0.5) * math.Pow(variance+ln.epsilon, -1.5)
		}

		dMean := 0.0
		for j := 0; j < features; j++ {
			dMean += dxNorm[j] * (-1.0 / std)
			dMean += dVar * (-2.0 * (ln.input.data[baseIdx+j] - mean) / N)
		}

		for j := 0; j < features; j++ {
			idx := baseIdx + j
			gradInput.data[idx] = dxNorm[j]/std + dVar*2*(ln.input.data[idx]-mean)/N + dMean/N
		}
	}

	return gradInput, nil
}

func (ln *LayerNormLayer) parameters() []*tensor {
	return []*tensor{ln.gamma, ln.beta}
}

func (ln *LayerNormLayer) gradients() []*tensor {
	return []*tensor{ln.gradGamma, ln.gradBeta}
}

func (ln *LayerNormLayer) outputShape() []int { return ln.inputShape }
func (ln *LayerNormLayer) name() string       { return "layer_norm" }

// RMSNormLayer - Root Mean Square Layer Normalization (used in LLaMA, Gemma)
// Simpler and faster than LayerNorm - no mean subtraction
type RMSNormLayer struct {
	epsilon   float64
	gamma     *tensor
	gradGamma *tensor
	input     *tensor
	rms       *tensor
	features  int
	built     bool
}

type RMSNormBuilder struct {
	layer *RMSNormLayer
}

func RMSNorm(epsilon float64) *RMSNormBuilder {
	return &RMSNormBuilder{
		layer: &RMSNormLayer{
			epsilon: epsilon,
		},
	}
}

func (b *RMSNormBuilder) Build() Layer {
	return b.layer
}

func (rn *RMSNormLayer) build(inputShape []int, rng *rand.Rand) error {
	if len(inputShape) == 0 {
		return errors.New("flow: RMSNorm requires non-empty input shape")
	}
	rn.features = inputShape[len(inputShape)-1]

	rn.gamma = newTensor(rn.features)
	rn.gamma.fill(1.0)
	rn.gradGamma = newTensor(rn.features)

	rn.built = true
	return nil
}

func (rn *RMSNormLayer) forward(input *tensor, training bool) (*tensor, error) {
	if !rn.built {
		return nil, errors.New("flow: RMSNorm not built")
	}

	rn.input = input
	batchSize := input.shape[0]
	features := rn.features

	rn.rms = newTensor(batchSize)
	output := newTensor(input.shape...)

	for i := 0; i < batchSize; i++ {
		// Compute RMS for this sample
		sumSq := 0.0
		for j := 0; j < features; j++ {
			sumSq += input.data[i*features+j] * input.data[i*features+j]
		}
		rn.rms.data[i] = math.Sqrt(sumSq/float64(features) + rn.epsilon)

		// Normalize and scale
		for j := 0; j < features; j++ {
			idx := i*features + j
			output.data[idx] = rn.gamma.data[j] * input.data[idx] / rn.rms.data[i]
		}
	}

	return output, nil
}

func (rn *RMSNormLayer) backward(gradOutput *tensor) (*tensor, error) {
	batchSize := rn.input.shape[0]
	features := rn.features
	N := float64(features)

	rn.gradGamma.zeroGrad()

	// Gradient w.r.t. gamma
	for i := 0; i < batchSize; i++ {
		for j := 0; j < features; j++ {
			idx := i*features + j
			rn.gradGamma.data[j] += gradOutput.data[idx] * rn.input.data[idx] / rn.rms.data[i]
		}
	}

	gradInput := newTensor(rn.input.shape...)

	for i := 0; i < batchSize; i++ {
		rms := rn.rms.data[i]
		rms3 := rms * rms * rms

		// Sum for gradient computation
		sumGradX := 0.0
		for j := 0; j < features; j++ {
			idx := i*features + j
			sumGradX += gradOutput.data[idx] * rn.gamma.data[j] * rn.input.data[idx]
		}

		for j := 0; j < features; j++ {
			idx := i*features + j
			gradInput.data[idx] = rn.gamma.data[j]*gradOutput.data[idx]/rms -
				rn.gamma.data[j]*rn.input.data[idx]*sumGradX/(N*rms3)
		}
	}

	return gradInput, nil
}

func (rn *RMSNormLayer) parameters() []*tensor {
	return []*tensor{rn.gamma}
}

func (rn *RMSNormLayer) gradients() []*tensor {
	return []*tensor{rn.gradGamma}
}

func (rn *RMSNormLayer) outputShape() []int { return []int{rn.features} }
func (rn *RMSNormLayer) name() string       { return "rms_norm" }

// GroupNormLayer - Group Normalization
// Divides channels into groups, normalizes each group
type GroupNormLayer struct {
	numGroups int
	epsilon   float64
	gamma     *tensor
	beta      *tensor
	gradGamma *tensor
	gradBeta  *tensor
	input     *tensor
	features  int
	built     bool
}

type GroupNormBuilder struct {
	layer *GroupNormLayer
}

func GroupNorm(numGroups int, epsilon float64) *GroupNormBuilder {
	return &GroupNormBuilder{
		layer: &GroupNormLayer{
			numGroups: numGroups,
			epsilon:   epsilon,
		},
	}
}

func (b *GroupNormBuilder) Build() Layer {
	return b.layer
}

func (gn *GroupNormLayer) build(inputShape []int, rng *rand.Rand) error {
	if len(inputShape) == 0 {
		return errors.New("flow: GroupNorm requires non-empty input shape")
	}
	gn.features = inputShape[len(inputShape)-1]
	if gn.features%gn.numGroups != 0 {
		return errors.New("flow: features must be divisible by numGroups")
	}

	gn.gamma = newTensor(gn.features)
	gn.gamma.fill(1.0)
	gn.beta = newTensor(gn.features)
	gn.beta.fill(0.0)
	gn.gradGamma = newTensor(gn.features)
	gn.gradBeta = newTensor(gn.features)

	gn.built = true
	return nil
}

func (gn *GroupNormLayer) forward(input *tensor, training bool) (*tensor, error) {
	if !gn.built {
		return nil, errors.New("flow: GroupNorm not built")
	}

	gn.input = input
	batchSize := input.shape[0]
	features := gn.features
	groupSize := features / gn.numGroups

	output := newTensor(input.shape...)

	for i := 0; i < batchSize; i++ {
		for g := 0; g < gn.numGroups; g++ {
			startIdx := g * groupSize
			endIdx := startIdx + groupSize

			// Compute group mean
			mean := 0.0
			for j := startIdx; j < endIdx; j++ {
				mean += input.data[i*features+j]
			}
			mean /= float64(groupSize)

			// Compute group variance
			variance := 0.0
			for j := startIdx; j < endIdx; j++ {
				diff := input.data[i*features+j] - mean
				variance += diff * diff
			}
			variance /= float64(groupSize)

			// Normalize and scale
			std := math.Sqrt(variance + gn.epsilon)
			for j := startIdx; j < endIdx; j++ {
				idx := i*features + j
				xNorm := (input.data[idx] - mean) / std
				output.data[idx] = gn.gamma.data[j]*xNorm + gn.beta.data[j]
			}
		}
	}

	return output, nil
}

func (gn *GroupNormLayer) backward(gradOutput *tensor) (*tensor, error) {
	// Simplified backward pass
	batchSize := gn.input.shape[0]
	features := gn.features

	gn.gradGamma.zeroGrad()
	gn.gradBeta.zeroGrad()
	gradInput := newTensor(gn.input.shape...)

	groupSize := features / gn.numGroups

	for i := 0; i < batchSize; i++ {
		for g := 0; g < gn.numGroups; g++ {
			startIdx := g * groupSize
			endIdx := startIdx + groupSize

			// Recompute mean/variance
			mean := 0.0
			for j := startIdx; j < endIdx; j++ {
				mean += gn.input.data[i*features+j]
			}
			mean /= float64(groupSize)

			variance := 0.0
			for j := startIdx; j < endIdx; j++ {
				diff := gn.input.data[i*features+j] - mean
				variance += diff * diff
			}
			variance /= float64(groupSize)
			std := math.Sqrt(variance + gn.epsilon)

			for j := startIdx; j < endIdx; j++ {
				idx := i*features + j
				xNorm := (gn.input.data[idx] - mean) / std
				gn.gradGamma.data[j] += gradOutput.data[idx] * xNorm
				gn.gradBeta.data[j] += gradOutput.data[idx]
				gradInput.data[idx] = gradOutput.data[idx] * gn.gamma.data[j] / std
			}
		}
	}

	return gradInput, nil
}

func (gn *GroupNormLayer) parameters() []*tensor {
	return []*tensor{gn.gamma, gn.beta}
}

func (gn *GroupNormLayer) gradients() []*tensor {
	return []*tensor{gn.gradGamma, gn.gradBeta}
}

func (gn *GroupNormLayer) outputShape() []int { return []int{gn.features} }
func (gn *GroupNormLayer) name() string       { return "group_norm" }
