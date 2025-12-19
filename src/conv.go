package flow

import (
	"errors"
	"math"
	"math/rand"
)

// Conv2DLayer - 2D Convolution layer
type Conv2DLayer struct {
	filters     int
	kernelSize  [2]int
	stride      [2]int
	padding     string // "valid" or "same"
	activation  Activation
	initializer Initializer
	biasInit    Initializer
	useBias     bool
	weights     *tensor // [kernelH, kernelW, inChannels, outChannels]
	bias        *tensor
	input       *tensor
	preAct      *tensor
	gradW       *tensor
	gradB       *tensor
	inputShape  []int // [H, W, C]
	built       bool
}

type Conv2DBuilder struct {
	layer *Conv2DLayer
}

func Conv2D(filters int, kernelSize [2]int) *Conv2DBuilder {
	return &Conv2DBuilder{
		layer: &Conv2DLayer{
			filters:    filters,
			kernelSize: kernelSize,
			stride:     [2]int{1, 1},
			padding:    "valid",
		},
	}
}

func (b *Conv2DBuilder) WithStride(strideH, strideW int) *Conv2DBuilder {
	b.layer.stride = [2]int{strideH, strideW}
	return b
}

func (b *Conv2DBuilder) WithPadding(padding string) *Conv2DBuilder {
	b.layer.padding = padding
	return b
}

func (b *Conv2DBuilder) WithActivation(act Activation) *Conv2DBuilder {
	b.layer.activation = act
	return b
}

func (b *Conv2DBuilder) WithInitializer(init Initializer) *Conv2DBuilder {
	b.layer.initializer = init
	return b
}

func (b *Conv2DBuilder) WithBiasInitializer(init Initializer) *Conv2DBuilder {
	b.layer.biasInit = init
	return b
}

func (b *Conv2DBuilder) WithBias(useBias bool) *Conv2DBuilder {
	b.layer.useBias = useBias
	return b
}

func (b *Conv2DBuilder) Build() Layer {
	return b.layer
}

func (c *Conv2DLayer) build(inputShape []int, rng *rand.Rand) error {
	if len(inputShape) != 3 {
		return errors.New("flow: Conv2D requires input shape [H, W, C]")
	}
	if c.initializer == nil {
		return errors.New("flow: Conv2D requires initializer")
	}
	if c.activation == nil {
		return errors.New("flow: Conv2D requires activation")
	}
	if c.useBias && c.biasInit == nil {
		return errors.New("flow: Conv2D with bias requires bias initializer")
	}

	c.inputShape = inputShape
	inChannels := inputShape[2]

	// Weights shape: [kernelH, kernelW, inChannels, outChannels]
	c.weights = newTensor(c.kernelSize[0], c.kernelSize[1], inChannels, c.filters)
	fanIn := c.kernelSize[0] * c.kernelSize[1] * inChannels
	fanOut := c.kernelSize[0] * c.kernelSize[1] * c.filters
	c.initializer.initialize(c.weights, fanIn, fanOut, rng)

	c.gradW = newTensor(c.kernelSize[0], c.kernelSize[1], inChannels, c.filters)

	if c.useBias {
		c.bias = newTensor(c.filters)
		c.biasInit.initialize(c.bias, fanIn, fanOut, rng)
		c.gradB = newTensor(c.filters)
	}

	c.built = true
	return nil
}

func (c *Conv2DLayer) computeOutputSize(inputH, inputW int) (int, int) {
	var outH, outW int
	if c.padding == "same" {
		outH = (inputH + c.stride[0] - 1) / c.stride[0]
		outW = (inputW + c.stride[1] - 1) / c.stride[1]
	} else { // valid
		outH = (inputH-c.kernelSize[0])/c.stride[0] + 1
		outW = (inputW-c.kernelSize[1])/c.stride[1] + 1
	}
	return outH, outW
}

func (c *Conv2DLayer) forward(input *tensor, training bool) (*tensor, error) {
	if !c.built {
		return nil, errors.New("flow: Conv2D not built")
	}

	batchSize := input.shape[0]
	inputH := input.shape[1]
	inputW := input.shape[2]
	inChannels := input.shape[3]

	outH, outW := c.computeOutputSize(inputH, inputW)

	c.input = input
	c.preAct = newTensor(batchSize, outH, outW, c.filters)

	// Compute padding
	var padTop, padLeft int
	if c.padding == "same" {
		padH := maxInt((outH-1)*c.stride[0]+c.kernelSize[0]-inputH, 0)
		padW := maxInt((outW-1)*c.stride[1]+c.kernelSize[1]-inputW, 0)
		padTop = padH / 2
		padLeft = padW / 2
	}

	// Convolution
	for b := 0; b < batchSize; b++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				for f := 0; f < c.filters; f++ {
					sum := 0.0
					for kh := 0; kh < c.kernelSize[0]; kh++ {
						for kw := 0; kw < c.kernelSize[1]; kw++ {
							ih := oh*c.stride[0] + kh - padTop
							iw := ow*c.stride[1] + kw - padLeft
							if ih >= 0 && ih < inputH && iw >= 0 && iw < inputW {
								for ic := 0; ic < inChannels; ic++ {
									inputIdx := b*inputH*inputW*inChannels + ih*inputW*inChannels + iw*inChannels + ic
									weightIdx := kh*c.kernelSize[1]*inChannels*c.filters + kw*inChannels*c.filters + ic*c.filters + f
									sum += input.data[inputIdx] * c.weights.data[weightIdx]
								}
							}
						}
					}
					outIdx := b*outH*outW*c.filters + oh*outW*c.filters + ow*c.filters + f
					c.preAct.data[outIdx] = sum
					if c.useBias {
						c.preAct.data[outIdx] += c.bias.data[f]
					}
				}
			}
		}
	}

	output := newTensor(c.preAct.shape...)
	c.activation.forward(c.preAct, output)
	return output, nil
}

func (c *Conv2DLayer) backward(gradOutput *tensor) (*tensor, error) {
	batchSize := c.input.shape[0]
	inputH := c.input.shape[1]
	inputW := c.input.shape[2]
	inChannels := c.input.shape[3]
	outH := gradOutput.shape[1]
	outW := gradOutput.shape[2]

	// Gradient through activation
	gradPreAct := newTensor(gradOutput.shape...)
	c.activation.backward(c.preAct, gradOutput, gradPreAct)

	// Compute padding
	var padTop, padLeft int
	if c.padding == "same" {
		padH := maxInt((outH-1)*c.stride[0]+c.kernelSize[0]-inputH, 0)
		padW := maxInt((outW-1)*c.stride[1]+c.kernelSize[1]-inputW, 0)
		padTop = padH / 2
		padLeft = padW / 2
	}

	c.gradW.zeroGrad()
	if c.useBias {
		c.gradB.zeroGrad()
	}
	gradInput := newTensor(c.input.shape...)

	for b := 0; b < batchSize; b++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				for f := 0; f < c.filters; f++ {
					outIdx := b*outH*outW*c.filters + oh*outW*c.filters + ow*c.filters + f
					dout := gradPreAct.data[outIdx]

					if c.useBias {
						c.gradB.data[f] += dout
					}

					for kh := 0; kh < c.kernelSize[0]; kh++ {
						for kw := 0; kw < c.kernelSize[1]; kw++ {
							ih := oh*c.stride[0] + kh - padTop
							iw := ow*c.stride[1] + kw - padLeft
							if ih >= 0 && ih < inputH && iw >= 0 && iw < inputW {
								for ic := 0; ic < inChannels; ic++ {
									inputIdx := b*inputH*inputW*inChannels + ih*inputW*inChannels + iw*inChannels + ic
									weightIdx := kh*c.kernelSize[1]*inChannels*c.filters + kw*inChannels*c.filters + ic*c.filters + f

									c.gradW.data[weightIdx] += c.input.data[inputIdx] * dout
									gradInput.data[inputIdx] += c.weights.data[weightIdx] * dout
								}
							}
						}
					}
				}
			}
		}
	}

	// Average gradients
	scale := 1.0 / float64(batchSize)
	mulScalar(c.gradW, scale)
	if c.useBias {
		mulScalar(c.gradB, scale)
	}

	return gradInput, nil
}

func (c *Conv2DLayer) parameters() []*tensor {
	if c.useBias {
		return []*tensor{c.weights, c.bias}
	}
	return []*tensor{c.weights}
}

func (c *Conv2DLayer) gradients() []*tensor {
	if c.useBias {
		return []*tensor{c.gradW, c.gradB}
	}
	return []*tensor{c.gradW}
}

func (c *Conv2DLayer) outputShape() []int {
	outH, outW := c.computeOutputSize(c.inputShape[0], c.inputShape[1])
	return []int{outH, outW, c.filters}
}

func (c *Conv2DLayer) name() string { return "conv2d" }

// MaxPool2DLayer - Max pooling layer
type MaxPool2DLayer struct {
	poolSize   [2]int
	stride     [2]int
	padding    string
	inputShape []int
	maxIndices *tensor // Store indices for backward pass
	built      bool
}

type MaxPool2DBuilder struct {
	layer *MaxPool2DLayer
}

func MaxPool2D(poolSize [2]int) *MaxPool2DBuilder {
	return &MaxPool2DBuilder{
		layer: &MaxPool2DLayer{
			poolSize: poolSize,
			stride:   poolSize, // Default stride = pool size
			padding:  "valid",
		},
	}
}

func (b *MaxPool2DBuilder) WithStride(strideH, strideW int) *MaxPool2DBuilder {
	b.layer.stride = [2]int{strideH, strideW}
	return b
}

func (b *MaxPool2DBuilder) WithPadding(padding string) *MaxPool2DBuilder {
	b.layer.padding = padding
	return b
}

func (b *MaxPool2DBuilder) Build() Layer {
	return b.layer
}

func (m *MaxPool2DLayer) build(inputShape []int, rng *rand.Rand) error {
	if len(inputShape) != 3 {
		return errors.New("flow: MaxPool2D requires input shape [H, W, C]")
	}
	m.inputShape = inputShape
	m.built = true
	return nil
}

func (m *MaxPool2DLayer) computeOutputSize(inputH, inputW int) (int, int) {
	var outH, outW int
	if m.padding == "same" {
		outH = (inputH + m.stride[0] - 1) / m.stride[0]
		outW = (inputW + m.stride[1] - 1) / m.stride[1]
	} else {
		outH = (inputH-m.poolSize[0])/m.stride[0] + 1
		outW = (inputW-m.poolSize[1])/m.stride[1] + 1
	}
	return outH, outW
}

func (m *MaxPool2DLayer) forward(input *tensor, training bool) (*tensor, error) {
	batchSize := input.shape[0]
	inputH := input.shape[1]
	inputW := input.shape[2]
	channels := input.shape[3]

	outH, outW := m.computeOutputSize(inputH, inputW)
	output := newTensor(batchSize, outH, outW, channels)
	m.maxIndices = newTensor(batchSize, outH, outW, channels)

	for b := 0; b < batchSize; b++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				for c := 0; c < channels; c++ {
					maxVal := math.Inf(-1)
					maxIdx := 0

					for ph := 0; ph < m.poolSize[0]; ph++ {
						for pw := 0; pw < m.poolSize[1]; pw++ {
							ih := oh*m.stride[0] + ph
							iw := ow*m.stride[1] + pw
							if ih < inputH && iw < inputW {
								idx := b*inputH*inputW*channels + ih*inputW*channels + iw*channels + c
								if input.data[idx] > maxVal {
									maxVal = input.data[idx]
									maxIdx = idx
								}
							}
						}
					}

					outIdx := b*outH*outW*channels + oh*outW*channels + ow*channels + c
					output.data[outIdx] = maxVal
					m.maxIndices.data[outIdx] = float64(maxIdx)
				}
			}
		}
	}

	return output, nil
}

func (m *MaxPool2DLayer) backward(gradOutput *tensor) (*tensor, error) {
	batchSize := gradOutput.shape[0]
	outH := gradOutput.shape[1]
	outW := gradOutput.shape[2]
	channels := gradOutput.shape[3]

	gradInput := newTensor(batchSize, m.inputShape[0], m.inputShape[1], m.inputShape[2])

	for b := 0; b < batchSize; b++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				for c := 0; c < channels; c++ {
					outIdx := b*outH*outW*channels + oh*outW*channels + ow*channels + c
					maxIdx := int(m.maxIndices.data[outIdx])
					gradInput.data[maxIdx] += gradOutput.data[outIdx]
				}
			}
		}
	}

	return gradInput, nil
}

func (m *MaxPool2DLayer) parameters() []*tensor { return nil }
func (m *MaxPool2DLayer) gradients() []*tensor  { return nil }

func (m *MaxPool2DLayer) outputShape() []int {
	outH, outW := m.computeOutputSize(m.inputShape[0], m.inputShape[1])
	return []int{outH, outW, m.inputShape[2]}
}

func (m *MaxPool2DLayer) name() string { return "max_pool2d" }

// AvgPool2DLayer - Average pooling layer
type AvgPool2DLayer struct {
	poolSize   [2]int
	stride     [2]int
	padding    string
	inputShape []int
	built      bool
}

type AvgPool2DBuilder struct {
	layer *AvgPool2DLayer
}

func AvgPool2D(poolSize [2]int) *AvgPool2DBuilder {
	return &AvgPool2DBuilder{
		layer: &AvgPool2DLayer{
			poolSize: poolSize,
			stride:   poolSize,
			padding:  "valid",
		},
	}
}

func (b *AvgPool2DBuilder) WithStride(strideH, strideW int) *AvgPool2DBuilder {
	b.layer.stride = [2]int{strideH, strideW}
	return b
}

func (b *AvgPool2DBuilder) WithPadding(padding string) *AvgPool2DBuilder {
	b.layer.padding = padding
	return b
}

func (b *AvgPool2DBuilder) Build() Layer {
	return b.layer
}

func (a *AvgPool2DLayer) build(inputShape []int, rng *rand.Rand) error {
	if len(inputShape) != 3 {
		return errors.New("flow: AvgPool2D requires input shape [H, W, C]")
	}
	a.inputShape = inputShape
	a.built = true
	return nil
}

func (a *AvgPool2DLayer) computeOutputSize(inputH, inputW int) (int, int) {
	var outH, outW int
	if a.padding == "same" {
		outH = (inputH + a.stride[0] - 1) / a.stride[0]
		outW = (inputW + a.stride[1] - 1) / a.stride[1]
	} else {
		outH = (inputH-a.poolSize[0])/a.stride[0] + 1
		outW = (inputW-a.poolSize[1])/a.stride[1] + 1
	}
	return outH, outW
}

func (a *AvgPool2DLayer) forward(input *tensor, training bool) (*tensor, error) {
	batchSize := input.shape[0]
	inputH := input.shape[1]
	inputW := input.shape[2]
	channels := input.shape[3]

	outH, outW := a.computeOutputSize(inputH, inputW)
	output := newTensor(batchSize, outH, outW, channels)

	poolArea := float64(a.poolSize[0] * a.poolSize[1])

	for b := 0; b < batchSize; b++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				for c := 0; c < channels; c++ {
					sum := 0.0
					count := 0

					for ph := 0; ph < a.poolSize[0]; ph++ {
						for pw := 0; pw < a.poolSize[1]; pw++ {
							ih := oh*a.stride[0] + ph
							iw := ow*a.stride[1] + pw
							if ih < inputH && iw < inputW {
								idx := b*inputH*inputW*channels + ih*inputW*channels + iw*channels + c
								sum += input.data[idx]
								count++
							}
						}
					}

					outIdx := b*outH*outW*channels + oh*outW*channels + ow*channels + c
					if count > 0 {
						output.data[outIdx] = sum / poolArea
					}
				}
			}
		}
	}

	return output, nil
}

func (a *AvgPool2DLayer) backward(gradOutput *tensor) (*tensor, error) {
	batchSize := gradOutput.shape[0]
	inputH := a.inputShape[0]
	inputW := a.inputShape[1]
	channels := a.inputShape[2]
	outH := gradOutput.shape[1]
	outW := gradOutput.shape[2]

	gradInput := newTensor(batchSize, inputH, inputW, channels)
	poolArea := float64(a.poolSize[0] * a.poolSize[1])

	for b := 0; b < batchSize; b++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				for c := 0; c < channels; c++ {
					outIdx := b*outH*outW*channels + oh*outW*channels + ow*channels + c
					grad := gradOutput.data[outIdx] / poolArea

					for ph := 0; ph < a.poolSize[0]; ph++ {
						for pw := 0; pw < a.poolSize[1]; pw++ {
							ih := oh*a.stride[0] + ph
							iw := ow*a.stride[1] + pw
							if ih < inputH && iw < inputW {
								idx := b*inputH*inputW*channels + ih*inputW*channels + iw*channels + c
								gradInput.data[idx] += grad
							}
						}
					}
				}
			}
		}
	}

	return gradInput, nil
}

func (a *AvgPool2DLayer) parameters() []*tensor { return nil }
func (a *AvgPool2DLayer) gradients() []*tensor  { return nil }

func (a *AvgPool2DLayer) outputShape() []int {
	outH, outW := a.computeOutputSize(a.inputShape[0], a.inputShape[1])
	return []int{outH, outW, a.inputShape[2]}
}

func (a *AvgPool2DLayer) name() string { return "avg_pool2d" }
