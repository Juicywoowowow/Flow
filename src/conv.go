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

// =============================================================================
// DepthwiseConv2D - Depthwise Separable Convolution (Spatial Part)
// Applies a single filter per input channel (vs. standard conv which applies
// each filter to ALL input channels). Used in MobileNet, EfficientNet, etc.
// Computational cost: O(K²·C·H·W) vs O(K²·C_in·C_out·H·W) for standard conv
// =============================================================================

// DepthwiseConv2DLayer applies separate convolutions to each input channel
type DepthwiseConv2DLayer struct {
	kernelSize      [2]int
	stride          [2]int
	padding         string // "valid" or "same"
	depthMultiplier int    // Number of output channels per input channel
	activation      Activation
	initializer     Initializer
	biasInit        Initializer
	useBias         bool
	weights         *tensor // [kernelH, kernelW, inChannels, depthMultiplier]
	bias            *tensor // [inChannels * depthMultiplier]
	input           *tensor
	preAct          *tensor
	gradW           *tensor
	gradB           *tensor
	inputShape      []int // [H, W, C]
	built           bool
}

type DepthwiseConv2DBuilder struct {
	layer *DepthwiseConv2DLayer
}

// DepthwiseConv2D creates a depthwise convolution layer
// Each input channel is convolved with its own set of filters
// Output channels = input channels * depthMultiplier
func DepthwiseConv2D(kernelSize [2]int) *DepthwiseConv2DBuilder {
	return &DepthwiseConv2DBuilder{
		layer: &DepthwiseConv2DLayer{
			kernelSize:      kernelSize,
			stride:          [2]int{1, 1},
			padding:         "valid",
			depthMultiplier: 1,
		},
	}
}

func (b *DepthwiseConv2DBuilder) WithStride(strideH, strideW int) *DepthwiseConv2DBuilder {
	b.layer.stride = [2]int{strideH, strideW}
	return b
}

func (b *DepthwiseConv2DBuilder) WithPadding(padding string) *DepthwiseConv2DBuilder {
	b.layer.padding = padding
	return b
}

// WithDepthMultiplier sets the number of output channels per input channel
// Default is 1. If set to 2, each input channel produces 2 output channels.
func (b *DepthwiseConv2DBuilder) WithDepthMultiplier(mult int) *DepthwiseConv2DBuilder {
	b.layer.depthMultiplier = mult
	return b
}

func (b *DepthwiseConv2DBuilder) WithActivation(act Activation) *DepthwiseConv2DBuilder {
	b.layer.activation = act
	return b
}

func (b *DepthwiseConv2DBuilder) WithInitializer(init Initializer) *DepthwiseConv2DBuilder {
	b.layer.initializer = init
	return b
}

func (b *DepthwiseConv2DBuilder) WithBiasInitializer(init Initializer) *DepthwiseConv2DBuilder {
	b.layer.biasInit = init
	return b
}

func (b *DepthwiseConv2DBuilder) WithBias(useBias bool) *DepthwiseConv2DBuilder {
	b.layer.useBias = useBias
	return b
}

func (b *DepthwiseConv2DBuilder) Build() Layer {
	return b.layer
}

func (d *DepthwiseConv2DLayer) build(inputShape []int, rng *rand.Rand) error {
	if len(inputShape) != 3 {
		return errors.New("flow: DepthwiseConv2D requires input shape [H, W, C]")
	}
	if d.initializer == nil {
		return errors.New("flow: DepthwiseConv2D requires initializer")
	}
	if d.activation == nil {
		return errors.New("flow: DepthwiseConv2D requires activation")
	}
	if d.useBias && d.biasInit == nil {
		return errors.New("flow: DepthwiseConv2D with bias requires bias initializer")
	}
	if d.depthMultiplier < 1 {
		return errors.New("flow: DepthwiseConv2D depthMultiplier must be >= 1")
	}

	d.inputShape = inputShape
	inChannels := inputShape[2]
	outChannels := inChannels * d.depthMultiplier

	// Weights shape: [kernelH, kernelW, inChannels, depthMultiplier]
	// Each input channel has depthMultiplier separate K×K filters
	d.weights = newTensor(d.kernelSize[0], d.kernelSize[1], inChannels, d.depthMultiplier)
	fanIn := d.kernelSize[0] * d.kernelSize[1]
	fanOut := d.kernelSize[0] * d.kernelSize[1] * d.depthMultiplier
	d.initializer.initialize(d.weights, fanIn, fanOut, rng)

	d.gradW = newTensor(d.kernelSize[0], d.kernelSize[1], inChannels, d.depthMultiplier)

	if d.useBias {
		d.bias = newTensor(outChannels)
		d.biasInit.initialize(d.bias, fanIn, fanOut, rng)
		d.gradB = newTensor(outChannels)
	}

	d.built = true
	return nil
}

func (d *DepthwiseConv2DLayer) computeOutputSize(inputH, inputW int) (int, int) {
	var outH, outW int
	if d.padding == "same" {
		outH = (inputH + d.stride[0] - 1) / d.stride[0]
		outW = (inputW + d.stride[1] - 1) / d.stride[1]
	} else { // valid
		outH = (inputH-d.kernelSize[0])/d.stride[0] + 1
		outW = (inputW-d.kernelSize[1])/d.stride[1] + 1
	}
	return outH, outW
}

func (d *DepthwiseConv2DLayer) forward(input *tensor, training bool) (*tensor, error) {
	if !d.built {
		return nil, errors.New("flow: DepthwiseConv2D not built")
	}

	batchSize := input.shape[0]
	inputH := input.shape[1]
	inputW := input.shape[2]
	inChannels := input.shape[3]

	outH, outW := d.computeOutputSize(inputH, inputW)
	outChannels := inChannels * d.depthMultiplier

	d.input = input
	d.preAct = newTensor(batchSize, outH, outW, outChannels)

	// Compute padding
	var padTop, padLeft int
	if d.padding == "same" {
		padH := maxInt((outH-1)*d.stride[0]+d.kernelSize[0]-inputH, 0)
		padW := maxInt((outW-1)*d.stride[1]+d.kernelSize[1]-inputW, 0)
		padTop = padH / 2
		padLeft = padW / 2
	}

	// Depthwise convolution: each input channel convolves with its own filter(s)
	for b := 0; b < batchSize; b++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				for ic := 0; ic < inChannels; ic++ {
					for dm := 0; dm < d.depthMultiplier; dm++ {
						sum := 0.0

						for kh := 0; kh < d.kernelSize[0]; kh++ {
							for kw := 0; kw < d.kernelSize[1]; kw++ {
								ih := oh*d.stride[0] + kh - padTop
								iw := ow*d.stride[1] + kw - padLeft

								if ih >= 0 && ih < inputH && iw >= 0 && iw < inputW {
									inputIdx := b*inputH*inputW*inChannels + ih*inputW*inChannels + iw*inChannels + ic
									// Weight index: [kh, kw, ic, dm]
									weightIdx := kh*d.kernelSize[1]*inChannels*d.depthMultiplier +
										kw*inChannels*d.depthMultiplier +
										ic*d.depthMultiplier + dm
									sum += input.data[inputIdx] * d.weights.data[weightIdx]
								}
							}
						}

						// Output channel = ic * depthMultiplier + dm
						oc := ic*d.depthMultiplier + dm
						outIdx := b*outH*outW*outChannels + oh*outW*outChannels + ow*outChannels + oc
						d.preAct.data[outIdx] = sum

						if d.useBias {
							d.preAct.data[outIdx] += d.bias.data[oc]
						}
					}
				}
			}
		}
	}

	output := newTensor(d.preAct.shape...)
	d.activation.forward(d.preAct, output)
	return output, nil
}

func (d *DepthwiseConv2DLayer) backward(gradOutput *tensor) (*tensor, error) {
	batchSize := d.input.shape[0]
	inputH := d.input.shape[1]
	inputW := d.input.shape[2]
	inChannels := d.input.shape[3]
	outH := gradOutput.shape[1]
	outW := gradOutput.shape[2]
	outChannels := inChannels * d.depthMultiplier

	// Gradient through activation
	gradPreAct := newTensor(gradOutput.shape...)
	d.activation.backward(d.preAct, gradOutput, gradPreAct)

	// Compute padding
	var padTop, padLeft int
	if d.padding == "same" {
		padH := maxInt((outH-1)*d.stride[0]+d.kernelSize[0]-inputH, 0)
		padW := maxInt((outW-1)*d.stride[1]+d.kernelSize[1]-inputW, 0)
		padTop = padH / 2
		padLeft = padW / 2
	}

	d.gradW.zeroGrad()
	if d.useBias {
		d.gradB.zeroGrad()
	}
	gradInput := newTensor(d.input.shape...)

	for b := 0; b < batchSize; b++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				for ic := 0; ic < inChannels; ic++ {
					for dm := 0; dm < d.depthMultiplier; dm++ {
						oc := ic*d.depthMultiplier + dm
						outIdx := b*outH*outW*outChannels + oh*outW*outChannels + ow*outChannels + oc
						dout := gradPreAct.data[outIdx]

						if d.useBias {
							d.gradB.data[oc] += dout
						}

						for kh := 0; kh < d.kernelSize[0]; kh++ {
							for kw := 0; kw < d.kernelSize[1]; kw++ {
								ih := oh*d.stride[0] + kh - padTop
								iw := ow*d.stride[1] + kw - padLeft

								if ih >= 0 && ih < inputH && iw >= 0 && iw < inputW {
									inputIdx := b*inputH*inputW*inChannels + ih*inputW*inChannels + iw*inChannels + ic
									weightIdx := kh*d.kernelSize[1]*inChannels*d.depthMultiplier +
										kw*inChannels*d.depthMultiplier +
										ic*d.depthMultiplier + dm

									d.gradW.data[weightIdx] += d.input.data[inputIdx] * dout
									gradInput.data[inputIdx] += d.weights.data[weightIdx] * dout
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
	mulScalar(d.gradW, scale)
	if d.useBias {
		mulScalar(d.gradB, scale)
	}

	return gradInput, nil
}

func (d *DepthwiseConv2DLayer) parameters() []*tensor {
	if d.useBias {
		return []*tensor{d.weights, d.bias}
	}
	return []*tensor{d.weights}
}

func (d *DepthwiseConv2DLayer) gradients() []*tensor {
	if d.useBias {
		return []*tensor{d.gradW, d.gradB}
	}
	return []*tensor{d.gradW}
}

func (d *DepthwiseConv2DLayer) outputShape() []int {
	outH, outW := d.computeOutputSize(d.inputShape[0], d.inputShape[1])
	outChannels := d.inputShape[2] * d.depthMultiplier
	return []int{outH, outW, outChannels}
}

func (d *DepthwiseConv2DLayer) name() string { return "depthwise_conv2d" }

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
