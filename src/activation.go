package flow

import "math"

// Activation represents an activation function
type Activation interface {
	forward(x *tensor, out *tensor)
	backward(x *tensor, gradOut *tensor, gradIn *tensor)
	name() string
}

// ReLUActivation - Rectified Linear Unit
type ReLUActivation struct{}

func ReLU() Activation { return &ReLUActivation{} }

func (r *ReLUActivation) forward(x *tensor, out *tensor) {
	for i, v := range x.data {
		if v > 0 {
			out.data[i] = v
		} else {
			out.data[i] = 0
		}
	}
}

func (r *ReLUActivation) backward(x *tensor, gradOut *tensor, gradIn *tensor) {
	for i, v := range x.data {
		if v > 0 {
			gradIn.data[i] = gradOut.data[i]
		} else {
			gradIn.data[i] = 0
		}
	}
}

func (r *ReLUActivation) name() string { return "relu" }

// LeakyReLUActivation - Leaky ReLU with configurable negative slope
type LeakyReLUActivation struct {
	NegativeSlope float64
}

func LeakyReLU(negativeSlope float64) Activation {
	return &LeakyReLUActivation{NegativeSlope: negativeSlope}
}

func (l *LeakyReLUActivation) forward(x *tensor, out *tensor) {
	for i, v := range x.data {
		if v > 0 {
			out.data[i] = v
		} else {
			out.data[i] = v * l.NegativeSlope
		}
	}
}

func (l *LeakyReLUActivation) backward(x *tensor, gradOut *tensor, gradIn *tensor) {
	for i, v := range x.data {
		if v > 0 {
			gradIn.data[i] = gradOut.data[i]
		} else {
			gradIn.data[i] = gradOut.data[i] * l.NegativeSlope
		}
	}
}

func (l *LeakyReLUActivation) name() string { return "leaky_relu" }

// ELUActivation - Exponential Linear Unit
type ELUActivation struct {
	Alpha float64
}

func ELU(alpha float64) Activation {
	return &ELUActivation{Alpha: alpha}
}

func (e *ELUActivation) forward(x *tensor, out *tensor) {
	for i, v := range x.data {
		if v > 0 {
			out.data[i] = v
		} else {
			// Clamp to prevent underflow (exp of very negative = 0)
			clampedV := math.Max(v, -700) // exp(-700) is very small but safe
			out.data[i] = e.Alpha * (math.Exp(clampedV) - 1)
		}
	}
}

func (e *ELUActivation) backward(x *tensor, gradOut *tensor, gradIn *tensor) {
	for i, v := range x.data {
		if v > 0 {
			gradIn.data[i] = gradOut.data[i]
		} else {
			clampedV := math.Max(v, -700)
			gradIn.data[i] = gradOut.data[i] * e.Alpha * math.Exp(clampedV)
		}
	}
}

func (e *ELUActivation) name() string { return "elu" }

// SigmoidActivation
type SigmoidActivation struct{}

func Sigmoid() Activation { return &SigmoidActivation{} }

func (s *SigmoidActivation) forward(x *tensor, out *tensor) {
	for i, v := range x.data {
		// Clamp input to prevent overflow: exp(-v) overflows for v < -709
		if v >= 0 {
			out.data[i] = 1.0 / (1.0 + math.Exp(-v))
		} else {
			// Use numerically stable form for negative values
			expV := math.Exp(v)
			out.data[i] = expV / (1.0 + expV)
		}
	}
}

func (s *SigmoidActivation) backward(x *tensor, gradOut *tensor, gradIn *tensor) {
	for i, v := range x.data {
		sig := 1.0 / (1.0 + math.Exp(-v))
		gradIn.data[i] = gradOut.data[i] * sig * (1 - sig)
	}
}

func (s *SigmoidActivation) name() string { return "sigmoid" }

// TanhActivation
type TanhActivation struct{}

func Tanh() Activation { return &TanhActivation{} }

func (t *TanhActivation) forward(x *tensor, out *tensor) {
	for i, v := range x.data {
		out.data[i] = math.Tanh(v)
	}
}

func (t *TanhActivation) backward(x *tensor, gradOut *tensor, gradIn *tensor) {
	for i, v := range x.data {
		th := math.Tanh(v)
		gradIn.data[i] = gradOut.data[i] * (1 - th*th)
	}
}

func (t *TanhActivation) name() string { return "tanh" }

// SoftmaxActivation - operates on last dimension
type SoftmaxActivation struct{}

func Softmax() Activation { return &SoftmaxActivation{} }

func (s *SoftmaxActivation) forward(x *tensor, out *tensor) {
	if len(x.shape) == 1 {
		maxV := maxVal(x)
		sum := 0.0
		for i, v := range x.data {
			out.data[i] = math.Exp(v - maxV)
			sum += out.data[i]
		}
		for i := range out.data {
			out.data[i] /= sum
		}
	} else {
		rows := x.shape[0]
		cols := x.shape[1]
		for r := 0; r < rows; r++ {
			maxV := x.data[r*cols]
			for c := 1; c < cols; c++ {
				if x.data[r*cols+c] > maxV {
					maxV = x.data[r*cols+c]
				}
			}
			sum := 0.0
			for c := 0; c < cols; c++ {
				out.data[r*cols+c] = math.Exp(x.data[r*cols+c] - maxV)
				sum += out.data[r*cols+c]
			}
			for c := 0; c < cols; c++ {
				out.data[r*cols+c] /= sum
			}
		}
	}
}

func (s *SoftmaxActivation) backward(x *tensor, gradOut *tensor, gradIn *tensor) {
	// For cross-entropy loss, gradient is simplified to (softmax - target)
	// This is handled in the loss function, so we pass through here
	copy(gradIn.data, gradOut.data)
}

func (s *SoftmaxActivation) name() string { return "softmax" }

// SwishActivation - x * sigmoid(x)
type SwishActivation struct{}

func Swish() Activation { return &SwishActivation{} }

func (sw *SwishActivation) forward(x *tensor, out *tensor) {
	for i, v := range x.data {
		sig := 1.0 / (1.0 + math.Exp(-v))
		out.data[i] = v * sig
	}
}

func (sw *SwishActivation) backward(x *tensor, gradOut *tensor, gradIn *tensor) {
	for i, v := range x.data {
		sig := 1.0 / (1.0 + math.Exp(-v))
		swish := v * sig
		gradIn.data[i] = gradOut.data[i] * (swish + sig*(1-swish))
	}
}

func (sw *SwishActivation) name() string { return "swish" }

// GELUActivation - Gaussian Error Linear Unit
type GELUActivation struct{}

func GELU() Activation { return &GELUActivation{} }

func (g *GELUActivation) forward(x *tensor, out *tensor) {
	sqrt2 := math.Sqrt(2.0)
	for i, v := range x.data {
		out.data[i] = 0.5 * v * (1 + math.Erf(v/sqrt2))
	}
}

func (g *GELUActivation) backward(x *tensor, gradOut *tensor, gradIn *tensor) {
	sqrt2 := math.Sqrt(2.0)
	sqrt2pi := math.Sqrt(2.0 / math.Pi)
	for i, v := range x.data {
		cdf := 0.5 * (1 + math.Erf(v/sqrt2))
		pdf := sqrt2pi * math.Exp(-0.5*v*v)
		gradIn.data[i] = gradOut.data[i] * (cdf + v*pdf)
	}
}

func (g *GELUActivation) name() string { return "gelu" }

// LinearActivation - no-op, identity function
type LinearActivation struct{}

func Linear() Activation { return &LinearActivation{} }

func (l *LinearActivation) forward(x *tensor, out *tensor) {
	copy(out.data, x.data)
}

func (l *LinearActivation) backward(x *tensor, gradOut *tensor, gradIn *tensor) {
	copy(gradIn.data, gradOut.data)
}

func (l *LinearActivation) name() string { return "linear" }
