package flow

import "math"

// Loss computes loss and gradients
type Loss interface {
	compute(pred, target *tensor) float64
	gradient(pred, target *tensor, gradOut *tensor)
	name() string
}

// MSELoss - Mean Squared Error
type MSELoss struct {
	Reduction string // "mean" or "sum"
}

type MSEConfig struct {
	Reduction string
}

func MSE(config MSEConfig) Loss {
	return &MSELoss{Reduction: config.Reduction}
}

func (m *MSELoss) compute(pred, target *tensor) float64 {
	sum := 0.0
	for i := range pred.data {
		diff := pred.data[i] - target.data[i]
		sum += diff * diff
	}
	if m.Reduction == "mean" {
		return sum / float64(len(pred.data))
	}
	return sum
}

func (m *MSELoss) gradient(pred, target *tensor, gradOut *tensor) {
	scale := 2.0
	if m.Reduction == "mean" {
		scale = 2.0 / float64(len(pred.data))
	}
	for i := range pred.data {
		gradOut.data[i] = scale * (pred.data[i] - target.data[i])
	}
}

func (m *MSELoss) name() string { return "mse" }

// MAELoss - Mean Absolute Error
type MAELoss struct {
	Reduction string
}

type MAEConfig struct {
	Reduction string
}

func MAE(config MAEConfig) Loss {
	return &MAELoss{Reduction: config.Reduction}
}

func (m *MAELoss) compute(pred, target *tensor) float64 {
	sum := 0.0
	for i := range pred.data {
		sum += math.Abs(pred.data[i] - target.data[i])
	}
	if m.Reduction == "mean" {
		return sum / float64(len(pred.data))
	}
	return sum
}

func (m *MAELoss) gradient(pred, target *tensor, gradOut *tensor) {
	scale := 1.0
	if m.Reduction == "mean" {
		scale = 1.0 / float64(len(pred.data))
	}
	for i := range pred.data {
		if pred.data[i] > target.data[i] {
			gradOut.data[i] = scale
		} else if pred.data[i] < target.data[i] {
			gradOut.data[i] = -scale
		} else {
			gradOut.data[i] = 0
		}
	}
}

func (m *MAELoss) name() string { return "mae" }

// HuberLoss - Smooth L1 Loss
type HuberLoss struct {
	Delta     float64
	Reduction string
}

type HuberConfig struct {
	Delta     float64
	Reduction string
}

func Huber(config HuberConfig) Loss {
	return &HuberLoss{Delta: config.Delta, Reduction: config.Reduction}
}

func (h *HuberLoss) compute(pred, target *tensor) float64 {
	sum := 0.0
	for i := range pred.data {
		diff := math.Abs(pred.data[i] - target.data[i])
		if diff <= h.Delta {
			sum += 0.5 * diff * diff
		} else {
			sum += h.Delta*diff - 0.5*h.Delta*h.Delta
		}
	}
	if h.Reduction == "mean" {
		return sum / float64(len(pred.data))
	}
	return sum
}

func (h *HuberLoss) gradient(pred, target *tensor, gradOut *tensor) {
	scale := 1.0
	if h.Reduction == "mean" {
		scale = 1.0 / float64(len(pred.data))
	}
	for i := range pred.data {
		diff := pred.data[i] - target.data[i]
		if math.Abs(diff) <= h.Delta {
			gradOut.data[i] = scale * diff
		} else if diff > 0 {
			gradOut.data[i] = scale * h.Delta
		} else {
			gradOut.data[i] = -scale * h.Delta
		}
	}
}

func (h *HuberLoss) name() string { return "huber" }

// CrossEntropyLoss - for classification with softmax
type CrossEntropyLoss struct {
	LabelSmoothing float64
}

type CrossEntropyConfig struct {
	LabelSmoothing float64
}

func CrossEntropy(config CrossEntropyConfig) Loss {
	return &CrossEntropyLoss{LabelSmoothing: config.LabelSmoothing}
}

func (c *CrossEntropyLoss) compute(pred, target *tensor) float64 {
	eps := 1e-15
	sum := 0.0
	nClasses := pred.shape[len(pred.shape)-1]
	nSamples := len(pred.data) / nClasses

	for i := 0; i < nSamples; i++ {
		for j := 0; j < nClasses; j++ {
			idx := i*nClasses + j
			t := target.data[idx]
			if c.LabelSmoothing > 0 {
				t = t*(1-c.LabelSmoothing) + c.LabelSmoothing/float64(nClasses)
			}
			p := math.Max(pred.data[idx], eps)
			sum -= t * math.Log(p)
		}
	}
	return sum / float64(nSamples)
}

func (c *CrossEntropyLoss) gradient(pred, target *tensor, gradOut *tensor) {
	nClasses := pred.shape[len(pred.shape)-1]
	nSamples := len(pred.data) / nClasses
	scale := 1.0 / float64(nSamples)

	for i := 0; i < nSamples; i++ {
		for j := 0; j < nClasses; j++ {
			idx := i*nClasses + j
			t := target.data[idx]
			if c.LabelSmoothing > 0 {
				t = t*(1-c.LabelSmoothing) + c.LabelSmoothing/float64(nClasses)
			}
			gradOut.data[idx] = scale * (pred.data[idx] - t)
		}
	}
}

func (c *CrossEntropyLoss) name() string { return "cross_entropy" }

// BinaryCrossEntropyLoss - for binary classification
type BinaryCrossEntropyLoss struct {
	Reduction string
}

type BinaryCrossEntropyConfig struct {
	Reduction string
}

func BinaryCrossEntropy(config BinaryCrossEntropyConfig) Loss {
	return &BinaryCrossEntropyLoss{Reduction: config.Reduction}
}

func (b *BinaryCrossEntropyLoss) compute(pred, target *tensor) float64 {
	eps := 1e-15
	sum := 0.0
	for i := range pred.data {
		p := math.Max(math.Min(pred.data[i], 1-eps), eps)
		t := target.data[i]
		sum -= t*math.Log(p) + (1-t)*math.Log(1-p)
	}
	if b.Reduction == "mean" {
		return sum / float64(len(pred.data))
	}
	return sum
}

func (b *BinaryCrossEntropyLoss) gradient(pred, target *tensor, gradOut *tensor) {
	eps := 1e-7 // Larger epsilon for gradient stability
	scale := 1.0
	if b.Reduction == "mean" {
		scale = 1.0 / float64(len(pred.data))
	}
	for i := range pred.data {
		p := math.Max(math.Min(pred.data[i], 1-eps), eps)
		t := target.data[i]
		// Numerically stable gradient: (p - t) / (p * (1 - p))
		// Clamp denominator to avoid division by near-zero
		denom := math.Max(p*(1-p), eps)
		gradOut.data[i] = scale * (p - t) / denom
	}
}

func (b *BinaryCrossEntropyLoss) name() string { return "binary_cross_entropy" }

// KLDivergenceLoss - Kullback-Leibler divergence
type KLDivergenceLoss struct {
	Reduction string
}

type KLDivConfig struct {
	Reduction string
}

func KLDivergence(config KLDivConfig) Loss {
	return &KLDivergenceLoss{Reduction: config.Reduction}
}

func (k *KLDivergenceLoss) compute(pred, target *tensor) float64 {
	eps := 1e-15
	sum := 0.0
	for i := range pred.data {
		p := math.Max(pred.data[i], eps)
		t := math.Max(target.data[i], eps)
		if t > eps {
			sum += t * (math.Log(t) - math.Log(p))
		}
	}
	if k.Reduction == "mean" {
		return sum / float64(len(pred.data))
	}
	return sum
}

func (k *KLDivergenceLoss) gradient(pred, target *tensor, gradOut *tensor) {
	eps := 1e-7 // Larger epsilon for gradient stability
	scale := 1.0
	if k.Reduction == "mean" {
		scale = 1.0 / float64(len(pred.data))
	}
	for i := range pred.data {
		p := math.Max(pred.data[i], eps)
		t := target.data[i]
		gradOut.data[i] = -scale * t / p
	}
}

func (k *KLDivergenceLoss) name() string { return "kl_divergence" }
