package flow

import (
	"errors"
	"math"
	"math/rand"
)

// Tensor is the core data structure - internal only, not exposed to users
type tensor struct {
	data   []float64
	shape  []int
	stride []int
	grad   []float64
}

func newTensor(shape ...int) *tensor {
	size := 1
	for _, s := range shape {
		if s <= 0 {
			s = 1 // Ensure non-zero size
		}
		size *= s
	}
	if size <= 0 {
		size = 1 // Minimum size of 1
	}
	stride := make([]int, len(shape))
	for i := len(shape) - 1; i >= 0; i-- {
		if i == len(shape)-1 {
			stride[i] = 1
		} else {
			stride[i] = stride[i+1] * shape[i+1]
		}
	}
	return &tensor{
		data:   make([]float64, size),
		shape:  shape,
		stride: stride,
		grad:   make([]float64, size),
	}
}

func (t *tensor) size() int {
	return len(t.data)
}

func (t *tensor) at(indices ...int) float64 {
	idx := 0
	for i, v := range indices {
		idx += v * t.stride[i]
	}
	return t.data[idx]
}

func (t *tensor) set(value float64, indices ...int) {
	idx := 0
	for i, v := range indices {
		idx += v * t.stride[i]
	}
	t.data[idx] = value
}

func (t *tensor) fill(value float64) {
	for i := range t.data {
		t.data[i] = value
	}
}

func (t *tensor) fillRandNorm(mean, std float64, rng *rand.Rand) {
	for i := range t.data {
		t.data[i] = rng.NormFloat64()*std + mean
	}
}

func (t *tensor) fillRandUniform(low, high float64, rng *rand.Rand) {
	for i := range t.data {
		t.data[i] = rng.Float64()*(high-low) + low
	}
}

func (t *tensor) zeroGrad() {
	for i := range t.grad {
		t.grad[i] = 0
	}
}

func (t *tensor) clone() *tensor {
	nt := newTensor(t.shape...)
	copy(nt.data, t.data)
	copy(nt.grad, t.grad)
	return nt
}

// Matrix operations - optimized for speed, no bounds checking
func matmul(a, b, out *tensor) {
	m := a.shape[0]
	k := a.shape[1]
	n := b.shape[1]

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			for l := 0; l < k; l++ {
				sum += a.data[i*k+l] * b.data[l*n+j]
			}
			out.data[i*n+j] = sum
		}
	}
}

func matmulTransA(a, b, out *tensor) {
	m := a.shape[1]
	k := a.shape[0]
	n := b.shape[1]

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			for l := 0; l < k; l++ {
				sum += a.data[l*m+i] * b.data[l*n+j]
			}
			out.data[i*n+j] = sum
		}
	}
}

func matmulTransB(a, b, out *tensor) {
	m := a.shape[0]
	k := a.shape[1]
	n := b.shape[0]

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			for l := 0; l < k; l++ {
				sum += a.data[i*k+l] * b.data[j*k+l]
			}
			out.data[i*n+j] = sum
		}
	}
}

func addVec(a *tensor, b *tensor) {
	for i := range a.data {
		a.data[i] += b.data[i%len(b.data)]
	}
}

func addScalar(a *tensor, s float64) {
	for i := range a.data {
		a.data[i] += s
	}
}

func mulScalar(a *tensor, s float64) {
	for i := range a.data {
		a.data[i] *= s
	}
}

func elemMul(a, b, out *tensor) {
	for i := range a.data {
		out.data[i] = a.data[i] * b.data[i]
	}
}

func elemAdd(a, b, out *tensor) {
	for i := range a.data {
		out.data[i] = a.data[i] + b.data[i]
	}
}

func elemSub(a, b, out *tensor) {
	for i := range a.data {
		out.data[i] = a.data[i] - b.data[i]
	}
}

func sumAxis0(a *tensor, out *tensor) {
	rows := a.shape[0]
	cols := a.shape[1]
	for j := 0; j < cols; j++ {
		sum := 0.0
		for i := 0; i < rows; i++ {
			sum += a.data[i*cols+j]
		}
		out.data[j] = sum
	}
}

func clip(a *tensor, min, max float64) {
	for i := range a.data {
		if a.data[i] < min {
			a.data[i] = min
		} else if a.data[i] > max {
			a.data[i] = max
		}
	}
}

func maxVal(a *tensor) float64 {
	if len(a.data) == 0 {
		return 0 // Safe default for empty tensor
	}
	m := a.data[0]
	for _, v := range a.data[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

func l2Norm(a *tensor) float64 {
	sum := 0.0
	for _, v := range a.data {
		sum += v * v
	}
	return math.Sqrt(sum)
}

func validateShape(expected, got []int) error {
	if len(expected) != len(got) {
		return errors.New("flow: shape mismatch - different dimensions")
	}
	for i := range expected {
		if expected[i] != got[i] {
			return errors.New("flow: shape mismatch")
		}
	}
	return nil
}
