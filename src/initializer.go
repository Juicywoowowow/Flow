package flow

import (
	"math"
	"math/rand"
)

// Initializer sets up initial weights for layers
type Initializer interface {
	initialize(t *tensor, fanIn, fanOut int, rng *rand.Rand)
	name() string
}

// HeNormalInit - He/Kaiming normal initialization
type HeNormalInit struct {
	Gain float64
}

func HeNormal(gain float64) Initializer {
	return &HeNormalInit{Gain: gain}
}

func (h *HeNormalInit) initialize(t *tensor, fanIn, fanOut int, rng *rand.Rand) {
	std := h.Gain * math.Sqrt(2.0/float64(fanIn))
	t.fillRandNorm(0, std, rng)
}

func (h *HeNormalInit) name() string { return "he_normal" }

// HeUniformInit - He/Kaiming uniform initialization
type HeUniformInit struct {
	Gain float64
}

func HeUniform(gain float64) Initializer {
	return &HeUniformInit{Gain: gain}
}

func (h *HeUniformInit) initialize(t *tensor, fanIn, fanOut int, rng *rand.Rand) {
	limit := h.Gain * math.Sqrt(6.0/float64(fanIn))
	t.fillRandUniform(-limit, limit, rng)
}

func (h *HeUniformInit) name() string { return "he_uniform" }

// XavierNormalInit - Xavier/Glorot normal initialization
type XavierNormalInit struct {
	Gain float64
}

func XavierNormal(gain float64) Initializer {
	return &XavierNormalInit{Gain: gain}
}

func (x *XavierNormalInit) initialize(t *tensor, fanIn, fanOut int, rng *rand.Rand) {
	std := x.Gain * math.Sqrt(2.0/float64(fanIn+fanOut))
	t.fillRandNorm(0, std, rng)
}

func (x *XavierNormalInit) name() string { return "xavier_normal" }

// XavierUniformInit - Xavier/Glorot uniform initialization
type XavierUniformInit struct {
	Gain float64
}

func XavierUniform(gain float64) Initializer {
	return &XavierUniformInit{Gain: gain}
}

func (x *XavierUniformInit) initialize(t *tensor, fanIn, fanOut int, rng *rand.Rand) {
	limit := x.Gain * math.Sqrt(6.0/float64(fanIn+fanOut))
	t.fillRandUniform(-limit, limit, rng)
}

func (x *XavierUniformInit) name() string { return "xavier_uniform" }

// LeCunNormalInit - LeCun normal initialization
type LeCunNormalInit struct {
	Gain float64
}

func LeCunNormal(gain float64) Initializer {
	return &LeCunNormalInit{Gain: gain}
}

func (l *LeCunNormalInit) initialize(t *tensor, fanIn, fanOut int, rng *rand.Rand) {
	std := l.Gain * math.Sqrt(1.0/float64(fanIn))
	t.fillRandNorm(0, std, rng)
}

func (l *LeCunNormalInit) name() string { return "lecun_normal" }

// LeCunUniformInit - LeCun uniform initialization
type LeCunUniformInit struct {
	Gain float64
}

func LeCunUniform(gain float64) Initializer {
	return &LeCunUniformInit{Gain: gain}
}

func (l *LeCunUniformInit) initialize(t *tensor, fanIn, fanOut int, rng *rand.Rand) {
	limit := l.Gain * math.Sqrt(3.0/float64(fanIn))
	t.fillRandUniform(-limit, limit, rng)
}

func (l *LeCunUniformInit) name() string { return "lecun_uniform" }

// ZerosInit - initialize with zeros
type ZerosInit struct{}

func Zeros() Initializer { return &ZerosInit{} }

func (z *ZerosInit) initialize(t *tensor, fanIn, fanOut int, rng *rand.Rand) {
	t.fill(0)
}

func (z *ZerosInit) name() string { return "zeros" }

// OnesInit - initialize with ones
type OnesInit struct{}

func Ones() Initializer { return &OnesInit{} }

func (o *OnesInit) initialize(t *tensor, fanIn, fanOut int, rng *rand.Rand) {
	t.fill(1)
}

func (o *OnesInit) name() string { return "ones" }

// ConstantInit - initialize with constant value
type ConstantInit struct {
	Value float64
}

func Constant(value float64) Initializer {
	return &ConstantInit{Value: value}
}

func (c *ConstantInit) initialize(t *tensor, fanIn, fanOut int, rng *rand.Rand) {
	t.fill(c.Value)
}

func (c *ConstantInit) name() string { return "constant" }

// RandomNormalInit - simple random normal
type RandomNormalInit struct {
	Mean   float64
	StdDev float64
}

func RandomNormal(mean, stddev float64) Initializer {
	return &RandomNormalInit{Mean: mean, StdDev: stddev}
}

func (r *RandomNormalInit) initialize(t *tensor, fanIn, fanOut int, rng *rand.Rand) {
	t.fillRandNorm(r.Mean, r.StdDev, rng)
}

func (r *RandomNormalInit) name() string { return "random_normal" }

// RandomUniformInit - simple random uniform
type RandomUniformInit struct {
	MinVal float64
	MaxVal float64
}

func RandomUniform(minVal, maxVal float64) Initializer {
	return &RandomUniformInit{MinVal: minVal, MaxVal: maxVal}
}

func (r *RandomUniformInit) initialize(t *tensor, fanIn, fanOut int, rng *rand.Rand) {
	t.fillRandUniform(r.MinVal, r.MaxVal, rng)
}

func (r *RandomUniformInit) name() string { return "random_uniform" }
