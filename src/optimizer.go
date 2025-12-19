package flow

import "math"

// Optimizer updates network parameters
type Optimizer interface {
	init(params []*tensor)
	step(params []*tensor, grads []*tensor)
	name() string
}

// SGDOptimizer - Stochastic Gradient Descent
type SGDOptimizer struct {
	LR          float64
	Momentum    float64
	Dampening   float64
	WeightDecay float64
	Nesterov    bool
	velocities  []*tensor
	initialized bool
}

type SGDConfig struct {
	LR          float64
	Momentum    float64
	Dampening   float64
	WeightDecay float64
	Nesterov    bool
}

func SGD(config SGDConfig) Optimizer {
	return &SGDOptimizer{
		LR:          config.LR,
		Momentum:    config.Momentum,
		Dampening:   config.Dampening,
		WeightDecay: config.WeightDecay,
		Nesterov:    config.Nesterov,
	}
}

func (s *SGDOptimizer) init(params []*tensor) {
	s.velocities = make([]*tensor, len(params))
	for i, p := range params {
		s.velocities[i] = newTensor(p.shape...)
	}
	s.initialized = true
}

func (s *SGDOptimizer) step(params []*tensor, grads []*tensor) {
	if !s.initialized {
		s.init(params)
	}
	for i, p := range params {
		g := grads[i]
		v := s.velocities[i]

		for j := range p.data {
			grad := g.data[j]
			if s.WeightDecay != 0 {
				grad += s.WeightDecay * p.data[j]
			}
			if s.Momentum != 0 {
				v.data[j] = s.Momentum*v.data[j] + (1-s.Dampening)*grad
				if s.Nesterov {
					grad = grad + s.Momentum*v.data[j]
				} else {
					grad = v.data[j]
				}
			}
			p.data[j] -= s.LR * grad
		}
	}
}

func (s *SGDOptimizer) name() string { return "sgd" }

// AdamOptimizer - Adaptive Moment Estimation
type AdamOptimizer struct {
	LR          float64
	Beta1       float64
	Beta2       float64
	Epsilon     float64
	WeightDecay float64
	AMSGrad     bool
	m           []*tensor
	v           []*tensor
	vMax        []*tensor
	t           int
	initialized bool
}

type AdamConfig struct {
	LR          float64
	Beta1       float64
	Beta2       float64
	Epsilon     float64
	WeightDecay float64
	AMSGrad     bool
}

func Adam(config AdamConfig) Optimizer {
	return &AdamOptimizer{
		LR:          config.LR,
		Beta1:       config.Beta1,
		Beta2:       config.Beta2,
		Epsilon:     config.Epsilon,
		WeightDecay: config.WeightDecay,
		AMSGrad:     config.AMSGrad,
	}
}

func (a *AdamOptimizer) init(params []*tensor) {
	a.m = make([]*tensor, len(params))
	a.v = make([]*tensor, len(params))
	if a.AMSGrad {
		a.vMax = make([]*tensor, len(params))
	}
	for i, p := range params {
		a.m[i] = newTensor(p.shape...)
		a.v[i] = newTensor(p.shape...)
		if a.AMSGrad {
			a.vMax[i] = newTensor(p.shape...)
		}
	}
	a.t = 0
	a.initialized = true
}

func (a *AdamOptimizer) step(params []*tensor, grads []*tensor) {
	if !a.initialized {
		a.init(params)
	}
	a.t++
	bc1 := 1 - math.Pow(a.Beta1, float64(a.t))
	bc2 := 1 - math.Pow(a.Beta2, float64(a.t))

	for i, p := range params {
		g := grads[i]
		m := a.m[i]
		v := a.v[i]

		for j := range p.data {
			grad := g.data[j]
			if a.WeightDecay != 0 {
				grad += a.WeightDecay * p.data[j]
			}
			m.data[j] = a.Beta1*m.data[j] + (1-a.Beta1)*grad
			v.data[j] = a.Beta2*v.data[j] + (1-a.Beta2)*grad*grad

			mHat := m.data[j] / bc1
			vHat := v.data[j] / bc2

			if a.AMSGrad {
				if vHat > a.vMax[i].data[j] {
					a.vMax[i].data[j] = vHat
				}
				vHat = a.vMax[i].data[j]
			}

			p.data[j] -= a.LR * mHat / (math.Sqrt(vHat) + a.Epsilon)
		}
	}
}

func (a *AdamOptimizer) name() string { return "adam" }

// AdamWOptimizer - Adam with decoupled weight decay
type AdamWOptimizer struct {
	LR          float64
	Beta1       float64
	Beta2       float64
	Epsilon     float64
	WeightDecay float64
	m           []*tensor
	v           []*tensor
	t           int
	initialized bool
}

type AdamWConfig struct {
	LR          float64
	Beta1       float64
	Beta2       float64
	Epsilon     float64
	WeightDecay float64
}

func AdamW(config AdamWConfig) Optimizer {
	return &AdamWOptimizer{
		LR:          config.LR,
		Beta1:       config.Beta1,
		Beta2:       config.Beta2,
		Epsilon:     config.Epsilon,
		WeightDecay: config.WeightDecay,
	}
}

func (a *AdamWOptimizer) init(params []*tensor) {
	a.m = make([]*tensor, len(params))
	a.v = make([]*tensor, len(params))
	for i, p := range params {
		a.m[i] = newTensor(p.shape...)
		a.v[i] = newTensor(p.shape...)
	}
	a.t = 0
	a.initialized = true
}

func (a *AdamWOptimizer) step(params []*tensor, grads []*tensor) {
	if !a.initialized {
		a.init(params)
	}
	a.t++
	bc1 := 1 - math.Pow(a.Beta1, float64(a.t))
	bc2 := 1 - math.Pow(a.Beta2, float64(a.t))

	for i, p := range params {
		g := grads[i]
		m := a.m[i]
		v := a.v[i]

		for j := range p.data {
			// Decoupled weight decay
			p.data[j] -= a.LR * a.WeightDecay * p.data[j]

			grad := g.data[j]
			m.data[j] = a.Beta1*m.data[j] + (1-a.Beta1)*grad
			v.data[j] = a.Beta2*v.data[j] + (1-a.Beta2)*grad*grad

			mHat := m.data[j] / bc1
			vHat := v.data[j] / bc2

			p.data[j] -= a.LR * mHat / (math.Sqrt(vHat) + a.Epsilon)
		}
	}
}

func (a *AdamWOptimizer) name() string { return "adamw" }

// RMSpropOptimizer
type RMSpropOptimizer struct {
	LR          float64
	Alpha       float64
	Epsilon     float64
	WeightDecay float64
	Momentum    float64
	Centered    bool
	v           []*tensor
	g           []*tensor
	buf         []*tensor
	initialized bool
}

type RMSpropConfig struct {
	LR          float64
	Alpha       float64
	Epsilon     float64
	WeightDecay float64
	Momentum    float64
	Centered    bool
}

func RMSprop(config RMSpropConfig) Optimizer {
	return &RMSpropOptimizer{
		LR:          config.LR,
		Alpha:       config.Alpha,
		Epsilon:     config.Epsilon,
		WeightDecay: config.WeightDecay,
		Momentum:    config.Momentum,
		Centered:    config.Centered,
	}
}

func (r *RMSpropOptimizer) init(params []*tensor) {
	r.v = make([]*tensor, len(params))
	r.buf = make([]*tensor, len(params))
	if r.Centered {
		r.g = make([]*tensor, len(params))
	}
	for i, p := range params {
		r.v[i] = newTensor(p.shape...)
		r.buf[i] = newTensor(p.shape...)
		if r.Centered {
			r.g[i] = newTensor(p.shape...)
		}
	}
	r.initialized = true
}

func (r *RMSpropOptimizer) step(params []*tensor, grads []*tensor) {
	if !r.initialized {
		r.init(params)
	}

	for i, p := range params {
		grad := grads[i]
		v := r.v[i]
		buf := r.buf[i]

		for j := range p.data {
			g := grad.data[j]
			if r.WeightDecay != 0 {
				g += r.WeightDecay * p.data[j]
			}

			v.data[j] = r.Alpha*v.data[j] + (1-r.Alpha)*g*g

			var avg float64
			if r.Centered {
				r.g[i].data[j] = r.Alpha*r.g[i].data[j] + (1-r.Alpha)*g
				avg = v.data[j] - r.g[i].data[j]*r.g[i].data[j]
			} else {
				avg = v.data[j]
			}

			if r.Momentum > 0 {
				buf.data[j] = r.Momentum*buf.data[j] + g/(math.Sqrt(avg)+r.Epsilon)
				p.data[j] -= r.LR * buf.data[j]
			} else {
				p.data[j] -= r.LR * g / (math.Sqrt(avg) + r.Epsilon)
			}
		}
	}
}

func (r *RMSpropOptimizer) name() string { return "rmsprop" }

// AdagradOptimizer
type AdagradOptimizer struct {
	LR          float64
	LRDecay     float64
	WeightDecay float64
	Epsilon     float64
	sum         []*tensor
	step_count  int
	initialized bool
}

type AdagradConfig struct {
	LR          float64
	LRDecay     float64
	WeightDecay float64
	Epsilon     float64
}

func Adagrad(config AdagradConfig) Optimizer {
	return &AdagradOptimizer{
		LR:          config.LR,
		LRDecay:     config.LRDecay,
		WeightDecay: config.WeightDecay,
		Epsilon:     config.Epsilon,
	}
}

func (a *AdagradOptimizer) init(params []*tensor) {
	a.sum = make([]*tensor, len(params))
	for i, p := range params {
		a.sum[i] = newTensor(p.shape...)
	}
	a.step_count = 0
	a.initialized = true
}

func (a *AdagradOptimizer) step(params []*tensor, grads []*tensor) {
	if !a.initialized {
		a.init(params)
	}
	a.step_count++
	lr := a.LR / (1 + float64(a.step_count-1)*a.LRDecay)

	for i, p := range params {
		g := grads[i]
		s := a.sum[i]

		for j := range p.data {
			grad := g.data[j]
			if a.WeightDecay != 0 {
				grad += a.WeightDecay * p.data[j]
			}
			s.data[j] += grad * grad
			p.data[j] -= lr * grad / (math.Sqrt(s.data[j]) + a.Epsilon)
		}
	}
}

func (a *AdagradOptimizer) name() string { return "adagrad" }

// LionOptimizer - EvoLved Sign Momentum (Google 2023)
// Smaller memory footprint than Adam, works better with larger batches
type LionOptimizer struct {
	LR          float64
	Beta1       float64
	Beta2       float64
	WeightDecay float64
	m           []*tensor
	initialized bool
}

type LionConfig struct {
	LR          float64
	Beta1       float64
	Beta2       float64
	WeightDecay float64
}

func Lion(config LionConfig) Optimizer {
	return &LionOptimizer{
		LR:          config.LR,
		Beta1:       config.Beta1,
		Beta2:       config.Beta2,
		WeightDecay: config.WeightDecay,
	}
}

func (l *LionOptimizer) init(params []*tensor) {
	l.m = make([]*tensor, len(params))
	for i, p := range params {
		l.m[i] = newTensor(p.shape...)
	}
	l.initialized = true
}

func (l *LionOptimizer) step(params []*tensor, grads []*tensor) {
	if !l.initialized {
		l.init(params)
	}

	for i, p := range params {
		g := grads[i]
		m := l.m[i]

		for j := range p.data {
			grad := g.data[j]

			// Compute update direction: sign(beta1 * m + (1 - beta1) * grad)
			update := l.Beta1*m.data[j] + (1-l.Beta1)*grad
			var sign float64
			if update > 0 {
				sign = 1.0
			} else if update < 0 {
				sign = -1.0
			} else {
				sign = 0.0
			}

			// Update momentum for next step
			m.data[j] = l.Beta2*m.data[j] + (1-l.Beta2)*grad

			// Apply weight decay and update
			p.data[j] -= l.LR * (sign + l.WeightDecay*p.data[j])
		}
	}
}

func (l *LionOptimizer) name() string { return "lion" }

// AdaFactorOptimizer - Memory-efficient optimizer for large models
type AdaFactorOptimizer struct {
	LR             float64
	Beta2Decay     float64
	Epsilon1       float64
	Epsilon2       float64
	ClipThreshold  float64
	WeightDecay    float64
	ScaleParameter bool
	RelativeStep   bool
	v              []*tensor
	step_count     int
	initialized    bool
}

type AdaFactorConfig struct {
	LR             float64
	Beta2Decay     float64
	Epsilon1       float64
	Epsilon2       float64
	ClipThreshold  float64
	WeightDecay    float64
	ScaleParameter bool
	RelativeStep   bool
}

func AdaFactor(config AdaFactorConfig) Optimizer {
	return &AdaFactorOptimizer{
		LR:             config.LR,
		Beta2Decay:     config.Beta2Decay,
		Epsilon1:       config.Epsilon1,
		Epsilon2:       config.Epsilon2,
		ClipThreshold:  config.ClipThreshold,
		WeightDecay:    config.WeightDecay,
		ScaleParameter: config.ScaleParameter,
		RelativeStep:   config.RelativeStep,
	}
}

func (a *AdaFactorOptimizer) init(params []*tensor) {
	a.v = make([]*tensor, len(params))
	for i, p := range params {
		a.v[i] = newTensor(p.shape...)
	}
	a.step_count = 0
	a.initialized = true
}

func (a *AdaFactorOptimizer) step(params []*tensor, grads []*tensor) {
	if !a.initialized {
		a.init(params)
	}
	a.step_count++

	rho := math.Min(a.LR, 1.0/math.Sqrt(float64(a.step_count)))
	beta2 := 1.0 - math.Pow(float64(a.step_count), -a.Beta2Decay)

	for i, p := range params {
		g := grads[i]
		v := a.v[i]

		// Update second moment
		for j := range p.data {
			v.data[j] = beta2*v.data[j] + (1-beta2)*g.data[j]*g.data[j]
		}

		// Compute RMS for clipping
		rms := 0.0
		for j := range v.data {
			rms += v.data[j]
		}
		rms = math.Sqrt(rms/float64(len(v.data))) + a.Epsilon1

		for j := range p.data {
			// Compute update
			denom := math.Sqrt(v.data[j]) + a.Epsilon2
			update := g.data[j] / denom

			// Clip update
			updateRMS := math.Abs(update)
			if updateRMS > a.ClipThreshold*rms {
				update = update * a.ClipThreshold * rms / updateRMS
			}

			// Apply weight decay
			if a.WeightDecay != 0 {
				update += a.WeightDecay * p.data[j]
			}

			p.data[j] -= rho * update
		}
	}
}

func (a *AdaFactorOptimizer) name() string { return "adafactor" }
