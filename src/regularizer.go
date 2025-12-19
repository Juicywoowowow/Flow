package flow

// Regularizer applies regularization to weights
type Regularizer interface {
	loss(weights *tensor) float64
	gradient(weights *tensor, grad *tensor)
	name() string
}

// L1Regularizer - Lasso regularization
type L1Regularizer struct {
	Lambda float64
}

func L1(lambda float64) Regularizer {
	return &L1Regularizer{Lambda: lambda}
}

func (l *L1Regularizer) loss(weights *tensor) float64 {
	sum := 0.0
	for _, v := range weights.data {
		if v > 0 {
			sum += v
		} else {
			sum -= v
		}
	}
	return l.Lambda * sum
}

func (l *L1Regularizer) gradient(weights *tensor, grad *tensor) {
	for i, v := range weights.data {
		if v > 0 {
			grad.data[i] += l.Lambda
		} else if v < 0 {
			grad.data[i] -= l.Lambda
		}
	}
}

func (l *L1Regularizer) name() string { return "l1" }

// L2Regularizer - Ridge regularization
type L2Regularizer struct {
	Lambda float64
}

func L2(lambda float64) Regularizer {
	return &L2Regularizer{Lambda: lambda}
}

func (l *L2Regularizer) loss(weights *tensor) float64 {
	sum := 0.0
	for _, v := range weights.data {
		sum += v * v
	}
	return 0.5 * l.Lambda * sum
}

func (l *L2Regularizer) gradient(weights *tensor, grad *tensor) {
	for i, v := range weights.data {
		grad.data[i] += l.Lambda * v
	}
}

func (l *L2Regularizer) name() string { return "l2" }

// ElasticNetRegularizer - L1 + L2
type ElasticNetRegularizer struct {
	L1Lambda float64
	L2Lambda float64
	L1Ratio  float64
}

func ElasticNet(l1Lambda, l2Lambda, l1Ratio float64) Regularizer {
	return &ElasticNetRegularizer{
		L1Lambda: l1Lambda,
		L2Lambda: l2Lambda,
		L1Ratio:  l1Ratio,
	}
}

func (e *ElasticNetRegularizer) loss(weights *tensor) float64 {
	l1Sum := 0.0
	l2Sum := 0.0
	for _, v := range weights.data {
		if v > 0 {
			l1Sum += v
		} else {
			l1Sum -= v
		}
		l2Sum += v * v
	}
	return e.L1Ratio*e.L1Lambda*l1Sum + (1-e.L1Ratio)*0.5*e.L2Lambda*l2Sum
}

func (e *ElasticNetRegularizer) gradient(weights *tensor, grad *tensor) {
	for i, v := range weights.data {
		// L1 component
		if v > 0 {
			grad.data[i] += e.L1Ratio * e.L1Lambda
		} else if v < 0 {
			grad.data[i] -= e.L1Ratio * e.L1Lambda
		}
		// L2 component
		grad.data[i] += (1 - e.L1Ratio) * e.L2Lambda * v
	}
}

func (e *ElasticNetRegularizer) name() string { return "elastic_net" }

// NoRegularizer - no regularization
type NoRegularizer struct{}

func NoReg() Regularizer { return &NoRegularizer{} }

func (n *NoRegularizer) loss(weights *tensor) float64           { return 0 }
func (n *NoRegularizer) gradient(weights *tensor, grad *tensor) {}
func (n *NoRegularizer) name() string                           { return "none" }
