package flow

import "math"

// Scheduler adjusts learning rate during training
type Scheduler interface {
	step(epoch int, currentLR float64) float64
	name() string
}

// StepDecayScheduler - drops LR by factor every N epochs
type StepDecayScheduler struct {
	StepSize int
	Gamma    float64
}

type StepDecayConfig struct {
	StepSize int
	Gamma    float64
}

func StepDecay(config StepDecayConfig) Scheduler {
	return &StepDecayScheduler{
		StepSize: config.StepSize,
		Gamma:    config.Gamma,
	}
}

func (s *StepDecayScheduler) step(epoch int, currentLR float64) float64 {
	return currentLR * math.Pow(s.Gamma, float64(epoch/s.StepSize))
}

func (s *StepDecayScheduler) name() string { return "step_decay" }

// ExponentialDecayScheduler - exponential decay each epoch
type ExponentialDecayScheduler struct {
	Gamma float64
}

type ExponentialDecayConfig struct {
	Gamma float64
}

func ExponentialDecay(config ExponentialDecayConfig) Scheduler {
	return &ExponentialDecayScheduler{Gamma: config.Gamma}
}

func (e *ExponentialDecayScheduler) step(epoch int, currentLR float64) float64 {
	return currentLR * e.Gamma
}

func (e *ExponentialDecayScheduler) name() string { return "exponential_decay" }

// CosineAnnealingScheduler - cosine annealing
type CosineAnnealingScheduler struct {
	TMax   int
	EtaMin float64
	EtaMax float64
}

type CosineAnnealingConfig struct {
	TMax   int
	EtaMin float64
	EtaMax float64
}

func CosineAnnealing(config CosineAnnealingConfig) Scheduler {
	return &CosineAnnealingScheduler{
		TMax:   config.TMax,
		EtaMin: config.EtaMin,
		EtaMax: config.EtaMax,
	}
}

func (c *CosineAnnealingScheduler) step(epoch int, currentLR float64) float64 {
	return c.EtaMin + 0.5*(c.EtaMax-c.EtaMin)*(1+math.Cos(math.Pi*float64(epoch)/float64(c.TMax)))
}

func (c *CosineAnnealingScheduler) name() string { return "cosine_annealing" }

// WarmRestartsScheduler - cosine annealing with warm restarts
type WarmRestartsScheduler struct {
	T0       int
	TMult    int
	EtaMin   float64
	EtaMax   float64
	currT    int
	currTMax int
}

type WarmRestartsConfig struct {
	T0     int
	TMult  int
	EtaMin float64
	EtaMax float64
}

func WarmRestarts(config WarmRestartsConfig) Scheduler {
	return &WarmRestartsScheduler{
		T0:       config.T0,
		TMult:    config.TMult,
		EtaMin:   config.EtaMin,
		EtaMax:   config.EtaMax,
		currT:    0,
		currTMax: config.T0,
	}
}

func (w *WarmRestartsScheduler) step(epoch int, currentLR float64) float64 {
	w.currT++
	if w.currT >= w.currTMax {
		w.currT = 0
		w.currTMax *= w.TMult
	}
	return w.EtaMin + 0.5*(w.EtaMax-w.EtaMin)*(1+math.Cos(math.Pi*float64(w.currT)/float64(w.currTMax)))
}

func (w *WarmRestartsScheduler) name() string { return "warm_restarts" }

// LinearDecayScheduler - linear decay from start to end
type LinearDecayScheduler struct {
	StartLR     float64
	EndLR       float64
	TotalEpochs int
}

type LinearDecayConfig struct {
	StartLR     float64
	EndLR       float64
	TotalEpochs int
}

func LinearDecay(config LinearDecayConfig) Scheduler {
	return &LinearDecayScheduler{
		StartLR:     config.StartLR,
		EndLR:       config.EndLR,
		TotalEpochs: config.TotalEpochs,
	}
}

func (l *LinearDecayScheduler) step(epoch int, currentLR float64) float64 {
	if epoch >= l.TotalEpochs {
		return l.EndLR
	}
	return l.StartLR + (l.EndLR-l.StartLR)*float64(epoch)/float64(l.TotalEpochs)
}

func (l *LinearDecayScheduler) name() string { return "linear_decay" }

// PolynomialDecayScheduler - polynomial decay
type PolynomialDecayScheduler struct {
	StartLR     float64
	EndLR       float64
	Power       float64
	TotalEpochs int
}

type PolynomialDecayConfig struct {
	StartLR     float64
	EndLR       float64
	Power       float64
	TotalEpochs int
}

func PolynomialDecay(config PolynomialDecayConfig) Scheduler {
	return &PolynomialDecayScheduler{
		StartLR:     config.StartLR,
		EndLR:       config.EndLR,
		Power:       config.Power,
		TotalEpochs: config.TotalEpochs,
	}
}

func (p *PolynomialDecayScheduler) step(epoch int, currentLR float64) float64 {
	if epoch >= p.TotalEpochs {
		return p.EndLR
	}
	decay := math.Pow(1-float64(epoch)/float64(p.TotalEpochs), p.Power)
	return (p.StartLR-p.EndLR)*decay + p.EndLR
}

func (p *PolynomialDecayScheduler) name() string { return "polynomial_decay" }

// ConstantScheduler - no change to learning rate
type ConstantScheduler struct{}

func ConstantLR() Scheduler { return &ConstantScheduler{} }

func (c *ConstantScheduler) step(epoch int, currentLR float64) float64 {
	return currentLR
}

func (c *ConstantScheduler) name() string { return "constant" }

// WarmupScheduler - linear warmup then constant
type WarmupScheduler struct {
	WarmupEpochs int
	TargetLR     float64
	InitialLR    float64
}

type WarmupConfig struct {
	WarmupEpochs int
	TargetLR     float64
	InitialLR    float64
}

func Warmup(config WarmupConfig) Scheduler {
	return &WarmupScheduler{
		WarmupEpochs: config.WarmupEpochs,
		TargetLR:     config.TargetLR,
		InitialLR:    config.InitialLR,
	}
}

func (w *WarmupScheduler) step(epoch int, currentLR float64) float64 {
	if epoch >= w.WarmupEpochs {
		return w.TargetLR
	}
	return w.InitialLR + (w.TargetLR-w.InitialLR)*float64(epoch)/float64(w.WarmupEpochs)
}

func (w *WarmupScheduler) name() string { return "warmup" }
