package flow

import (
	"fmt"
	"math"
)

// Callback is called during training at various points
type Callback interface {
	onTrainBegin(logs map[string]float64)
	onTrainEnd(logs map[string]float64)
	onEpochBegin(epoch int, logs map[string]float64)
	onEpochEnd(epoch int, logs map[string]float64) bool // return true to stop training
	onBatchBegin(batch int, logs map[string]float64)
	onBatchEnd(batch int, logs map[string]float64)
	name() string
}

// EarlyStoppingCallback stops training when metric stops improving
type EarlyStoppingCallback struct {
	Monitor      string
	MinDelta     float64
	Patience     int
	Mode         string // "min" or "max"
	RestoreBest  bool
	bestValue    float64
	bestWeights  [][]*tensor
	wait         int
	stoppedEpoch int
}

type EarlyStoppingConfig struct {
	Monitor     string
	MinDelta    float64
	Patience    int
	Mode        string
	RestoreBest bool
}

func EarlyStopping(config EarlyStoppingConfig) Callback {
	best := math.Inf(1)
	if config.Mode == "max" {
		best = math.Inf(-1)
	}
	return &EarlyStoppingCallback{
		Monitor:     config.Monitor,
		MinDelta:    config.MinDelta,
		Patience:    config.Patience,
		Mode:        config.Mode,
		RestoreBest: config.RestoreBest,
		bestValue:   best,
		wait:        0,
	}
}

func (e *EarlyStoppingCallback) onTrainBegin(logs map[string]float64) {
	e.wait = 0
	if e.Mode == "max" {
		e.bestValue = math.Inf(-1)
	} else {
		e.bestValue = math.Inf(1)
	}
}

func (e *EarlyStoppingCallback) onTrainEnd(logs map[string]float64) {}

func (e *EarlyStoppingCallback) onEpochBegin(epoch int, logs map[string]float64) {}

func (e *EarlyStoppingCallback) onEpochEnd(epoch int, logs map[string]float64) bool {
	current, ok := logs[e.Monitor]
	if !ok {
		return false
	}

	improved := false
	if e.Mode == "max" {
		improved = current > e.bestValue+e.MinDelta
	} else {
		improved = current < e.bestValue-e.MinDelta
	}

	if improved {
		e.bestValue = current
		e.wait = 0
	} else {
		e.wait++
		if e.wait >= e.Patience {
			e.stoppedEpoch = epoch
			return true // stop training
		}
	}
	return false
}

func (e *EarlyStoppingCallback) onBatchBegin(batch int, logs map[string]float64) {}
func (e *EarlyStoppingCallback) onBatchEnd(batch int, logs map[string]float64)   {}
func (e *EarlyStoppingCallback) name() string                                    { return "early_stopping" }

// PrintProgressCallback prints training progress
type PrintProgressCallback struct {
	PrintEvery int
}

type PrintProgressConfig struct {
	PrintEvery int
}

func PrintProgress(config PrintProgressConfig) Callback {
	return &PrintProgressCallback{PrintEvery: config.PrintEvery}
}

func (p *PrintProgressCallback) onTrainBegin(logs map[string]float64) {
	fmt.Println("Training started...")
}

func (p *PrintProgressCallback) onTrainEnd(logs map[string]float64) {
	fmt.Println("Training complete.")
}

func (p *PrintProgressCallback) onEpochBegin(epoch int, logs map[string]float64) {}

func (p *PrintProgressCallback) onEpochEnd(epoch int, logs map[string]float64) bool {
	if (epoch+1)%p.PrintEvery == 0 {
		fmt.Printf("Epoch %d:", epoch+1)
		for k, v := range logs {
			fmt.Printf(" %s=%.4f", k, v)
		}
		fmt.Println()
	}
	return false
}

func (p *PrintProgressCallback) onBatchBegin(batch int, logs map[string]float64) {}
func (p *PrintProgressCallback) onBatchEnd(batch int, logs map[string]float64)   {}
func (p *PrintProgressCallback) name() string                                    { return "print_progress" }

// HistoryCallback records training history
type HistoryCallback struct {
	History map[string][]float64
}

func History() Callback {
	return &HistoryCallback{
		History: make(map[string][]float64),
	}
}

func (h *HistoryCallback) onTrainBegin(logs map[string]float64) {
	h.History = make(map[string][]float64)
}

func (h *HistoryCallback) onTrainEnd(logs map[string]float64) {}

func (h *HistoryCallback) onEpochBegin(epoch int, logs map[string]float64) {}

func (h *HistoryCallback) onEpochEnd(epoch int, logs map[string]float64) bool {
	for k, v := range logs {
		h.History[k] = append(h.History[k], v)
	}
	return false
}

func (h *HistoryCallback) onBatchBegin(batch int, logs map[string]float64) {}
func (h *HistoryCallback) onBatchEnd(batch int, logs map[string]float64)   {}
func (h *HistoryCallback) name() string                                    { return "history" }

// LRSchedulerCallback applies learning rate schedule
type LRSchedulerCallback struct {
	Scheduler Scheduler
	InitialLR float64
	currentLR float64
}

type LRSchedulerConfig struct {
	Scheduler Scheduler
	InitialLR float64
}

func LRSchedulerCallback_(config LRSchedulerConfig) Callback {
	return &LRSchedulerCallback{
		Scheduler: config.Scheduler,
		InitialLR: config.InitialLR,
		currentLR: config.InitialLR,
	}
}

func (l *LRSchedulerCallback) onTrainBegin(logs map[string]float64) {
	l.currentLR = l.InitialLR
}

func (l *LRSchedulerCallback) onTrainEnd(logs map[string]float64) {}

func (l *LRSchedulerCallback) onEpochBegin(epoch int, logs map[string]float64) {
	l.currentLR = l.Scheduler.step(epoch, l.currentLR)
	logs["lr"] = l.currentLR
}

func (l *LRSchedulerCallback) onEpochEnd(epoch int, logs map[string]float64) bool {
	return false
}

func (l *LRSchedulerCallback) onBatchBegin(batch int, logs map[string]float64) {}
func (l *LRSchedulerCallback) onBatchEnd(batch int, logs map[string]float64)   {}
func (l *LRSchedulerCallback) name() string                                    { return "lr_scheduler" }

// GradientClippingCallback clips gradients
type GradientClippingCallback struct {
	MaxNorm   float64
	ClipValue float64
	Mode      string // "norm" or "value"
}

type GradientClippingConfig struct {
	MaxNorm   float64
	ClipValue float64
	Mode      string
}

func GradientClipping(config GradientClippingConfig) Callback {
	return &GradientClippingCallback{
		MaxNorm:   config.MaxNorm,
		ClipValue: config.ClipValue,
		Mode:      config.Mode,
	}
}

func (g *GradientClippingCallback) onTrainBegin(logs map[string]float64)               {}
func (g *GradientClippingCallback) onTrainEnd(logs map[string]float64)                 {}
func (g *GradientClippingCallback) onEpochBegin(epoch int, logs map[string]float64)    {}
func (g *GradientClippingCallback) onEpochEnd(epoch int, logs map[string]float64) bool { return false }
func (g *GradientClippingCallback) onBatchBegin(batch int, logs map[string]float64)    {}
func (g *GradientClippingCallback) onBatchEnd(batch int, logs map[string]float64)      {}
func (g *GradientClippingCallback) name() string                                       { return "gradient_clipping" }
