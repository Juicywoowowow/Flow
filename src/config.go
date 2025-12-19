package flow

// TrainConfig holds all training configuration - ALL fields required
type TrainConfig struct {
	Epochs                    int
	BatchSize                 int
	Shuffle                   bool
	ValidationSplit           float64
	Verbose                   int
	GradientAccumulationSteps int // Accumulate gradients over N steps for larger effective batch
}

// CompileConfig holds model compilation settings - ALL fields required
type CompileConfig struct {
	Optimizer    Optimizer
	Loss         Loss
	Metrics      []Metric
	Regularizer  Regularizer
	GradientClip GradientClipConfig
}

// GradientClipConfig for gradient clipping
type GradientClipConfig struct {
	Mode     string // "norm", "value", or "none"
	MaxNorm  float64
	MaxValue float64
}

// NetworkConfig for network construction
type NetworkConfig struct {
	Seed    int64
	Verbose bool
}

// DataConfig for data handling
type DataConfig struct {
	Shuffle   bool
	Seed      int64
	Normalize bool
	NormMean  float64
	NormStd   float64
}

// ValidateTrainConfig checks all required fields are set
func ValidateTrainConfig(cfg TrainConfig) error {
	if cfg.Epochs <= 0 {
		return errorf("Epochs must be > 0, got %d", cfg.Epochs)
	}
	if cfg.BatchSize <= 0 {
		return errorf("BatchSize must be > 0, got %d", cfg.BatchSize)
	}
	if cfg.ValidationSplit < 0 || cfg.ValidationSplit >= 1 {
		return errorf("ValidationSplit must be in [0, 1), got %f", cfg.ValidationSplit)
	}
	if cfg.GradientAccumulationSteps < 0 {
		return errorf("GradientAccumulationSteps must be >= 0, got %d", cfg.GradientAccumulationSteps)
	}
	return nil
}

// ValidateCompileConfig checks all required fields are set
func ValidateCompileConfig(cfg CompileConfig) error {
	if cfg.Optimizer == nil {
		return errorf("Optimizer is required")
	}
	if cfg.Loss == nil {
		return errorf("Loss is required")
	}
	if cfg.Regularizer == nil {
		return errorf("Regularizer is required - use NoReg() if not needed")
	}
	if cfg.GradientClip.Mode == "" {
		return errorf("GradientClip.Mode is required - use 'none' if not needed")
	}
	return nil
}
