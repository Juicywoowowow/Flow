package flow

import "math"

// Metric computes evaluation metrics
type Metric interface {
	reset()
	update(pred, target *tensor)
	result() float64
	name() string
}

// AccuracyMetric - classification accuracy
type AccuracyMetric struct {
	correct int
	total   int
}

func Accuracy() Metric {
	return &AccuracyMetric{}
}

func (a *AccuracyMetric) reset() {
	a.correct = 0
	a.total = 0
}

func (a *AccuracyMetric) update(pred, target *tensor) {
	if len(pred.shape) == 1 {
		// Binary classification
		for i := range pred.data {
			predClass := 0
			if pred.data[i] >= 0.5 {
				predClass = 1
			}
			targetClass := int(target.data[i])
			if predClass == targetClass {
				a.correct++
			}
			a.total++
		}
	} else {
		// Multi-class
		batchSize := pred.shape[0]
		numClasses := pred.shape[1]
		for i := 0; i < batchSize; i++ {
			predClass := 0
			maxVal := pred.data[i*numClasses]
			for j := 1; j < numClasses; j++ {
				if pred.data[i*numClasses+j] > maxVal {
					maxVal = pred.data[i*numClasses+j]
					predClass = j
				}
			}
			targetClass := 0
			maxValT := target.data[i*numClasses]
			for j := 1; j < numClasses; j++ {
				if target.data[i*numClasses+j] > maxValT {
					maxValT = target.data[i*numClasses+j]
					targetClass = j
				}
			}
			if predClass == targetClass {
				a.correct++
			}
			a.total++
		}
	}
}

func (a *AccuracyMetric) result() float64 {
	if a.total == 0 {
		return 0
	}
	return float64(a.correct) / float64(a.total)
}

func (a *AccuracyMetric) name() string { return "accuracy" }

// PrecisionMetric - precision for binary classification
type PrecisionMetric struct {
	truePositives  int
	falsePositives int
	Threshold      float64
}

type PrecisionConfig struct {
	Threshold float64
}

func Precision(config PrecisionConfig) Metric {
	return &PrecisionMetric{Threshold: config.Threshold}
}

func (p *PrecisionMetric) reset() {
	p.truePositives = 0
	p.falsePositives = 0
}

func (p *PrecisionMetric) update(pred, target *tensor) {
	for i := range pred.data {
		predPos := pred.data[i] >= p.Threshold
		actualPos := target.data[i] >= 0.5
		if predPos {
			if actualPos {
				p.truePositives++
			} else {
				p.falsePositives++
			}
		}
	}
}

func (p *PrecisionMetric) result() float64 {
	denom := p.truePositives + p.falsePositives
	if denom == 0 {
		return 0
	}
	return float64(p.truePositives) / float64(denom)
}

func (p *PrecisionMetric) name() string { return "precision" }

// RecallMetric - recall for binary classification
type RecallMetric struct {
	truePositives  int
	falseNegatives int
	Threshold      float64
}

type RecallConfig struct {
	Threshold float64
}

func Recall(config RecallConfig) Metric {
	return &RecallMetric{Threshold: config.Threshold}
}

func (r *RecallMetric) reset() {
	r.truePositives = 0
	r.falseNegatives = 0
}

func (r *RecallMetric) update(pred, target *tensor) {
	for i := range pred.data {
		predPos := pred.data[i] >= r.Threshold
		actualPos := target.data[i] >= 0.5
		if actualPos {
			if predPos {
				r.truePositives++
			} else {
				r.falseNegatives++
			}
		}
	}
}

func (r *RecallMetric) result() float64 {
	denom := r.truePositives + r.falseNegatives
	if denom == 0 {
		return 0
	}
	return float64(r.truePositives) / float64(denom)
}

func (r *RecallMetric) name() string { return "recall" }

// F1ScoreMetric - F1 score (harmonic mean of precision and recall)
type F1ScoreMetric struct {
	precision *PrecisionMetric
	recall    *RecallMetric
}

type F1Config struct {
	Threshold float64
}

func F1Score(config F1Config) Metric {
	return &F1ScoreMetric{
		precision: &PrecisionMetric{Threshold: config.Threshold},
		recall:    &RecallMetric{Threshold: config.Threshold},
	}
}

func (f *F1ScoreMetric) reset() {
	f.precision.reset()
	f.recall.reset()
}

func (f *F1ScoreMetric) update(pred, target *tensor) {
	f.precision.update(pred, target)
	f.recall.update(pred, target)
}

func (f *F1ScoreMetric) result() float64 {
	p := f.precision.result()
	r := f.recall.result()
	if p+r == 0 {
		return 0
	}
	return 2 * p * r / (p + r)
}

func (f *F1ScoreMetric) name() string { return "f1_score" }

// MeanSquaredErrorMetric
type MeanSquaredErrorMetric struct {
	sum   float64
	count int
}

func MeanSquaredError() Metric {
	return &MeanSquaredErrorMetric{}
}

func (m *MeanSquaredErrorMetric) reset() {
	m.sum = 0
	m.count = 0
}

func (m *MeanSquaredErrorMetric) update(pred, target *tensor) {
	for i := range pred.data {
		diff := pred.data[i] - target.data[i]
		m.sum += diff * diff
		m.count++
	}
}

func (m *MeanSquaredErrorMetric) result() float64 {
	if m.count == 0 {
		return 0
	}
	return m.sum / float64(m.count)
}

func (m *MeanSquaredErrorMetric) name() string { return "mse" }

// MeanAbsoluteErrorMetric
type MeanAbsoluteErrorMetric struct {
	sum   float64
	count int
}

func MeanAbsoluteError() Metric {
	return &MeanAbsoluteErrorMetric{}
}

func (m *MeanAbsoluteErrorMetric) reset() {
	m.sum = 0
	m.count = 0
}

func (m *MeanAbsoluteErrorMetric) update(pred, target *tensor) {
	for i := range pred.data {
		m.sum += math.Abs(pred.data[i] - target.data[i])
		m.count++
	}
}

func (m *MeanAbsoluteErrorMetric) result() float64 {
	if m.count == 0 {
		return 0
	}
	return m.sum / float64(m.count)
}

func (m *MeanAbsoluteErrorMetric) name() string { return "mae" }

// TopKAccuracyMetric
type TopKAccuracyMetric struct {
	K       int
	correct int
	total   int
}

type TopKConfig struct {
	K int
}

func TopKAccuracy(config TopKConfig) Metric {
	return &TopKAccuracyMetric{K: config.K}
}

func (t *TopKAccuracyMetric) reset() {
	t.correct = 0
	t.total = 0
}

func (t *TopKAccuracyMetric) update(pred, target *tensor) {
	batchSize := pred.shape[0]
	numClasses := pred.shape[1]

	for i := 0; i < batchSize; i++ {
		// Find target class
		targetClass := 0
		maxT := target.data[i*numClasses]
		for j := 1; j < numClasses; j++ {
			if target.data[i*numClasses+j] > maxT {
				maxT = target.data[i*numClasses+j]
				targetClass = j
			}
		}

		// Find top-k predictions
		indices := make([]int, numClasses)
		for j := 0; j < numClasses; j++ {
			indices[j] = j
		}
		// Simple selection sort for top K
		for k := 0; k < t.K && k < numClasses; k++ {
			maxIdx := k
			for j := k + 1; j < numClasses; j++ {
				if pred.data[i*numClasses+indices[j]] > pred.data[i*numClasses+indices[maxIdx]] {
					maxIdx = j
				}
			}
			indices[k], indices[maxIdx] = indices[maxIdx], indices[k]
		}

		// Check if target in top K
		for k := 0; k < t.K && k < numClasses; k++ {
			if indices[k] == targetClass {
				t.correct++
				break
			}
		}
		t.total++
	}
}

func (t *TopKAccuracyMetric) result() float64 {
	if t.total == 0 {
		return 0
	}
	return float64(t.correct) / float64(t.total)
}

func (t *TopKAccuracyMetric) name() string { return "top_k_accuracy" }
