package flow

import (
	"fmt"
	"math/rand"
)

// shuffleData shuffles input and target data in-place
func shuffleData(inputs, targets *tensor, rng *rand.Rand) {
	n := inputs.shape[0]
	inputCols := inputs.size() / n
	targetCols := targets.size() / n

	for i := n - 1; i > 0; i-- {
		j := rng.Intn(i + 1)
		// Swap rows in inputs
		for k := 0; k < inputCols; k++ {
			inputs.data[i*inputCols+k], inputs.data[j*inputCols+k] =
				inputs.data[j*inputCols+k], inputs.data[i*inputCols+k]
		}
		// Swap rows in targets
		for k := 0; k < targetCols; k++ {
			targets.data[i*targetCols+k], targets.data[j*targetCols+k] =
				targets.data[j*targetCols+k], targets.data[i*targetCols+k]
		}
	}
}

// getBatch extracts a batch from data
func getBatch(data *tensor, start, batchSize int) *tensor {
	totalSamples := data.shape[0]
	end := start + batchSize
	if end > totalSamples {
		end = totalSamples
	}
	actualBatch := end - start

	var batchShape []int
	if len(data.shape) == 1 {
		batchShape = []int{actualBatch}
	} else {
		batchShape = append([]int{actualBatch}, data.shape[1:]...)
	}

	batch := newTensor(batchShape...)

	elementsPerSample := data.size() / totalSamples
	copy(batch.data, data.data[start*elementsPerSample:end*elementsPerSample])

	return batch
}

// oneHotEncode converts integer labels to one-hot encoding
func oneHotEncode(labels []int, numClasses int) *tensor {
	n := len(labels)
	out := newTensor(n, numClasses)
	for i, label := range labels {
		out.data[i*numClasses+label] = 1.0
	}
	return out
}

// normalize applies z-score normalization
func normalize(data *tensor, mean, std float64) {
	for i := range data.data {
		data.data[i] = (data.data[i] - mean) / std
	}
}

// splitData splits data into train and validation sets
func splitData(inputs, targets *tensor, valSplit float64) (*tensor, *tensor, *tensor, *tensor) {
	n := inputs.shape[0]
	valSize := int(float64(n) * valSplit)
	trainSize := n - valSize

	inputCols := inputs.size() / n
	targetCols := targets.size() / n

	trainInputShape := append([]int{trainSize}, inputs.shape[1:]...)
	valInputShape := append([]int{valSize}, inputs.shape[1:]...)
	trainTargetShape := append([]int{trainSize}, targets.shape[1:]...)
	valTargetShape := append([]int{valSize}, targets.shape[1:]...)

	trainInputs := newTensor(trainInputShape...)
	valInputs := newTensor(valInputShape...)
	trainTargets := newTensor(trainTargetShape...)
	valTargets := newTensor(valTargetShape...)

	copy(trainInputs.data, inputs.data[:trainSize*inputCols])
	copy(valInputs.data, inputs.data[trainSize*inputCols:])
	copy(trainTargets.data, targets.data[:trainSize*targetCols])
	copy(valTargets.data, targets.data[trainSize*targetCols:])

	return trainInputs, trainTargets, valInputs, valTargets
}

// errorf creates a formatted error
func errorf(format string, args ...interface{}) error {
	return fmt.Errorf("flow: "+format, args...)
}

// min returns the minimum of two ints
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// max returns the maximum of two ints
func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
