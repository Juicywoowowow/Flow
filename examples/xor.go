// XOR Example - Classic XOR problem with Flow
//
// Demonstrates:
// - Small network configuration
// - Binary classification
// - Explicit hyperparameter setting
package main

import (
	"fmt"
	"log"

	flow "flow/src"
)

func main() {
	// XOR input data
	inputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}

	// XOR target outputs
	targets := [][]float64{
		{0},
		{1},
		{1},
		{0},
	}

	// Build network - ALL parameters explicit
	net, err := flow.NewNetwork(flow.NetworkConfig{
		Seed:    42,
		Verbose: true,
	}).
		AddLayer(flow.Dense(8).
			WithActivation(flow.Tanh()).
			WithInitializer(flow.XavierNormal(1.0)).
			WithBiasInitializer(flow.Zeros()).
			WithBias(true).
			Build()).
		AddLayer(flow.Dense(4).
			WithActivation(flow.Tanh()).
			WithInitializer(flow.XavierNormal(1.0)).
			WithBiasInitializer(flow.Zeros()).
			WithBias(true).
			Build()).
		AddLayer(flow.Dense(1).
			WithActivation(flow.Sigmoid()).
			WithInitializer(flow.XavierNormal(1.0)).
			WithBiasInitializer(flow.Zeros()).
			WithBias(true).
			Build()).
		Build([]int{2})

	if err != nil {
		log.Fatalf("Failed to build network: %v", err)
	}

	// Compile with explicit config
	err = net.Compile(flow.CompileConfig{
		Optimizer: flow.SGD(flow.SGDConfig{
			LR:          0.5,
			Momentum:    0.9,
			Dampening:   0.0,
			WeightDecay: 0.0,
			Nesterov:    true,
		}),
		Loss: flow.MSE(flow.MSEConfig{
			Reduction: "mean",
		}),
		Metrics:     []flow.Metric{flow.Accuracy()},
		Regularizer: flow.NoReg(),
		GradientClip: flow.GradientClipConfig{
			Mode:     "none",
			MaxNorm:  0.0,
			MaxValue: 0.0,
		},
	})

	if err != nil {
		log.Fatalf("Failed to compile network: %v", err)
	}

	fmt.Println(net.Summary())

	// Train with explicit config
	result, err := net.Train(inputs, targets, flow.TrainConfig{
		Epochs:          1000,
		BatchSize:       4,
		Shuffle:         true,
		ValidationSplit: 0.0,
		Verbose:         1,
	}, []flow.Callback{
		flow.PrintProgress(flow.PrintProgressConfig{PrintEvery: 100}),
	})

	if err != nil {
		log.Fatalf("Training failed: %v", err)
	}

	fmt.Printf("\nFinal Loss: %.6f\n", result.FinalLoss)
	fmt.Printf("Final Accuracy: %.2f%%\n", result.FinalMetrics["accuracy"]*100)

	// Test predictions
	fmt.Println("\nPredictions:")
	predictions, err := net.Predict(inputs)
	if err != nil {
		log.Fatalf("Prediction failed: %v", err)
	}

	for i, pred := range predictions {
		expected := targets[i][0]
		actual := pred[0]
		rounded := 0.0
		if actual >= 0.5 {
			rounded = 1.0
		}
		status := "✓"
		if rounded != expected {
			status = "✗"
		}
		fmt.Printf("  %v -> %.4f (rounded: %.0f, expected: %.0f) %s\n",
			inputs[i], actual, rounded, expected, status)
	}
}
