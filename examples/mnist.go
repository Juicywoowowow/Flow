// MNIST Example - Digit classification with Flow
//
// Demonstrates:
// - Multi-layer network
// - Softmax output for multi-class classification
// - Adam optimizer with all hyperparameters
// - Early stopping callback
// - Validation split
package main

import (
	"fmt"
	"log"
	"math/rand"

	flow "flow/src"
)

func main() {
	// Generate synthetic MNIST-like data
	// In real usage, load actual MNIST dataset
	rng := rand.New(rand.NewSource(42))

	numSamples := 1000
	inputDim := 784 // 28x28 flattened
	numClasses := 10

	inputs := make([][]float64, numSamples)
	targets := make([][]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		inputs[i] = make([]float64, inputDim)
		targets[i] = make([]float64, numClasses)

		// Random digit class
		digit := rng.Intn(numClasses)
		targets[i][digit] = 1.0

		// Generate synthetic features (noise + digit-specific pattern)
		for j := 0; j < inputDim; j++ {
			inputs[i][j] = rng.NormFloat64()*0.3 + float64(digit)/10.0*rng.Float64()
		}
	}

	// Build network
	net, err := flow.NewNetwork(flow.NetworkConfig{
		Seed:    123,
		Verbose: true,
	}).
		AddLayer(flow.Dense(256).
			WithActivation(flow.ReLU()).
			WithInitializer(flow.HeNormal(1.0)).
			WithBiasInitializer(flow.Zeros()).
			WithBias(true).
			Build()).
		AddLayer(flow.Dropout(0.3).
			WithSeed(42).
			Build()).
		AddLayer(flow.Dense(128).
			WithActivation(flow.ReLU()).
			WithInitializer(flow.HeNormal(1.0)).
			WithBiasInitializer(flow.Zeros()).
			WithBias(true).
			Build()).
		AddLayer(flow.Dropout(0.2).
			WithSeed(43).
			Build()).
		AddLayer(flow.Dense(numClasses).
			WithActivation(flow.Softmax()).
			WithInitializer(flow.XavierNormal(1.0)).
			WithBiasInitializer(flow.Zeros()).
			WithBias(true).
			Build()).
		Build([]int{inputDim})

	if err != nil {
		log.Fatalf("Failed to build network: %v", err)
	}

	// Compile with Adam optimizer - ALL hyperparameters explicit
	err = net.Compile(flow.CompileConfig{
		Optimizer: flow.Adam(flow.AdamConfig{
			LR:          0.001,
			Beta1:       0.9,
			Beta2:       0.999,
			Epsilon:     1e-8,
			WeightDecay: 0.0001,
			AMSGrad:     false,
		}),
		Loss: flow.CrossEntropy(flow.CrossEntropyConfig{
			LabelSmoothing: 0.1,
		}),
		Metrics: []flow.Metric{
			flow.Accuracy(),
			flow.TopKAccuracy(flow.TopKConfig{K: 3}),
		},
		Regularizer: flow.L2(0.0001),
		GradientClip: flow.GradientClipConfig{
			Mode:     "norm",
			MaxNorm:  1.0,
			MaxValue: 0.0,
		},
	})

	if err != nil {
		log.Fatalf("Failed to compile network: %v", err)
	}

	fmt.Println(net.Summary())

	// Train with early stopping
	result, err := net.Train(inputs, targets, flow.TrainConfig{
		Epochs:          50,
		BatchSize:       32,
		Shuffle:         true,
		ValidationSplit: 0.2,
		Verbose:         1,
	}, []flow.Callback{
		flow.PrintProgress(flow.PrintProgressConfig{PrintEvery: 5}),
		flow.EarlyStopping(flow.EarlyStoppingConfig{
			Monitor:     "val_loss",
			MinDelta:    0.001,
			Patience:    5,
			Mode:        "min",
			RestoreBest: true,
		}),
		flow.History(),
	})

	if err != nil {
		log.Fatalf("Training failed: %v", err)
	}

	fmt.Printf("\nTraining Complete!\n")
	fmt.Printf("Final Loss: %.4f\n", result.FinalLoss)
	fmt.Printf("Final Accuracy: %.2f%%\n", result.FinalMetrics["accuracy"]*100)
	fmt.Printf("Final Top-3 Accuracy: %.2f%%\n", result.FinalMetrics["top_k_accuracy"]*100)

	// Evaluate on test data (using same data for demo)
	evalResults, err := net.Evaluate(inputs[:100], targets[:100])
	if err != nil {
		log.Fatalf("Evaluation failed: %v", err)
	}

	fmt.Printf("\nEvaluation Results:\n")
	for k, v := range evalResults {
		fmt.Printf("  %s: %.4f\n", k, v)
	}

	// Save model
	err = net.Save("mnist_model.json")
	if err != nil {
		log.Printf("Warning: Failed to save model: %v", err)
	} else {
		fmt.Println("\nModel saved to mnist_model.json")
	}
}
