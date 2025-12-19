// Regression Example - Function approximation with Flow
//
// Demonstrates:
// - Regression with linear output
// - MSE loss
// - Learning rate scheduling
// - RMSprop optimizer
package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"

	flow "flow/src"
)

func main() {
	// Generate synthetic data: y = sin(x) + noise
	rng := rand.New(rand.NewSource(99))

	numSamples := 500

	inputs := make([][]float64, numSamples)
	targets := make([][]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		x := rng.Float64()*4*math.Pi - 2*math.Pi // x in [-2π, 2π]
		y := math.Sin(x) + rng.NormFloat64()*0.1 // sin(x) + noise

		inputs[i] = []float64{x}
		targets[i] = []float64{y}
	}

	// Build regression network
	net, err := flow.NewNetwork(flow.NetworkConfig{
		Seed:    777,
		Verbose: true,
	}).
		AddLayer(flow.Dense(64).
			WithActivation(flow.Tanh()).
			WithInitializer(flow.XavierUniform(1.0)).
			WithBiasInitializer(flow.Zeros()).
			WithBias(true).
			Build()).
		AddLayer(flow.Dense(32).
			WithActivation(flow.Tanh()).
			WithInitializer(flow.XavierUniform(1.0)).
			WithBiasInitializer(flow.Zeros()).
			WithBias(true).
			Build()).
		AddLayer(flow.Dense(16).
			WithActivation(flow.ReLU()).
			WithInitializer(flow.HeUniform(1.0)).
			WithBiasInitializer(flow.Zeros()).
			WithBias(true).
			Build()).
		AddLayer(flow.Dense(1).
			WithActivation(flow.Linear()).
			WithInitializer(flow.XavierUniform(1.0)).
			WithBiasInitializer(flow.Zeros()).
			WithBias(true).
			Build()).
		Build([]int{1})

	if err != nil {
		log.Fatalf("Failed to build network: %v", err)
	}

	// Compile with RMSprop - all hyperparameters explicit
	err = net.Compile(flow.CompileConfig{
		Optimizer: flow.RMSprop(flow.RMSpropConfig{
			LR:          0.01,
			Alpha:       0.99,
			Epsilon:     1e-8,
			WeightDecay: 0.0,
			Momentum:    0.0,
			Centered:    false,
		}),
		Loss: flow.Huber(flow.HuberConfig{
			Delta:     1.0,
			Reduction: "mean",
		}),
		Metrics: []flow.Metric{
			flow.MeanSquaredError(),
			flow.MeanAbsoluteError(),
		},
		Regularizer: flow.L2(0.0001),
		GradientClip: flow.GradientClipConfig{
			Mode:     "value",
			MaxNorm:  0.0,
			MaxValue: 5.0,
		},
	})

	if err != nil {
		log.Fatalf("Failed to compile network: %v", err)
	}

	fmt.Println(net.Summary())

	// Create learning rate scheduler callback
	lrCallback := flow.LRSchedulerCallback_(flow.LRSchedulerConfig{
		Scheduler: flow.CosineAnnealing(flow.CosineAnnealingConfig{
			TMax:   200,
			EtaMin: 0.0001,
			EtaMax: 0.01,
		}),
		InitialLR: 0.01,
	})

	// Train
	result, err := net.Train(inputs, targets, flow.TrainConfig{
		Epochs:          200,
		BatchSize:       16,
		Shuffle:         true,
		ValidationSplit: 0.2,
		Verbose:         1,
	}, []flow.Callback{
		flow.PrintProgress(flow.PrintProgressConfig{PrintEvery: 20}),
		lrCallback,
		flow.History(),
	})

	if err != nil {
		log.Fatalf("Training failed: %v", err)
	}

	fmt.Printf("\nTraining Complete!\n")
	fmt.Printf("Final Loss: %.6f\n", result.FinalLoss)
	fmt.Printf("Final MSE: %.6f\n", result.FinalMetrics["mse"])
	fmt.Printf("Final MAE: %.6f\n", result.FinalMetrics["mae"])

	// Test predictions at specific points
	testPoints := [][]float64{
		{0},
		{math.Pi / 2},
		{math.Pi},
		{3 * math.Pi / 2},
		{2 * math.Pi},
	}

	predictions, err := net.Predict(testPoints)
	if err != nil {
		log.Fatalf("Prediction failed: %v", err)
	}

	fmt.Println("\nPredictions vs Actual:")
	fmt.Println("  x          | Predicted | sin(x)    | Error")
	fmt.Println("  -----------+-----------+-----------+---------")
	for i, p := range predictions {
		x := testPoints[i][0]
		actual := math.Sin(x)
		pred := p[0]
		err := math.Abs(pred - actual)
		fmt.Printf("  %9.4f | %9.4f | %9.4f | %.4f\n", x, pred, actual, err)
	}
}
