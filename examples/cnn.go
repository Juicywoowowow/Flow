// CNN Example - Convolutional Neural Network with Flow
//
// Demonstrates:
// - Conv2D layers
// - MaxPool2D layers
// - Flatten layer
// - Building a CNN for image classification
package main

import (
	"fmt"
	"log"
	"math/rand"

	flow "flow/src"
)

func main() {
	// Generate synthetic image-like data
	// In real usage, load actual image data
	rng := rand.New(rand.NewSource(42))

	numSamples := 200
	imageH := 8
	imageW := 8
	channels := 1
	numClasses := 4

	// Flatten input for API (batch, H*W*C)
	inputDim := imageH * imageW * channels

	inputs := make([][]float64, numSamples)
	targets := make([][]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		inputs[i] = make([]float64, inputDim)
		targets[i] = make([]float64, numClasses)

		// Random class
		class := rng.Intn(numClasses)
		targets[i][class] = 1.0

		// Generate synthetic image with class-specific pattern
		for j := 0; j < inputDim; j++ {
			inputs[i][j] = rng.Float64()*0.5 + float64(class)*0.1
		}
	}

	fmt.Println("Building CNN with Conv2D, MaxPool2D, and Dense layers...")

	// Build CNN - note: input shape is [H, W, C] for conv layers
	net, err := flow.NewNetwork(flow.NetworkConfig{
		Seed:    123,
		Verbose: true,
	}).
		// First conv block
		AddLayer(flow.Conv2D(16, [2]int{3, 3}).
			WithStride(1, 1).
			WithPadding("same").
			WithActivation(flow.ReLU()).
			WithInitializer(flow.HeNormal(1.0)).
			WithBiasInitializer(flow.Zeros()).
			WithBias(true).
			Build()).
		AddLayer(flow.MaxPool2D([2]int{2, 2}).
			WithStride(2, 2).
			WithPadding("valid").
			Build()).
		// Second conv block
		AddLayer(flow.Conv2D(32, [2]int{3, 3}).
			WithStride(1, 1).
			WithPadding("same").
			WithActivation(flow.ReLU()).
			WithInitializer(flow.HeNormal(1.0)).
			WithBiasInitializer(flow.Zeros()).
			WithBias(true).
			Build()).
		AddLayer(flow.MaxPool2D([2]int{2, 2}).
			WithStride(2, 2).
			WithPadding("valid").
			Build()).
		// Flatten and dense layers
		AddLayer(flow.Flatten().Build()).
		AddLayer(flow.Dense(64).
			WithActivation(flow.ReLU()).
			WithInitializer(flow.HeNormal(1.0)).
			WithBiasInitializer(flow.Zeros()).
			WithBias(true).
			Build()).
		AddLayer(flow.Dropout(0.3).
			WithSeed(42).
			Build()).
		AddLayer(flow.Dense(numClasses).
			WithActivation(flow.Softmax()).
			WithInitializer(flow.XavierNormal(1.0)).
			WithBiasInitializer(flow.Zeros()).
			WithBias(true).
			Build()).
		Build([]int{imageH, imageW, channels})

	if err != nil {
		log.Fatalf("Failed to build network: %v", err)
	}

	// Compile with Lion optimizer - new cutting-edge optimizer!
	err = net.Compile(flow.CompileConfig{
		Optimizer: flow.Lion(flow.LionConfig{
			LR:          0.0001,
			Beta1:       0.9,
			Beta2:       0.99,
			WeightDecay: 0.01,
		}),
		Loss: flow.CrossEntropy(flow.CrossEntropyConfig{
			LabelSmoothing: 0.1,
		}),
		Metrics:     []flow.Metric{flow.Accuracy()},
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

	// Train
	result, err := net.Train(inputs, targets, flow.TrainConfig{
		Epochs:                    50,
		BatchSize:                 16,
		Shuffle:                   true,
		ValidationSplit:           0.2,
		Verbose:                   1,
		GradientAccumulationSteps: 0,
	}, []flow.Callback{
		flow.PrintProgress(flow.PrintProgressConfig{PrintEvery: 10}),
	})

	if err != nil {
		log.Fatalf("Training failed: %v", err)
	}

	fmt.Printf("\nTraining Complete!\n")
	fmt.Printf("Final Loss: %.4f\n", result.FinalLoss)
	fmt.Printf("Final Accuracy: %.2f%%\n", result.FinalMetrics["accuracy"]*100)
}
