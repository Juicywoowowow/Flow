// MobileNet-style Example - Demonstrates new Flow features
//
// Features demonstrated:
// - DepthwiseConv2D layers (MobileNet-style efficient convolutions)
// - SpatialDropout2D for better regularization in CNNs
// - Layer freezing for transfer learning
package main

import (
	"fmt"
	"log"
	"math/rand"

	flow "flow/src"
)

func main() {
	// Generate synthetic image-like data
	rng := rand.New(rand.NewSource(42))

	numSamples := 100
	imageH := 16
	imageW := 16
	channels := 3
	numClasses := 4

	inputDim := imageH * imageW * channels

	inputs := make([][]float64, numSamples)
	targets := make([][]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		inputs[i] = make([]float64, inputDim)
		targets[i] = make([]float64, numClasses)

		class := rng.Intn(numClasses)
		targets[i][class] = 1.0

		for j := 0; j < inputDim; j++ {
			inputs[i][j] = rng.Float64()*0.5 + float64(class)*0.1
		}
	}

	fmt.Println("==============================================")
	fmt.Println("MobileNet-style Network with New Flow Features")
	fmt.Println("==============================================")
	fmt.Println()

	// Build a MobileNet-inspired network using:
	// - DepthwiseConv2D (efficient separable convolutions)
	// - SpatialDropout2D (drops entire channels)
	net, err := flow.NewNetwork(flow.NetworkConfig{
		Seed:    123,
		Verbose: true,
	}).
		// First depthwise separable block
		// Depthwise: spatial filtering per channel
		AddLayer(flow.DepthwiseConv2D([2]int{3, 3}).
			WithStride(1, 1).
			WithPadding("same").
			WithDepthMultiplier(1).
			WithActivation(flow.ReLU()).
			WithInitializer(flow.HeNormal(1.0)).
			WithBias(true).
			WithBiasInitializer(flow.Zeros()).
			Build()).
		// Pointwise: 1x1 conv to mix channels
		AddLayer(flow.Conv2D(16, [2]int{1, 1}).
			WithStride(1, 1).
			WithPadding("valid").
			WithActivation(flow.ReLU()).
			WithInitializer(flow.HeNormal(1.0)).
			WithBias(true).
			WithBiasInitializer(flow.Zeros()).
			Build()).
		// Spatial dropout - drops entire feature maps
		AddLayer(flow.SpatialDropout2D(0.1).
			WithSeed(42).
			Build()).
		// Second depthwise separable block
		AddLayer(flow.DepthwiseConv2D([2]int{3, 3}).
			WithStride(2, 2). // Downsample
			WithPadding("same").
			WithDepthMultiplier(1).
			WithActivation(flow.ReLU()).
			WithInitializer(flow.HeNormal(1.0)).
			WithBias(true).
			WithBiasInitializer(flow.Zeros()).
			Build()).
		AddLayer(flow.Conv2D(32, [2]int{1, 1}).
			WithStride(1, 1).
			WithPadding("valid").
			WithActivation(flow.ReLU()).
			WithInitializer(flow.HeNormal(1.0)).
			WithBias(true).
			WithBiasInitializer(flow.Zeros()).
			Build()).
		AddLayer(flow.SpatialDropout2D(0.2).
			WithSeed(43).
			Build()).
		// Flatten and dense layers
		AddLayer(flow.Flatten().Build()).
		AddLayer(flow.Dense(64).
			WithActivation(flow.ReLU()).
			WithInitializer(flow.HeNormal(1.0)).
			WithBias(true).
			WithBiasInitializer(flow.Zeros()).
			Build()).
		AddLayer(flow.Dropout(0.3).
			WithSeed(44).
			Build()).
		AddLayer(flow.Dense(numClasses).
			WithActivation(flow.Softmax()).
			WithInitializer(flow.XavierNormal(1.0)).
			WithBias(true).
			WithBiasInitializer(flow.Zeros()).
			Build()).
		Build([]int{imageH, imageW, channels})

	if err != nil {
		log.Fatalf("Failed to build network: %v", err)
	}

	// Compile the network
	err = net.Compile(flow.CompileConfig{
		Optimizer: flow.Adam(flow.AdamConfig{
			LR:          0.001,
			Beta1:       0.9,
			Beta2:       0.999,
			Epsilon:     1e-8,
			WeightDecay: 0.0,
			AMSGrad:     false,
		}),
		Loss: flow.CrossEntropy(flow.CrossEntropyConfig{
			LabelSmoothing: 0.0,
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

	fmt.Println("Network Summary:")
	fmt.Println(net.Summary())

	// =========================================================================
	// LAYER FREEZING DEMONSTRATION
	// =========================================================================
	fmt.Println("==============================================")
	fmt.Println("Layer Freezing Demonstration")
	fmt.Println("==============================================")

	// Show initial freeze status
	fmt.Println("\n--- Initial State (all trainable) ---")
	fmt.Printf("Total parameters:     %d\n", net.TotalParameters())
	fmt.Printf("Trainable parameters: %d\n", net.TrainableParameters())

	// Freeze the first 3 layers (first depthwise separable block)
	fmt.Println("\n--- After FreezeTo(3) - Freezing first block ---")
	if err := net.FreezeTo(3); err != nil {
		log.Fatalf("Failed to freeze: %v", err)
	}
	fmt.Printf("Total parameters:     %d\n", net.TotalParameters())
	fmt.Printf("Trainable parameters: %d\n", net.TrainableParameters())
	fmt.Printf("Frozen layers: %v\n", net.FrozenLayers())

	// Show detailed freeze summary
	fmt.Println("\nDetailed Freeze Summary:")
	fmt.Println(net.FreezeSummary())

	// Unfreeze all for training
	net.UnfreezeAll()
	fmt.Println("--- After UnfreezeAll() ---")
	fmt.Printf("Trainable parameters: %d\n", net.TrainableParameters())

	// Freeze all except last 2 layers (for fine-tuning scenario)
	fmt.Println("\n--- Simulating Transfer Learning: Freeze all but classifier ---")
	numLayers := 10 // We have 10 layers total
	if err := net.FreezeExcept(numLayers-2, numLayers-1); err != nil {
		log.Fatalf("Failed to freeze: %v", err)
	}
	fmt.Println(net.FreezeSummary())

	// Unfreeze for full training demo
	net.UnfreezeAll()

	// =========================================================================
	// TRAINING
	// =========================================================================
	fmt.Println("==============================================")
	fmt.Println("Training with all layers trainable")
	fmt.Println("==============================================")

	result, err := net.Train(inputs, targets, flow.TrainConfig{
		Epochs:                    30,
		BatchSize:                 16,
		Shuffle:                   true,
		ValidationSplit:           0.2,
		Verbose:                   1,
		GradientAccumulationSteps: 0,
	}, []flow.Callback{
		flow.PrintProgress(flow.PrintProgressConfig{PrintEvery: 5}),
	})

	if err != nil {
		log.Fatalf("Training failed: %v", err)
	}

	fmt.Printf("\nTraining Complete!\n")
	fmt.Printf("Final Loss: %.4f\n", result.FinalLoss)
	fmt.Printf("Final Accuracy: %.2f%%\n", result.FinalMetrics["accuracy"]*100)

	// =========================================================================
	// SIMULATED FINE-TUNING SCENARIO
	// =========================================================================
	fmt.Println("\n==============================================")
	fmt.Println("Fine-tuning: Only training classifier layers")
	fmt.Println("==============================================")

	// Freeze feature extraction layers, only train classifier
	if err := net.FreezeTo(numLayers - 2); err != nil {
		log.Fatalf("Failed to freeze: %v", err)
	}
	fmt.Println(net.FreezeSummary())

	// Train with frozen layers (faster, fewer parameters to update)
	result2, err := net.Train(inputs, targets, flow.TrainConfig{
		Epochs:          10,
		BatchSize:       16,
		Shuffle:         true,
		ValidationSplit: 0.2,
		Verbose:         1,
	}, []flow.Callback{
		flow.PrintProgress(flow.PrintProgressConfig{PrintEvery: 2}),
	})

	if err != nil {
		log.Fatalf("Fine-tuning failed: %v", err)
	}

	fmt.Printf("\nFine-tuning Complete!\n")
	fmt.Printf("Final Loss: %.4f\n", result2.FinalLoss)
	fmt.Printf("Final Accuracy: %.2f%%\n", result2.FinalMetrics["accuracy"]*100)
}
