// Sequence Example - Recurrent Neural Networks with Flow
//
// Demonstrates:
// - LSTM layer
// - GRU layer
// - Sequence classification
package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"

	flow "flow/src"
)

func main() {
	fmt.Println("=== Flow Sequence Classification Demo ===\n")

	rng := rand.New(rand.NewSource(42))

	// Generate synthetic sequence data
	// Task: Classify sequences based on their pattern
	numSamples := 200
	seqLen := 10
	features := 4
	numClasses := 3

	// Input shape for API: [samples][seqLen * features] (flattened)
	inputs := make([][]float64, numSamples)
	targets := make([][]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		inputs[i] = make([]float64, seqLen*features)
		targets[i] = make([]float64, numClasses)

		class := rng.Intn(numClasses)
		targets[i][class] = 1.0

		// Generate class-specific sequence pattern
		for t := 0; t < seqLen; t++ {
			for f := 0; f < features; f++ {
				baseVal := rng.Float64() * 0.3
				switch class {
				case 0: // Rising pattern
					inputs[i][t*features+f] = baseVal + float64(t)*0.1
				case 1: // Falling pattern
					inputs[i][t*features+f] = baseVal + float64(seqLen-t)*0.1
				case 2: // Oscillating pattern
					inputs[i][t*features+f] = baseVal + 0.5*math.Sin(float64(t)*0.5)
				}
			}
		}
	}

	fmt.Printf("Dataset: %d samples, sequence length %d, %d features\n", numSamples, seqLen, features)
	fmt.Printf("Task: Classify sequences into %d patterns (rising, falling, oscillating)\n\n", numClasses)

	// =====================
	// Test 1: LSTM Network
	// =====================
	fmt.Println("--- Building LSTM Network ---")

	lstmNet, err := flow.NewNetwork(flow.NetworkConfig{
		Seed:    123,
		Verbose: true,
	}).
		AddLayer(flow.LSTM(32).
			WithReturnSequences(false). // Only return last hidden state
			WithInitializer(flow.XavierNormal(1.0)).
			WithRecurrentInitializer(flow.XavierNormal(1.0)).
			WithBiasInitializer(flow.Zeros()).
			Build()).
		AddLayer(flow.Dense(16).
			WithActivation(flow.ReLU()).
			WithInitializer(flow.HeNormal(1.0)).
			WithBiasInitializer(flow.Zeros()).
			WithBias(true).
			Build()).
		AddLayer(flow.Dense(numClasses).
			WithActivation(flow.Softmax()).
			WithInitializer(flow.XavierNormal(1.0)).
			WithBiasInitializer(flow.Zeros()).
			WithBias(true).
			Build()).
		Build([]int{seqLen, features})

	if err != nil {
		log.Fatalf("Failed to build LSTM network: %v", err)
	}

	err = lstmNet.Compile(flow.CompileConfig{
		Optimizer: flow.Adam(flow.AdamConfig{
			LR:          0.01,
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
			Mode:    "norm",
			MaxNorm: 5.0,
		},
	})

	if err != nil {
		log.Fatalf("Failed to compile LSTM: %v", err)
	}

	fmt.Println(lstmNet.Summary())

	// Train LSTM
	fmt.Println("Training LSTM...")
	lstmResult, err := lstmNet.Train(inputs, targets, flow.TrainConfig{
		Epochs:          30,
		BatchSize:       16,
		Shuffle:         true,
		ValidationSplit: 0.2,
		Verbose:         1,
	}, []flow.Callback{
		flow.PrintProgress(flow.PrintProgressConfig{PrintEvery: 10}),
	})

	if err != nil {
		log.Fatalf("LSTM training failed: %v", err)
	}

	fmt.Printf("\nLSTM Results:\n")
	fmt.Printf("  Final Loss: %.4f\n", lstmResult.FinalLoss)
	fmt.Printf("  Final Accuracy: %.2f%%\n\n", lstmResult.FinalMetrics["accuracy"]*100)

	// =====================
	// Test 2: GRU Network
	// =====================
	fmt.Println("--- Building GRU Network ---")

	gruNet, err := flow.NewNetwork(flow.NetworkConfig{
		Seed:    456,
		Verbose: true,
	}).
		AddLayer(flow.GRU(32).
			WithReturnSequences(false).
			WithInitializer(flow.XavierNormal(1.0)).
			WithRecurrentInitializer(flow.XavierNormal(1.0)).
			WithBiasInitializer(flow.Zeros()).
			Build()).
		AddLayer(flow.Dense(16).
			WithActivation(flow.ReLU()).
			WithInitializer(flow.HeNormal(1.0)).
			WithBiasInitializer(flow.Zeros()).
			WithBias(true).
			Build()).
		AddLayer(flow.Dense(numClasses).
			WithActivation(flow.Softmax()).
			WithInitializer(flow.XavierNormal(1.0)).
			WithBiasInitializer(flow.Zeros()).
			WithBias(true).
			Build()).
		Build([]int{seqLen, features})

	if err != nil {
		log.Fatalf("Failed to build GRU network: %v", err)
	}

	err = gruNet.Compile(flow.CompileConfig{
		Optimizer: flow.Adam(flow.AdamConfig{
			LR:          0.01,
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
			Mode:    "norm",
			MaxNorm: 5.0,
		},
	})

	if err != nil {
		log.Fatalf("Failed to compile GRU: %v", err)
	}

	fmt.Println(gruNet.Summary())

	// Train GRU
	fmt.Println("Training GRU...")
	gruResult, err := gruNet.Train(inputs, targets, flow.TrainConfig{
		Epochs:          30,
		BatchSize:       16,
		Shuffle:         true,
		ValidationSplit: 0.2,
		Verbose:         1,
	}, []flow.Callback{
		flow.PrintProgress(flow.PrintProgressConfig{PrintEvery: 10}),
	})

	if err != nil {
		log.Fatalf("GRU training failed: %v", err)
	}

	fmt.Printf("\nGRU Results:\n")
	fmt.Printf("  Final Loss: %.4f\n", gruResult.FinalLoss)
	fmt.Printf("  Final Accuracy: %.2f%%\n\n", gruResult.FinalMetrics["accuracy"]*100)

	// Summary comparison
	fmt.Println("=== Summary ===")
	fmt.Printf("LSTM: %.2f%% accuracy\n", lstmResult.FinalMetrics["accuracy"]*100)
	fmt.Printf("GRU:  %.2f%% accuracy\n", gruResult.FinalMetrics["accuracy"]*100)
	fmt.Println("\nBoth models learned to classify sequence patterns!")
}
