// TensorMod Example - Custom Tensor Transformations
//
// This example demonstrates: the TensorMod layer for custom transformations,
// TensorModInspect for debugging, utility functions, and the new error system.
package main

import (
	"fmt"
	"log"

	flow "flow/src"
)

func main() {
	fmt.Println("Flow TensorMod Demo")
	fmt.Println("===================")
	fmt.Println()

	// =========================================================================
	// Example 1: Simple TensorMod with Scale utility
	// =========================================================================
	fmt.Println("Test 1: Scale utility function")

	net1, err := flow.NewNetwork(flow.NetworkConfig{
		Seed:    42,
		Verbose: true,
	}).
		AddLayer(flow.Dense(4).
			WithActivation(flow.ReLU()).
			WithInitializer(flow.HeNormal(1.0)).
			Build()).
		AddLayer(flow.TensorMod(
			flow.Scale(2.0),     // Double all values
			flow.ScaleGrad(2.0), // Gradient is also 2
		).WithName("scale_2x").Build()).
		AddLayer(flow.Dense(1).
			WithActivation(flow.Sigmoid()).
			WithInitializer(flow.XavierNormal(1.0)).
			Build()).
		Build([]int{2})

	if err != nil {
		log.Fatalf("Build failed: %v", err)
	}

	fmt.Println(net1.Summary())

	// =========================================================================
	// Example 2: TensorMod with custom function
	// =========================================================================
	fmt.Println("\nTest 2: Custom clipping function")

	net2, err := flow.NewNetwork(flow.NetworkConfig{
		Seed:    42,
		Verbose: true,
	}).
		AddLayer(flow.Dense(4).
			WithActivation(flow.Linear()).
			WithInitializer(flow.HeNormal(1.0)).
			Build()).
		AddLayer(flow.TensorMod(
			// Forward: clip values to [-1, 1]
			func(data []float64, shape []int) []float64 {
				result := make([]float64, len(data))
				for i, v := range data {
					if v < -1 {
						result[i] = -1
					} else if v > 1 {
						result[i] = 1
					} else {
						result[i] = v
					}
				}
				return result
			},
			// Backward: gradient is 1 where not clipped (simplified, passes through)
			func(gradOutput []float64, shape []int) []float64 {
				result := make([]float64, len(gradOutput))
				copy(result, gradOutput)
				return result
			},
		).WithName("clip_to_1").WithValidation(flow.ValidationStrict).Build()).
		AddLayer(flow.Dense(1).
			WithActivation(flow.Sigmoid()).
			WithInitializer(flow.XavierNormal(1.0)).
			Build()).
		Build([]int{2})

	if err != nil {
		log.Fatalf("Build failed: %v", err)
	}

	fmt.Println(net2.Summary())

	// =========================================================================
	// Example 3: TensorModInspect for debugging
	// =========================================================================
	fmt.Println("\nTest 3: TensorModInspect for debugging")

	net3, err := flow.NewNetwork(flow.NetworkConfig{
		Seed:    42,
		Verbose: true,
	}).
		AddLayer(flow.Dense(4).
			WithActivation(flow.ReLU()).
			WithInitializer(flow.HeNormal(1.0)).
			Build()).
		AddLayer(flow.TensorModInspect(func(data []float64, shape []int) {
			// This will only run during inference (training=false)
			min, max := data[0], data[0]
			for _, v := range data {
				if v < min {
					min = v
				}
				if v > max {
					max = v
				}
			}
			fmt.Printf("  [Inspect] shape=%v len=%d range=[%.4f, %.4f]\n", shape, len(data), min, max)
		}).WithName("debug_relu_output").Build()).
		AddLayer(flow.Dense(1).
			WithActivation(flow.Sigmoid()).
			WithInitializer(flow.XavierNormal(1.0)).
			Build()).
		Build([]int{2})

	if err != nil {
		log.Fatalf("Build failed: %v", err)
	}

	fmt.Println(net3.Summary())

	// Compile and run inference to trigger inspect
	err = net3.Compile(flow.CompileConfig{
		Optimizer:   flow.Adam(flow.AdamConfig{LR: 0.01}),
		Loss:        flow.MSE(flow.MSEConfig{Reduction: "mean"}),
		Regularizer: flow.NoReg(),
		GradientClip: flow.GradientClipConfig{
			Mode: "none",
		},
	})
	if err != nil {
		log.Fatalf("Compile failed: %v", err)
	}

	fmt.Println("\nRunning inference (inspect should trigger):")
	testInput := [][]float64{{0.5, 0.5}, {-0.5, 0.5}}
	output, err := net3.Predict(testInput)
	if err != nil {
		log.Fatalf("Predict failed: %v", err)
	}
	fmt.Printf("Output: %v\n", output)

	// =========================================================================
	// Example 4: Error handling demo
	// =========================================================================
	fmt.Println("\nTest 4: Error format demonstration")

	// Create a FlowError to show the format
	sampleError := &flow.FlowError{
		Component:  "TensorMod",
		ErrorType:  "shape mismatch",
		LayerIndex: 2,
		LayerName:  "custom_transform",
		Phase:      "forward",
		InputInfo: &flow.TensorInfo{
			Shape:   []int{32, 128},
			Size:    4096,
			Address: "0xc0001a2000",
		},
		OutputInfo: &flow.TensorInfo{
			Shape:    []int{32, 64},
			Size:     2048,
			Address:  "0xc0001b4000",
			NaNCount: 3,
			InfCount: 0,
		},
		ExpectedInfo: "[32, 128] size=4096",
		Cause:        "output shape changed without .WithOutputShape() declaration",
	}

	fmt.Println("Sample error message format:")
	fmt.Println("---")
	fmt.Println(sampleError.Error())
	fmt.Println("---")

	fmt.Println("\nAll tests passed!")
}
