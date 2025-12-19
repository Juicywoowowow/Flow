// Package flow is a neural network library for Go.
//
// Flow provides a power-user focused API with explicit configuration
// and no hidden defaults. Every hyperparameter must be specified.
//
// Basic usage:
//
//	net, err := flow.NewNetwork(flow.NetworkConfig{Seed: 42}).
//		AddLayer(flow.Dense(128).
//			WithActivation(flow.ReLU()).
//			WithInitializer(flow.HeNormal(1.0)).
//			WithBiasInitializer(flow.Zeros()).
//			WithBias(true).
//			Build()).
//		AddLayer(flow.Dense(10).
//			WithActivation(flow.Softmax()).
//			WithInitializer(flow.XavierNormal(1.0)).
//			WithBiasInitializer(flow.Zeros()).
//			WithBias(true).
//			Build()).
//		Build([]int{784})
//
//	err = net.Compile(flow.CompileConfig{
//		Optimizer: flow.Adam(flow.AdamConfig{
//			LR:          0.001,
//			Beta1:       0.9,
//			Beta2:       0.999,
//			Epsilon:     1e-8,
//			WeightDecay: 0.0,
//			AMSGrad:     false,
//		}),
//		Loss: flow.CrossEntropy(flow.CrossEntropyConfig{
//			LabelSmoothing: 0.0,
//		}),
//		Metrics:     []flow.Metric{flow.Accuracy()},
//		Regularizer: flow.NoReg(),
//		GradientClip: flow.GradientClipConfig{
//			Mode:     "none",
//			MaxNorm:  0.0,
//			MaxValue: 0.0,
//		},
//	})
//
//	result, err := net.Train(inputs, targets, flow.TrainConfig{
//		Epochs:          100,
//		BatchSize:       32,
//		Shuffle:         true,
//		ValidationSplit: 0.2,
//		Verbose:         1,
//	}, []flow.Callback{
//		flow.PrintProgress(flow.PrintProgressConfig{PrintEvery: 10}),
//	})
package flow

// Version of the Flow library
const Version = "1.0.0"

// DebugMode enables verbose logging
var DebugMode = false

// SetDebug enables or disables debug mode
func SetDebug(enabled bool) {
	DebugMode = enabled
}
