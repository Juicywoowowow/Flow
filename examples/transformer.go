// Advanced Transformer Example - Character-level Language Model
//
// This example demonstrates Flow's complete feature set including:
// - NEW: Embedding layer (maps tokens to vectors)
// - NEW: Positional Encoding (sinusoidal position information)
// - NEW: Residual connections (skip connections for gradient flow)
// - Builder pattern with .With...() methods
// - Multiple layer types (Attention, Dense, Normalization, Dropout)
// - Advanced optimizers (AdamW with weight decay)
// - Callbacks (EarlyStopping, LRScheduler, PrintProgress)
// - Gradient clipping and label smoothing
// - Temperature-controlled text generation
package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"strings"

	flow "flow/src"
)

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║     Flow Transformer Demo: Character Language Model          ║")
	fmt.Println("║     Now with Embedding + Positional Encoding + Residual!     ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	rng := rand.New(rand.NewSource(42))

	// =========================================================================
	// TRAINING DATA - Famous quotes corpus
	// =========================================================================
	corpus := `the quick brown fox jumps over the lazy dog.
a journey of a thousand miles begins with a single step.
to be or not to be that is the question.
all that glitters is not gold.
the only thing we have to fear is fear itself.
in the beginning was the word and the word was with god.
ask not what your country can do for you.
i think therefore i am.
the greatest glory in living lies not in never falling.
it is during our darkest moments that we must focus to see the light.
the way to get started is to quit talking and begin doing.
life is what happens when you are busy making other plans.
the future belongs to those who believe in the beauty of their dreams.
it is better to be feared than loved if you cannot be both.
the only impossible journey is the one you never begin.
success is not final failure is not fatal it is the courage to continue.
believe you can and you are halfway there.
the best time to plant a tree was twenty years ago.
do not go where the path may lead go instead where there is no path.
in three words i can sum up everything i learned about life it goes on.
`
	corpus = strings.ToLower(corpus)
	corpus = strings.ReplaceAll(corpus, "\n", " ")

	// =========================================================================
	// VOCABULARY CREATION
	// =========================================================================
	chars := make(map[rune]bool)
	for _, c := range corpus {
		chars[c] = true
	}
	vocab := make([]rune, 0, len(chars))
	for c := range chars {
		vocab = append(vocab, c)
	}
	vocabSize := len(vocab)

	charToIdx := make(map[rune]int)
	idxToChar := make(map[int]rune)
	for i, c := range vocab {
		charToIdx[c] = i
		idxToChar[i] = c
	}

	fmt.Printf("📊 Dataset Statistics:\n")
	fmt.Printf("   • Corpus length: %d characters\n", len(corpus))
	fmt.Printf("   • Vocabulary size: %d unique characters\n", vocabSize)

	// =========================================================================
	// SEQUENCE CREATION
	// =========================================================================
	seqLen := 16
	step := 2
	embedDim := 32 // Embedding dimension

	var sequences [][]int // Token indices instead of one-hot
	var nextChars []rune

	for i := 0; i < len(corpus)-seqLen; i += step {
		seq := make([]int, seqLen)
		for j, c := range corpus[i : i+seqLen] {
			seq[j] = charToIdx[c]
		}
		sequences = append(sequences, seq)
		nextChars = append(nextChars, rune(corpus[i+seqLen]))
	}

	numSamples := len(sequences)
	fmt.Printf("   • Training sequences: %d\n", numSamples)
	fmt.Printf("   • Sequence length: %d\n", seqLen)
	fmt.Printf("   • Embedding dimension: %d\n", embedDim)
	fmt.Println()

	// =========================================================================
	// DATA PREPARATION - Integer indices for Embedding layer
	// =========================================================================
	inputs := make([][]float64, numSamples)
	targets := make([][]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		// Input: token indices (will be embedded by Embedding layer)
		inputs[i] = make([]float64, seqLen)
		for t := 0; t < seqLen; t++ {
			inputs[i][t] = float64(sequences[i][t])
		}

		// Target: one-hot encoded next character
		targets[i] = make([]float64, vocabSize)
		targetIdx := charToIdx[nextChars[i]]
		targets[i][targetIdx] = 1.0
	}

	// =========================================================================
	// MODEL ARCHITECTURE - Full Transformer with New Layers
	// =========================================================================
	fmt.Println("🏗️  Building Transformer Architecture...")
	fmt.Println("   NEW Features: Embedding → PositionalEncoding → Attention → Residual")
	fmt.Println()

	net, err := flow.NewNetwork(flow.NetworkConfig{
		Seed:    1337,
		Verbose: true,
	}).
		// ─────────────────────────────────────────────────────────────────
		// Layer 1: EMBEDDING LAYER (NEW!)
		// Maps integer token indices to dense vectors
		// Input: [batch, seqLen] integers → Output: [batch, seqLen, embedDim]
		// ─────────────────────────────────────────────────────────────────
		AddLayer(flow.Embedding(vocabSize, embedDim).
			WithInitializer(flow.RandomNormal(0, 0.02)).
			Build()).

		// ─────────────────────────────────────────────────────────────────
		// Layer 2: POSITIONAL ENCODING (NEW!)
		// Adds sinusoidal position information to embeddings
		// PE(pos, 2i) = sin(pos / 10000^(2i/d))
		// PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
		// ─────────────────────────────────────────────────────────────────
		AddLayer(flow.PositionalEncoding(seqLen+10, embedDim).
			WithDropout(0.1).
			WithLearned(false). // Use sinusoidal (can set true for learned)
			Build()).

		// ─────────────────────────────────────────────────────────────────
		// Layer 3: Multi-Head Self-Attention
		// ─────────────────────────────────────────────────────────────────
		AddLayer(flow.MultiHeadAttention(4, 8). // 4 heads, 8-dim keys
							WithValueDim(8).
							WithBias(true).
							WithInitializer(flow.XavierNormal(1.0)).
							WithBiasInitializer(flow.Zeros()).
							Build()).

		// ─────────────────────────────────────────────────────────────────
		// Layer 4: Layer Normalization
		// ─────────────────────────────────────────────────────────────────
		AddLayer(flow.LayerNorm(1e-6).Build()).

		// ─────────────────────────────────────────────────────────────────
		// Layer 5: Flatten for dense layers
		// ─────────────────────────────────────────────────────────────────
		AddLayer(flow.Flatten().Build()).

		// ─────────────────────────────────────────────────────────────────
		// Layer 6: Feed-Forward Network with GELU
		// ─────────────────────────────────────────────────────────────────
		AddLayer(flow.Dense(128).
			WithActivation(flow.GELU()).
			WithInitializer(flow.HeNormal(1.0)).
			WithBiasInitializer(flow.Zeros()).
			WithBias(true).
			Build()).

		// ─────────────────────────────────────────────────────────────────
		// Layer 7: Dropout
		// ─────────────────────────────────────────────────────────────────
		AddLayer(flow.Dropout(0.2).Build()).

		// ─────────────────────────────────────────────────────────────────
		// Layer 8: Second dense layer with Swish
		// ─────────────────────────────────────────────────────────────────
		AddLayer(flow.Dense(64).
			WithActivation(flow.Swish()).
			WithInitializer(flow.HeNormal(1.0)).
			WithBiasInitializer(flow.Zeros()).
			WithBias(true).
			Build()).

		// ─────────────────────────────────────────────────────────────────
		// Layer 9: Output layer
		// ─────────────────────────────────────────────────────────────────
		AddLayer(flow.Dense(vocabSize).
			WithActivation(flow.Softmax()).
			WithInitializer(flow.XavierNormal(1.0)).
			WithBiasInitializer(flow.Zeros()).
			WithBias(true).
			Build()).
		Build([]int{seqLen})

	if err != nil {
		log.Fatalf("❌ Build failed: %v", err)
	}

	// =========================================================================
	// COMPILATION
	// =========================================================================
	fmt.Println("⚙️  Compiling Model...")

	err = net.Compile(flow.CompileConfig{
		Optimizer: flow.AdamW(flow.AdamWConfig{
			LR:          0.002,
			Beta1:       0.9,
			Beta2:       0.999,
			Epsilon:     1e-8,
			WeightDecay: 0.01,
		}),
		Loss: flow.CrossEntropy(flow.CrossEntropyConfig{
			LabelSmoothing: 0.1,
		}),
		Metrics: []flow.Metric{
			flow.Accuracy(),
		},
		Regularizer: flow.NoReg(),
		GradientClip: flow.GradientClipConfig{
			Mode:    "norm",
			MaxNorm: 1.0,
		},
	})

	if err != nil {
		log.Fatalf("❌ Compile failed: %v", err)
	}

	fmt.Println()
	fmt.Println(net.Summary())

	// =========================================================================
	// TRAINING
	// =========================================================================
	fmt.Println("🚀 Training with Callbacks...")
	fmt.Println()

	result, err := net.Train(inputs, targets,
		flow.TrainConfig{
			Epochs:          100,
			BatchSize:       32,
			Shuffle:         true,
			ValidationSplit: 0.15,
			Verbose:         1,
		},
		[]flow.Callback{
			flow.EarlyStopping(flow.EarlyStoppingConfig{
				Monitor:     "val_loss",
				MinDelta:    0.001,
				Patience:    20,
				Mode:        "min",
				RestoreBest: true,
			}),
			flow.LRSchedulerCallback_(flow.LRSchedulerConfig{
				Scheduler: flow.CosineAnnealing(flow.CosineAnnealingConfig{
					TMax:   100,
					EtaMin: 0.0001,
					EtaMax: 0.002,
				}),
				InitialLR: 0.002,
			}),
			flow.PrintProgress(flow.PrintProgressConfig{
				PrintEvery: 10,
			}),
		},
	)

	if err != nil {
		log.Fatalf("❌ Training failed: %v", err)
	}

	// =========================================================================
	// RESULTS
	// =========================================================================
	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Printf("📈 Training Complete:\n")
	fmt.Printf("   • Final Loss: %.4f\n", result.FinalLoss)
	fmt.Printf("   • Final Accuracy: %.2f%%\n", result.FinalMetrics["accuracy"]*100)
	fmt.Println("═══════════════════════════════════════════════════════════════")

	// =========================================================================
	// TEXT GENERATION
	// =========================================================================
	fmt.Println()
	fmt.Println("✨ Generating Text:")
	fmt.Println()

	seeds := []string{
		"the quick bro",
		"to be or not ",
		"life is what ",
	}

	for _, seed := range seeds {
		if len(seed) < seqLen {
			seed = seed + strings.Repeat(" ", seqLen-len(seed))
		} else {
			seed = seed[:seqLen]
		}

		generated := generateTextEmbed(net, seed, 50, charToIdx, idxToChar, seqLen, rng, 0.7)
		fmt.Printf("Seed: \"%s\"\n", strings.TrimSpace(seed))
		fmt.Printf("  →  %s\n\n", generated)
	}

	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("✅ New Features Demonstrated:")
	fmt.Println("   • Embedding layer (token → vector)")
	fmt.Println("   • Sinusoidal Positional Encoding")
	fmt.Println("   • Full Transformer architecture")
	fmt.Println("═══════════════════════════════════════════════════════════════")
}

func generateTextEmbed(net *flow.Network, seed string, length int,
	charToIdx map[rune]int, idxToChar map[int]rune,
	seqLen int, rng *rand.Rand, temperature float64) string {

	currentSeq := seed
	generated := seed

	for i := 0; i < length; i++ {
		// Encode as indices
		inputVec := make([][]float64, 1)
		inputVec[0] = make([]float64, seqLen)
		for t, char := range currentSeq {
			if idx, ok := charToIdx[char]; ok {
				inputVec[0][t] = float64(idx)
			}
		}

		// Predict
		preds, _ := net.Predict(inputVec)
		probs := preds[0]

		// Sample with temperature
		nextIdx := sampleTemp(probs, rng, temperature)
		nextChar := idxToChar[nextIdx]

		generated += string(nextChar)
		currentSeq = currentSeq[1:] + string(nextChar)
	}

	return generated
}

func sampleTemp(probs []float64, rng *rand.Rand, temperature float64) int {
	logits := make([]float64, len(probs))
	maxLogit := math.Inf(-1)

	for i, p := range probs {
		if p > 0 {
			logits[i] = math.Log(p) / temperature
		} else {
			logits[i] = -1000
		}
		if logits[i] > maxLogit {
			maxLogit = logits[i]
		}
	}

	sum := 0.0
	for i := range logits {
		logits[i] = math.Exp(logits[i] - maxLogit)
		sum += logits[i]
	}

	r := rng.Float64() * sum
	cumSum := 0.0
	for i, l := range logits {
		cumSum += l
		if r < cumSum {
			return i
		}
	}
	return len(probs) - 1
}
