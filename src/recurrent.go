package flow

import (
	"errors"
	"math"
	"math/rand"
)

// GRULayer - Gated Recurrent Unit
// Simpler than LSTM with 2 gates: reset and update
type GRULayer struct {
	units           int
	returnSequences bool
	initializer     Initializer
	recurrentInit   Initializer
	biasInit        Initializer
	dropout         float64
	recurrentDrop   float64

	// Weights: [inputDim, units] for input, [units, units] for recurrent
	Wz, Wr, Wh *tensor // Input weights for update, reset, candidate
	Uz, Ur, Uh *tensor // Recurrent weights
	bz, br, bh *tensor // Biases

	// Gradients
	dWz, dWr, dWh *tensor
	dUz, dUr, dUh *tensor
	dbz, dbr, dbh *tensor

	// Cache for backward pass
	inputs       *tensor
	hiddenStates []*tensor
	zGates       []*tensor
	rGates       []*tensor
	hCandidates  []*tensor

	inputDim int
	seqLen   int
	rng      *rand.Rand
	built    bool
}

type GRUBuilder struct {
	layer *GRULayer
}

func GRU(units int) *GRUBuilder {
	return &GRUBuilder{
		layer: &GRULayer{
			units:           units,
			returnSequences: false,
		},
	}
}

func (b *GRUBuilder) WithReturnSequences(ret bool) *GRUBuilder {
	b.layer.returnSequences = ret
	return b
}

func (b *GRUBuilder) WithInitializer(init Initializer) *GRUBuilder {
	b.layer.initializer = init
	return b
}

func (b *GRUBuilder) WithRecurrentInitializer(init Initializer) *GRUBuilder {
	b.layer.recurrentInit = init
	return b
}

func (b *GRUBuilder) WithBiasInitializer(init Initializer) *GRUBuilder {
	b.layer.biasInit = init
	return b
}

func (b *GRUBuilder) WithDropout(rate float64) *GRUBuilder {
	b.layer.dropout = rate
	return b
}

func (b *GRUBuilder) WithRecurrentDropout(rate float64) *GRUBuilder {
	b.layer.recurrentDrop = rate
	return b
}

func (b *GRUBuilder) Build() Layer {
	return b.layer
}

func (g *GRULayer) build(inputShape []int, rng *rand.Rand) error {
	if len(inputShape) < 2 {
		return errors.New("flow: GRU requires input shape [seqLen, features]")
	}
	if g.initializer == nil {
		return errors.New("flow: GRU requires initializer")
	}
	if g.recurrentInit == nil {
		return errors.New("flow: GRU requires recurrent initializer")
	}
	if g.biasInit == nil {
		return errors.New("flow: GRU requires bias initializer")
	}

	g.seqLen = inputShape[0]
	g.inputDim = inputShape[1]
	g.rng = rng

	// Initialize input weights [inputDim, units]
	g.Wz = newTensor(g.inputDim, g.units)
	g.Wr = newTensor(g.inputDim, g.units)
	g.Wh = newTensor(g.inputDim, g.units)
	g.initializer.initialize(g.Wz, g.inputDim, g.units, rng)
	g.initializer.initialize(g.Wr, g.inputDim, g.units, rng)
	g.initializer.initialize(g.Wh, g.inputDim, g.units, rng)

	// Initialize recurrent weights [units, units]
	g.Uz = newTensor(g.units, g.units)
	g.Ur = newTensor(g.units, g.units)
	g.Uh = newTensor(g.units, g.units)
	g.recurrentInit.initialize(g.Uz, g.units, g.units, rng)
	g.recurrentInit.initialize(g.Ur, g.units, g.units, rng)
	g.recurrentInit.initialize(g.Uh, g.units, g.units, rng)

	// Initialize biases [units]
	g.bz = newTensor(g.units)
	g.br = newTensor(g.units)
	g.bh = newTensor(g.units)
	g.biasInit.initialize(g.bz, g.inputDim, g.units, rng)
	g.biasInit.initialize(g.br, g.inputDim, g.units, rng)
	g.biasInit.initialize(g.bh, g.inputDim, g.units, rng)

	// Initialize gradients
	g.dWz = newTensor(g.inputDim, g.units)
	g.dWr = newTensor(g.inputDim, g.units)
	g.dWh = newTensor(g.inputDim, g.units)
	g.dUz = newTensor(g.units, g.units)
	g.dUr = newTensor(g.units, g.units)
	g.dUh = newTensor(g.units, g.units)
	g.dbz = newTensor(g.units)
	g.dbr = newTensor(g.units)
	g.dbh = newTensor(g.units)

	g.built = true
	return nil
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (g *GRULayer) forward(input *tensor, training bool) (*tensor, error) {
	if !g.built {
		return nil, errors.New("flow: GRU not built")
	}

	batchSize := input.shape[0]
	seqLen := input.shape[1]
	features := input.shape[2]

	g.inputs = input
	g.hiddenStates = make([]*tensor, seqLen+1)
	g.zGates = make([]*tensor, seqLen)
	g.rGates = make([]*tensor, seqLen)
	g.hCandidates = make([]*tensor, seqLen)

	// Initialize h_0 to zeros
	g.hiddenStates[0] = newTensor(batchSize, g.units)

	// Process each time step
	for t := 0; t < seqLen; t++ {
		// Extract x_t [batchSize, features]
		xt := newTensor(batchSize, features)
		for b := 0; b < batchSize; b++ {
			for f := 0; f < features; f++ {
				xt.data[b*features+f] = input.data[b*seqLen*features+t*features+f]
			}
		}

		hPrev := g.hiddenStates[t]

		// Compute gates
		zt := newTensor(batchSize, g.units) // Update gate
		rt := newTensor(batchSize, g.units) // Reset gate
		ht := newTensor(batchSize, g.units) // Candidate
		hNew := newTensor(batchSize, g.units)

		for b := 0; b < batchSize; b++ {
			for u := 0; u < g.units; u++ {
				// z_t = sigmoid(W_z * x_t + U_z * h_{t-1} + b_z)
				zVal := g.bz.data[u]
				for f := 0; f < features; f++ {
					zVal += xt.data[b*features+f] * g.Wz.data[f*g.units+u]
				}
				for h := 0; h < g.units; h++ {
					zVal += hPrev.data[b*g.units+h] * g.Uz.data[h*g.units+u]
				}
				zt.data[b*g.units+u] = sigmoid(zVal)

				// r_t = sigmoid(W_r * x_t + U_r * h_{t-1} + b_r)
				rVal := g.br.data[u]
				for f := 0; f < features; f++ {
					rVal += xt.data[b*features+f] * g.Wr.data[f*g.units+u]
				}
				for h := 0; h < g.units; h++ {
					rVal += hPrev.data[b*g.units+h] * g.Ur.data[h*g.units+u]
				}
				rt.data[b*g.units+u] = sigmoid(rVal)

				// h̃_t = tanh(W_h * x_t + U_h * (r_t ⊙ h_{t-1}) + b_h)
				hVal := g.bh.data[u]
				for f := 0; f < features; f++ {
					hVal += xt.data[b*features+f] * g.Wh.data[f*g.units+u]
				}
				for h := 0; h < g.units; h++ {
					rh := rt.data[b*g.units+h] * hPrev.data[b*g.units+h]
					hVal += rh * g.Uh.data[h*g.units+u]
				}
				ht.data[b*g.units+u] = math.Tanh(hVal)

				// h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
				z := zt.data[b*g.units+u]
				hNew.data[b*g.units+u] = (1-z)*hPrev.data[b*g.units+u] + z*ht.data[b*g.units+u]
			}
		}

		g.zGates[t] = zt
		g.rGates[t] = rt
		g.hCandidates[t] = ht
		g.hiddenStates[t+1] = hNew
	}

	// Return output
	if g.returnSequences {
		// Return all hidden states [batch, seqLen, units]
		output := newTensor(batchSize, seqLen, g.units)
		for b := 0; b < batchSize; b++ {
			for t := 0; t < seqLen; t++ {
				for u := 0; u < g.units; u++ {
					output.data[b*seqLen*g.units+t*g.units+u] = g.hiddenStates[t+1].data[b*g.units+u]
				}
			}
		}
		return output, nil
	}

	// Return only last hidden state [batch, units]
	return g.hiddenStates[seqLen], nil
}

func (g *GRULayer) backward(gradOutput *tensor) (*tensor, error) {
	batchSize := g.inputs.shape[0]
	seqLen := g.inputs.shape[1]
	features := g.inputs.shape[2]

	// Zero gradients
	g.dWz.zeroGrad()
	g.dWr.zeroGrad()
	g.dWh.zeroGrad()
	g.dUz.zeroGrad()
	g.dUr.zeroGrad()
	g.dUh.zeroGrad()
	g.dbz.zeroGrad()
	g.dbr.zeroGrad()
	g.dbh.zeroGrad()

	gradInput := newTensor(g.inputs.shape...)
	dh := newTensor(batchSize, g.units)

	// If not returning sequences, gradOutput is [batch, units] for last step only
	if !g.returnSequences {
		copy(dh.data, gradOutput.data)
	}

	// Backprop through time
	for t := seqLen - 1; t >= 0; t-- {
		if g.returnSequences {
			// Add gradient for this timestep
			for b := 0; b < batchSize; b++ {
				for u := 0; u < g.units; u++ {
					dh.data[b*g.units+u] += gradOutput.data[b*seqLen*g.units+t*g.units+u]
				}
			}
		}

		// Extract cached values
		zt := g.zGates[t]
		rt := g.rGates[t]
		ht := g.hCandidates[t]
		hPrev := g.hiddenStates[t]

		// Extract x_t
		xt := newTensor(batchSize, features)
		for b := 0; b < batchSize; b++ {
			for f := 0; f < features; f++ {
				xt.data[b*features+f] = g.inputs.data[b*seqLen*features+t*features+f]
			}
		}

		dhPrev := newTensor(batchSize, g.units)

		for b := 0; b < batchSize; b++ {
			for u := 0; u < g.units; u++ {
				dhVal := dh.data[b*g.units+u]
				z := zt.data[b*g.units+u]
				r := rt.data[b*g.units+u]
				hCand := ht.data[b*g.units+u]
				hP := hPrev.data[b*g.units+u]

				// Gradient w.r.t h̃_t
				dhCand := dhVal * z * (1 - hCand*hCand) // tanh derivative

				// Gradient w.r.t z_t
				dz := dhVal * (hCand - hP) * z * (1 - z) // sigmoid derivative

				// Gradient w.r.t r_t (through h̃_t)
				dr := 0.0
				for h := 0; h < g.units; h++ {
					dr += dhCand * g.Uh.data[u*g.units+h] * hPrev.data[b*g.units+h]
				}
				dr *= r * (1 - r) // sigmoid derivative

				// Accumulate weight gradients
				for f := 0; f < features; f++ {
					g.dWz.data[f*g.units+u] += dz * xt.data[b*features+f]
					g.dWr.data[f*g.units+u] += dr * xt.data[b*features+f]
					g.dWh.data[f*g.units+u] += dhCand * xt.data[b*features+f]
				}

				for h := 0; h < g.units; h++ {
					g.dUz.data[h*g.units+u] += dz * hP
					g.dUr.data[h*g.units+u] += dr * hP
					g.dUh.data[h*g.units+u] += dhCand * r * hPrev.data[b*g.units+h]
				}

				g.dbz.data[u] += dz
				g.dbr.data[u] += dr
				g.dbh.data[u] += dhCand

				// Gradient w.r.t h_{t-1}
				dhPrev.data[b*g.units+u] += dhVal * (1 - z)
				for h := 0; h < g.units; h++ {
					dhPrev.data[b*g.units+h] += dz * g.Uz.data[h*g.units+u]
					dhPrev.data[b*g.units+h] += dr * g.Ur.data[h*g.units+u]
					dhPrev.data[b*g.units+h] += dhCand * r * g.Uh.data[h*g.units+u]
				}

				// Gradient w.r.t x_t
				for f := 0; f < features; f++ {
					gradInput.data[b*seqLen*features+t*features+f] += dz * g.Wz.data[f*g.units+u]
					gradInput.data[b*seqLen*features+t*features+f] += dr * g.Wr.data[f*g.units+u]
					gradInput.data[b*seqLen*features+t*features+f] += dhCand * g.Wh.data[f*g.units+u]
				}
			}
		}

		// Pass gradient to previous timestep
		copy(dh.data, dhPrev.data)
	}

	// Scale gradients by batch size
	scale := 1.0 / float64(batchSize)
	mulScalar(g.dWz, scale)
	mulScalar(g.dWr, scale)
	mulScalar(g.dWh, scale)
	mulScalar(g.dUz, scale)
	mulScalar(g.dUr, scale)
	mulScalar(g.dUh, scale)
	mulScalar(g.dbz, scale)
	mulScalar(g.dbr, scale)
	mulScalar(g.dbh, scale)

	return gradInput, nil
}

func (g *GRULayer) parameters() []*tensor {
	return []*tensor{g.Wz, g.Wr, g.Wh, g.Uz, g.Ur, g.Uh, g.bz, g.br, g.bh}
}

func (g *GRULayer) gradients() []*tensor {
	return []*tensor{g.dWz, g.dWr, g.dWh, g.dUz, g.dUr, g.dUh, g.dbz, g.dbr, g.dbh}
}

func (g *GRULayer) outputShape() []int {
	if g.returnSequences {
		return []int{g.seqLen, g.units}
	}
	return []int{g.units}
}

func (g *GRULayer) name() string { return "gru" }

// LSTMLayer - Long Short-Term Memory
// 4 gates: forget, input, cell candidate, output
type LSTMLayer struct {
	units           int
	returnSequences bool
	initializer     Initializer
	recurrentInit   Initializer
	biasInit        Initializer
	dropout         float64
	recurrentDrop   float64

	// Weights
	Wf, Wi, Wc, Wo *tensor // Input weights [inputDim, units]
	Uf, Ui, Uc, Uo *tensor // Recurrent weights [units, units]
	bf, bi, bc, bo *tensor // Biases [units]

	// Gradients
	dWf, dWi, dWc, dWo *tensor
	dUf, dUi, dUc, dUo *tensor
	dbf, dbi, dbc, dbo *tensor

	// Cache
	inputs       *tensor
	hiddenStates []*tensor
	cellStates   []*tensor
	fGates       []*tensor
	iGates       []*tensor
	cCandidates  []*tensor
	oGates       []*tensor

	inputDim int
	seqLen   int
	rng      *rand.Rand
	built    bool
}

type LSTMBuilder struct {
	layer *LSTMLayer
}

func LSTM(units int) *LSTMBuilder {
	return &LSTMBuilder{
		layer: &LSTMLayer{
			units:           units,
			returnSequences: false,
		},
	}
}

func (b *LSTMBuilder) WithReturnSequences(ret bool) *LSTMBuilder {
	b.layer.returnSequences = ret
	return b
}

func (b *LSTMBuilder) WithInitializer(init Initializer) *LSTMBuilder {
	b.layer.initializer = init
	return b
}

func (b *LSTMBuilder) WithRecurrentInitializer(init Initializer) *LSTMBuilder {
	b.layer.recurrentInit = init
	return b
}

func (b *LSTMBuilder) WithBiasInitializer(init Initializer) *LSTMBuilder {
	b.layer.biasInit = init
	return b
}

func (b *LSTMBuilder) WithDropout(rate float64) *LSTMBuilder {
	b.layer.dropout = rate
	return b
}

func (b *LSTMBuilder) WithRecurrentDropout(rate float64) *LSTMBuilder {
	b.layer.recurrentDrop = rate
	return b
}

func (b *LSTMBuilder) Build() Layer {
	return b.layer
}

func (l *LSTMLayer) build(inputShape []int, rng *rand.Rand) error {
	if len(inputShape) < 2 {
		return errors.New("flow: LSTM requires input shape [seqLen, features]")
	}
	if l.initializer == nil {
		return errors.New("flow: LSTM requires initializer")
	}
	if l.recurrentInit == nil {
		return errors.New("flow: LSTM requires recurrent initializer")
	}
	if l.biasInit == nil {
		return errors.New("flow: LSTM requires bias initializer")
	}

	l.seqLen = inputShape[0]
	l.inputDim = inputShape[1]
	l.rng = rng

	// Initialize input weights
	l.Wf = newTensor(l.inputDim, l.units)
	l.Wi = newTensor(l.inputDim, l.units)
	l.Wc = newTensor(l.inputDim, l.units)
	l.Wo = newTensor(l.inputDim, l.units)
	l.initializer.initialize(l.Wf, l.inputDim, l.units, rng)
	l.initializer.initialize(l.Wi, l.inputDim, l.units, rng)
	l.initializer.initialize(l.Wc, l.inputDim, l.units, rng)
	l.initializer.initialize(l.Wo, l.inputDim, l.units, rng)

	// Initialize recurrent weights
	l.Uf = newTensor(l.units, l.units)
	l.Ui = newTensor(l.units, l.units)
	l.Uc = newTensor(l.units, l.units)
	l.Uo = newTensor(l.units, l.units)
	l.recurrentInit.initialize(l.Uf, l.units, l.units, rng)
	l.recurrentInit.initialize(l.Ui, l.units, l.units, rng)
	l.recurrentInit.initialize(l.Uc, l.units, l.units, rng)
	l.recurrentInit.initialize(l.Uo, l.units, l.units, rng)

	// Initialize biases (forget gate bias often initialized to 1)
	l.bf = newTensor(l.units)
	l.bi = newTensor(l.units)
	l.bc = newTensor(l.units)
	l.bo = newTensor(l.units)
	l.bf.fill(1.0) // Forget gate bias = 1 for better gradient flow
	l.biasInit.initialize(l.bi, l.inputDim, l.units, rng)
	l.biasInit.initialize(l.bc, l.inputDim, l.units, rng)
	l.biasInit.initialize(l.bo, l.inputDim, l.units, rng)

	// Initialize gradients
	l.dWf = newTensor(l.inputDim, l.units)
	l.dWi = newTensor(l.inputDim, l.units)
	l.dWc = newTensor(l.inputDim, l.units)
	l.dWo = newTensor(l.inputDim, l.units)
	l.dUf = newTensor(l.units, l.units)
	l.dUi = newTensor(l.units, l.units)
	l.dUc = newTensor(l.units, l.units)
	l.dUo = newTensor(l.units, l.units)
	l.dbf = newTensor(l.units)
	l.dbi = newTensor(l.units)
	l.dbc = newTensor(l.units)
	l.dbo = newTensor(l.units)

	l.built = true
	return nil
}

func (l *LSTMLayer) forward(input *tensor, training bool) (*tensor, error) {
	if !l.built {
		return nil, errors.New("flow: LSTM not built")
	}

	batchSize := input.shape[0]
	seqLen := input.shape[1]
	features := input.shape[2]

	l.inputs = input
	l.hiddenStates = make([]*tensor, seqLen+1)
	l.cellStates = make([]*tensor, seqLen+1)
	l.fGates = make([]*tensor, seqLen)
	l.iGates = make([]*tensor, seqLen)
	l.cCandidates = make([]*tensor, seqLen)
	l.oGates = make([]*tensor, seqLen)

	// Initialize h_0 and c_0 to zeros
	l.hiddenStates[0] = newTensor(batchSize, l.units)
	l.cellStates[0] = newTensor(batchSize, l.units)

	for t := 0; t < seqLen; t++ {
		// Extract x_t
		xt := newTensor(batchSize, features)
		for b := 0; b < batchSize; b++ {
			for f := 0; f < features; f++ {
				xt.data[b*features+f] = input.data[b*seqLen*features+t*features+f]
			}
		}

		hPrev := l.hiddenStates[t]
		cPrev := l.cellStates[t]

		ft := newTensor(batchSize, l.units)
		it := newTensor(batchSize, l.units)
		ct := newTensor(batchSize, l.units)
		ot := newTensor(batchSize, l.units)
		cNew := newTensor(batchSize, l.units)
		hNew := newTensor(batchSize, l.units)

		for b := 0; b < batchSize; b++ {
			for u := 0; u < l.units; u++ {
				// Forget gate: f_t = σ(W_f * x_t + U_f * h_{t-1} + b_f)
				fVal := l.bf.data[u]
				for f := 0; f < features; f++ {
					fVal += xt.data[b*features+f] * l.Wf.data[f*l.units+u]
				}
				for h := 0; h < l.units; h++ {
					fVal += hPrev.data[b*l.units+h] * l.Uf.data[h*l.units+u]
				}
				ft.data[b*l.units+u] = sigmoid(fVal)

				// Input gate: i_t = σ(W_i * x_t + U_i * h_{t-1} + b_i)
				iVal := l.bi.data[u]
				for f := 0; f < features; f++ {
					iVal += xt.data[b*features+f] * l.Wi.data[f*l.units+u]
				}
				for h := 0; h < l.units; h++ {
					iVal += hPrev.data[b*l.units+h] * l.Ui.data[h*l.units+u]
				}
				it.data[b*l.units+u] = sigmoid(iVal)

				// Cell candidate: c̃_t = tanh(W_c * x_t + U_c * h_{t-1} + b_c)
				cVal := l.bc.data[u]
				for f := 0; f < features; f++ {
					cVal += xt.data[b*features+f] * l.Wc.data[f*l.units+u]
				}
				for h := 0; h < l.units; h++ {
					cVal += hPrev.data[b*l.units+h] * l.Uc.data[h*l.units+u]
				}
				ct.data[b*l.units+u] = math.Tanh(cVal)

				// Output gate: o_t = σ(W_o * x_t + U_o * h_{t-1} + b_o)
				oVal := l.bo.data[u]
				for f := 0; f < features; f++ {
					oVal += xt.data[b*features+f] * l.Wo.data[f*l.units+u]
				}
				for h := 0; h < l.units; h++ {
					oVal += hPrev.data[b*l.units+h] * l.Uo.data[h*l.units+u]
				}
				ot.data[b*l.units+u] = sigmoid(oVal)

				// Cell state: C_t = f_t ⊙ C_{t-1} + i_t ⊙ c̃_t
				cNew.data[b*l.units+u] = ft.data[b*l.units+u]*cPrev.data[b*l.units+u] +
					it.data[b*l.units+u]*ct.data[b*l.units+u]

				// Hidden state: h_t = o_t ⊙ tanh(C_t)
				hNew.data[b*l.units+u] = ot.data[b*l.units+u] * math.Tanh(cNew.data[b*l.units+u])
			}
		}

		l.fGates[t] = ft
		l.iGates[t] = it
		l.cCandidates[t] = ct
		l.oGates[t] = ot
		l.cellStates[t+1] = cNew
		l.hiddenStates[t+1] = hNew
	}

	if l.returnSequences {
		output := newTensor(batchSize, seqLen, l.units)
		for b := 0; b < batchSize; b++ {
			for t := 0; t < seqLen; t++ {
				for u := 0; u < l.units; u++ {
					output.data[b*seqLen*l.units+t*l.units+u] = l.hiddenStates[t+1].data[b*l.units+u]
				}
			}
		}
		return output, nil
	}

	return l.hiddenStates[seqLen], nil
}

func (l *LSTMLayer) backward(gradOutput *tensor) (*tensor, error) {
	batchSize := l.inputs.shape[0]
	seqLen := l.inputs.shape[1]
	features := l.inputs.shape[2]

	// Zero gradients
	l.dWf.zeroGrad()
	l.dWi.zeroGrad()
	l.dWc.zeroGrad()
	l.dWo.zeroGrad()
	l.dUf.zeroGrad()
	l.dUi.zeroGrad()
	l.dUc.zeroGrad()
	l.dUo.zeroGrad()
	l.dbf.zeroGrad()
	l.dbi.zeroGrad()
	l.dbc.zeroGrad()
	l.dbo.zeroGrad()

	gradInput := newTensor(l.inputs.shape...)
	dh := newTensor(batchSize, l.units)
	dc := newTensor(batchSize, l.units)

	if !l.returnSequences {
		copy(dh.data, gradOutput.data)
	}

	for t := seqLen - 1; t >= 0; t-- {
		if l.returnSequences {
			for b := 0; b < batchSize; b++ {
				for u := 0; u < l.units; u++ {
					dh.data[b*l.units+u] += gradOutput.data[b*seqLen*l.units+t*l.units+u]
				}
			}
		}

		ft := l.fGates[t]
		it := l.iGates[t]
		ct := l.cCandidates[t]
		ot := l.oGates[t]
		cNew := l.cellStates[t+1]
		cPrev := l.cellStates[t]
		hPrev := l.hiddenStates[t]

		xt := newTensor(batchSize, features)
		for b := 0; b < batchSize; b++ {
			for f := 0; f < features; f++ {
				xt.data[b*features+f] = l.inputs.data[b*seqLen*features+t*features+f]
			}
		}

		dhPrev := newTensor(batchSize, l.units)
		dcPrev := newTensor(batchSize, l.units)

		for b := 0; b < batchSize; b++ {
			for u := 0; u < l.units; u++ {
				dhVal := dh.data[b*l.units+u]
				o := ot.data[b*l.units+u]
				c := cNew.data[b*l.units+u]
				tanhC := math.Tanh(c)

				// dL/do = dL/dh * tanh(C)
				do := dhVal * tanhC * o * (1 - o)

				// dL/dC += dL/dh * o * (1 - tanh²(C)) + dL/dC_{t+1} * f_{t+1}
				dcVal := dc.data[b*l.units+u] + dhVal*o*(1-tanhC*tanhC)

				f := ft.data[b*l.units+u]
				i := it.data[b*l.units+u]
				cCand := ct.data[b*l.units+u]
				cP := cPrev.data[b*l.units+u]

				// dL/df = dL/dC * C_{t-1}
				df := dcVal * cP * f * (1 - f)

				// dL/di = dL/dC * c̃
				di := dcVal * cCand * i * (1 - i)

				// dL/dc̃ = dL/dC * i
				dCand := dcVal * i * (1 - cCand*cCand)

				// Accumulate weight gradients
				for fe := 0; fe < features; fe++ {
					l.dWf.data[fe*l.units+u] += df * xt.data[b*features+fe]
					l.dWi.data[fe*l.units+u] += di * xt.data[b*features+fe]
					l.dWc.data[fe*l.units+u] += dCand * xt.data[b*features+fe]
					l.dWo.data[fe*l.units+u] += do * xt.data[b*features+fe]
				}

				for h := 0; h < l.units; h++ {
					hP := hPrev.data[b*l.units+h]
					l.dUf.data[h*l.units+u] += df * hP
					l.dUi.data[h*l.units+u] += di * hP
					l.dUc.data[h*l.units+u] += dCand * hP
					l.dUo.data[h*l.units+u] += do * hP
				}

				l.dbf.data[u] += df
				l.dbi.data[u] += di
				l.dbc.data[u] += dCand
				l.dbo.data[u] += do

				// Gradient to previous cell state
				dcPrev.data[b*l.units+u] = dcVal * f

				// Gradient to previous hidden state
				for h := 0; h < l.units; h++ {
					dhPrev.data[b*l.units+h] += df * l.Uf.data[h*l.units+u]
					dhPrev.data[b*l.units+h] += di * l.Ui.data[h*l.units+u]
					dhPrev.data[b*l.units+h] += dCand * l.Uc.data[h*l.units+u]
					dhPrev.data[b*l.units+h] += do * l.Uo.data[h*l.units+u]
				}

				// Gradient to input
				for fe := 0; fe < features; fe++ {
					gradInput.data[b*seqLen*features+t*features+fe] += df * l.Wf.data[fe*l.units+u]
					gradInput.data[b*seqLen*features+t*features+fe] += di * l.Wi.data[fe*l.units+u]
					gradInput.data[b*seqLen*features+t*features+fe] += dCand * l.Wc.data[fe*l.units+u]
					gradInput.data[b*seqLen*features+t*features+fe] += do * l.Wo.data[fe*l.units+u]
				}
			}
		}

		copy(dh.data, dhPrev.data)
		copy(dc.data, dcPrev.data)
	}

	scale := 1.0 / float64(batchSize)
	mulScalar(l.dWf, scale)
	mulScalar(l.dWi, scale)
	mulScalar(l.dWc, scale)
	mulScalar(l.dWo, scale)
	mulScalar(l.dUf, scale)
	mulScalar(l.dUi, scale)
	mulScalar(l.dUc, scale)
	mulScalar(l.dUo, scale)
	mulScalar(l.dbf, scale)
	mulScalar(l.dbi, scale)
	mulScalar(l.dbc, scale)
	mulScalar(l.dbo, scale)

	return gradInput, nil
}

func (l *LSTMLayer) parameters() []*tensor {
	return []*tensor{
		l.Wf, l.Wi, l.Wc, l.Wo,
		l.Uf, l.Ui, l.Uc, l.Uo,
		l.bf, l.bi, l.bc, l.bo,
	}
}

func (l *LSTMLayer) gradients() []*tensor {
	return []*tensor{
		l.dWf, l.dWi, l.dWc, l.dWo,
		l.dUf, l.dUi, l.dUc, l.dUo,
		l.dbf, l.dbi, l.dbc, l.dbo,
	}
}

func (l *LSTMLayer) outputShape() []int {
	if l.returnSequences {
		return []int{l.seqLen, l.units}
	}
	return []int{l.units}
}

func (l *LSTMLayer) name() string { return "lstm" }

// SimpleRNNLayer - Basic RNN without gates
type SimpleRNNLayer struct {
	units           int
	returnSequences bool
	activation      Activation
	initializer     Initializer
	recurrentInit   Initializer
	biasInit        Initializer

	W  *tensor // [inputDim, units]
	U  *tensor // [units, units]
	b  *tensor // [units]
	dW *tensor
	dU *tensor
	db *tensor

	inputs       *tensor
	hiddenStates []*tensor
	preActs      []*tensor

	inputDim int
	seqLen   int
	rng      *rand.Rand
	built    bool
}

type SimpleRNNBuilder struct {
	layer *SimpleRNNLayer
}

func SimpleRNN(units int) *SimpleRNNBuilder {
	return &SimpleRNNBuilder{
		layer: &SimpleRNNLayer{
			units:           units,
			returnSequences: false,
		},
	}
}

func (b *SimpleRNNBuilder) WithReturnSequences(ret bool) *SimpleRNNBuilder {
	b.layer.returnSequences = ret
	return b
}

func (b *SimpleRNNBuilder) WithActivation(act Activation) *SimpleRNNBuilder {
	b.layer.activation = act
	return b
}

func (b *SimpleRNNBuilder) WithInitializer(init Initializer) *SimpleRNNBuilder {
	b.layer.initializer = init
	return b
}

func (b *SimpleRNNBuilder) WithRecurrentInitializer(init Initializer) *SimpleRNNBuilder {
	b.layer.recurrentInit = init
	return b
}

func (b *SimpleRNNBuilder) WithBiasInitializer(init Initializer) *SimpleRNNBuilder {
	b.layer.biasInit = init
	return b
}

func (b *SimpleRNNBuilder) Build() Layer {
	return b.layer
}

func (r *SimpleRNNLayer) build(inputShape []int, rng *rand.Rand) error {
	if len(inputShape) < 2 {
		return errors.New("flow: SimpleRNN requires input shape [seqLen, features]")
	}
	if r.activation == nil {
		return errors.New("flow: SimpleRNN requires activation")
	}
	if r.initializer == nil {
		return errors.New("flow: SimpleRNN requires initializer")
	}

	r.seqLen = inputShape[0]
	r.inputDim = inputShape[1]
	r.rng = rng

	r.W = newTensor(r.inputDim, r.units)
	r.U = newTensor(r.units, r.units)
	r.b = newTensor(r.units)

	r.initializer.initialize(r.W, r.inputDim, r.units, rng)
	if r.recurrentInit != nil {
		r.recurrentInit.initialize(r.U, r.units, r.units, rng)
	} else {
		r.initializer.initialize(r.U, r.units, r.units, rng)
	}
	if r.biasInit != nil {
		r.biasInit.initialize(r.b, r.inputDim, r.units, rng)
	}

	r.dW = newTensor(r.inputDim, r.units)
	r.dU = newTensor(r.units, r.units)
	r.db = newTensor(r.units)

	r.built = true
	return nil
}

func (r *SimpleRNNLayer) forward(input *tensor, training bool) (*tensor, error) {
	if !r.built {
		return nil, errors.New("flow: SimpleRNN not built")
	}

	batchSize := input.shape[0]
	seqLen := input.shape[1]
	features := input.shape[2]

	r.inputs = input
	r.hiddenStates = make([]*tensor, seqLen+1)
	r.preActs = make([]*tensor, seqLen)

	r.hiddenStates[0] = newTensor(batchSize, r.units)

	for t := 0; t < seqLen; t++ {
		xt := newTensor(batchSize, features)
		for b := 0; b < batchSize; b++ {
			for f := 0; f < features; f++ {
				xt.data[b*features+f] = input.data[b*seqLen*features+t*features+f]
			}
		}

		hPrev := r.hiddenStates[t]
		preAct := newTensor(batchSize, r.units)
		hNew := newTensor(batchSize, r.units)

		for b := 0; b < batchSize; b++ {
			for u := 0; u < r.units; u++ {
				val := r.b.data[u]
				for f := 0; f < features; f++ {
					val += xt.data[b*features+f] * r.W.data[f*r.units+u]
				}
				for h := 0; h < r.units; h++ {
					val += hPrev.data[b*r.units+h] * r.U.data[h*r.units+u]
				}
				preAct.data[b*r.units+u] = val
			}
		}

		r.activation.forward(preAct, hNew)
		r.preActs[t] = preAct
		r.hiddenStates[t+1] = hNew
	}

	if r.returnSequences {
		output := newTensor(batchSize, seqLen, r.units)
		for b := 0; b < batchSize; b++ {
			for t := 0; t < seqLen; t++ {
				for u := 0; u < r.units; u++ {
					output.data[b*seqLen*r.units+t*r.units+u] = r.hiddenStates[t+1].data[b*r.units+u]
				}
			}
		}
		return output, nil
	}

	return r.hiddenStates[seqLen], nil
}

func (r *SimpleRNNLayer) backward(gradOutput *tensor) (*tensor, error) {
	batchSize := r.inputs.shape[0]
	seqLen := r.inputs.shape[1]
	features := r.inputs.shape[2]

	r.dW.zeroGrad()
	r.dU.zeroGrad()
	r.db.zeroGrad()

	gradInput := newTensor(r.inputs.shape...)
	dh := newTensor(batchSize, r.units)

	if !r.returnSequences {
		copy(dh.data, gradOutput.data)
	}

	for t := seqLen - 1; t >= 0; t-- {
		if r.returnSequences {
			for b := 0; b < batchSize; b++ {
				for u := 0; u < r.units; u++ {
					dh.data[b*r.units+u] += gradOutput.data[b*seqLen*r.units+t*r.units+u]
				}
			}
		}

		hPrev := r.hiddenStates[t]
		preAct := r.preActs[t]

		// Gradient through activation
		dPreAct := newTensor(batchSize, r.units)
		r.activation.backward(preAct, dh, dPreAct)

		xt := newTensor(batchSize, features)
		for b := 0; b < batchSize; b++ {
			for f := 0; f < features; f++ {
				xt.data[b*features+f] = r.inputs.data[b*seqLen*features+t*features+f]
			}
		}

		dhPrev := newTensor(batchSize, r.units)

		for b := 0; b < batchSize; b++ {
			for u := 0; u < r.units; u++ {
				d := dPreAct.data[b*r.units+u]

				for f := 0; f < features; f++ {
					r.dW.data[f*r.units+u] += d * xt.data[b*features+f]
				}
				for h := 0; h < r.units; h++ {
					r.dU.data[h*r.units+u] += d * hPrev.data[b*r.units+h]
				}
				r.db.data[u] += d

				for h := 0; h < r.units; h++ {
					dhPrev.data[b*r.units+h] += d * r.U.data[h*r.units+u]
				}

				for f := 0; f < features; f++ {
					gradInput.data[b*seqLen*features+t*features+f] += d * r.W.data[f*r.units+u]
				}
			}
		}

		copy(dh.data, dhPrev.data)
	}

	scale := 1.0 / float64(batchSize)
	mulScalar(r.dW, scale)
	mulScalar(r.dU, scale)
	mulScalar(r.db, scale)

	return gradInput, nil
}

func (r *SimpleRNNLayer) parameters() []*tensor {
	return []*tensor{r.W, r.U, r.b}
}

func (r *SimpleRNNLayer) gradients() []*tensor {
	return []*tensor{r.dW, r.dU, r.db}
}

func (r *SimpleRNNLayer) outputShape() []int {
	if r.returnSequences {
		return []int{r.seqLen, r.units}
	}
	return []int{r.units}
}

func (r *SimpleRNNLayer) name() string { return "simple_rnn" }
