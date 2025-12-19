package flow

import (
	"errors"
	"math"
	"math/rand"
)

// MultiHeadAttentionLayer - Scaled Dot-Product Multi-Head Attention
// As described in "Attention Is All You Need" (Vaswani et al., 2017)
type MultiHeadAttentionLayer struct {
	numHeads    int
	keyDim      int
	valueDim    int
	dropout     float64
	useBias     bool
	initializer Initializer
	biasInit    Initializer

	// Projection weights [embedDim, numHeads * dim]
	Wq, Wk, Wv *tensor
	Wo         *tensor

	// Biases
	bq, bk, bv, bo *tensor

	// Gradients
	dWq, dWk, dWv, dWo *tensor
	dbq, dbk, dbv, dbo *tensor

	// Cache for backward pass
	query, key, value *tensor
	Q, K, V           *tensor
	attentionWeights  *tensor
	attentionOutput   *tensor

	embedDim int
	seqLen   int
	rng      *rand.Rand
	built    bool
}

type MultiHeadAttentionBuilder struct {
	layer *MultiHeadAttentionLayer
}

func MultiHeadAttention(numHeads, keyDim int) *MultiHeadAttentionBuilder {
	return &MultiHeadAttentionBuilder{
		layer: &MultiHeadAttentionLayer{
			numHeads: numHeads,
			keyDim:   keyDim,
			valueDim: keyDim, // Default same as keyDim
			useBias:  true,
		},
	}
}

func (b *MultiHeadAttentionBuilder) WithValueDim(dim int) *MultiHeadAttentionBuilder {
	b.layer.valueDim = dim
	return b
}

func (b *MultiHeadAttentionBuilder) WithDropout(rate float64) *MultiHeadAttentionBuilder {
	b.layer.dropout = rate
	return b
}

func (b *MultiHeadAttentionBuilder) WithBias(useBias bool) *MultiHeadAttentionBuilder {
	b.layer.useBias = useBias
	return b
}

func (b *MultiHeadAttentionBuilder) WithInitializer(init Initializer) *MultiHeadAttentionBuilder {
	b.layer.initializer = init
	return b
}

func (b *MultiHeadAttentionBuilder) WithBiasInitializer(init Initializer) *MultiHeadAttentionBuilder {
	b.layer.biasInit = init
	return b
}

func (b *MultiHeadAttentionBuilder) Build() Layer {
	return b.layer
}

func (m *MultiHeadAttentionLayer) build(inputShape []int, rng *rand.Rand) error {
	if len(inputShape) < 2 {
		return errors.New("flow: MultiHeadAttention requires input shape [seqLen, embedDim]")
	}
	if m.initializer == nil {
		return errors.New("flow: MultiHeadAttention requires initializer")
	}
	if m.useBias && m.biasInit == nil {
		return errors.New("flow: MultiHeadAttention with bias requires bias initializer")
	}

	// m.seqLen is not stored in struct, let's add it or rely on inputShape stored

	m.embedDim = inputShape[1]
	m.seqLen = inputShape[0] // Add this field to struct first
	m.rng = rng

	totalKeyDim := m.numHeads * m.keyDim
	totalValueDim := m.numHeads * m.valueDim

	// Query, Key, Value projection weights
	m.Wq = newTensor(m.embedDim, totalKeyDim)
	m.Wk = newTensor(m.embedDim, totalKeyDim)
	m.Wv = newTensor(m.embedDim, totalValueDim)
	m.Wo = newTensor(totalValueDim, m.embedDim)

	m.initializer.initialize(m.Wq, m.embedDim, totalKeyDim, rng)
	m.initializer.initialize(m.Wk, m.embedDim, totalKeyDim, rng)
	m.initializer.initialize(m.Wv, m.embedDim, totalValueDim, rng)
	m.initializer.initialize(m.Wo, totalValueDim, m.embedDim, rng)

	if m.useBias {
		m.bq = newTensor(totalKeyDim)
		m.bk = newTensor(totalKeyDim)
		m.bv = newTensor(totalValueDim)
		m.bo = newTensor(m.embedDim)
		m.biasInit.initialize(m.bq, m.embedDim, totalKeyDim, rng)
		m.biasInit.initialize(m.bk, m.embedDim, totalKeyDim, rng)
		m.biasInit.initialize(m.bv, m.embedDim, totalValueDim, rng)
		m.biasInit.initialize(m.bo, totalValueDim, m.embedDim, rng)
	}

	// Gradients
	m.dWq = newTensor(m.embedDim, totalKeyDim)
	m.dWk = newTensor(m.embedDim, totalKeyDim)
	m.dWv = newTensor(m.embedDim, totalValueDim)
	m.dWo = newTensor(totalValueDim, m.embedDim)

	if m.useBias {
		m.dbq = newTensor(totalKeyDim)
		m.dbk = newTensor(totalKeyDim)
		m.dbv = newTensor(totalValueDim)
		m.dbo = newTensor(m.embedDim)
	}

	m.built = true
	return nil
}

func (m *MultiHeadAttentionLayer) forward(input *tensor, training bool) (*tensor, error) {
	if !m.built {
		return nil, errors.New("flow: MultiHeadAttention not built")
	}

	// For self-attention: query = key = value = input
	// Input shape: [batch, seqLen, embedDim]
	batchSize := input.shape[0]
	seqLen := input.shape[1]
	embedDim := input.shape[2]

	m.query = input
	m.key = input
	m.value = input

	totalKeyDim := m.numHeads * m.keyDim
	totalValueDim := m.numHeads * m.valueDim

	// Project Q, K, V: [batch, seqLen, embedDim] -> [batch, seqLen, totalDim]
	m.Q = newTensor(batchSize, seqLen, totalKeyDim)
	m.K = newTensor(batchSize, seqLen, totalKeyDim)
	m.V = newTensor(batchSize, seqLen, totalValueDim)

	// Linear projections
	for b := 0; b < batchSize; b++ {
		for t := 0; t < seqLen; t++ {
			for d := 0; d < totalKeyDim; d++ {
				qVal := 0.0
				kVal := 0.0
				if m.useBias {
					qVal = m.bq.data[d]
					kVal = m.bk.data[d]
				}
				for e := 0; e < embedDim; e++ {
					inputVal := input.data[b*seqLen*embedDim+t*embedDim+e]
					qVal += inputVal * m.Wq.data[e*totalKeyDim+d]
					kVal += inputVal * m.Wk.data[e*totalKeyDim+d]
				}
				m.Q.data[b*seqLen*totalKeyDim+t*totalKeyDim+d] = qVal
				m.K.data[b*seqLen*totalKeyDim+t*totalKeyDim+d] = kVal
			}
			for d := 0; d < totalValueDim; d++ {
				vVal := 0.0
				if m.useBias {
					vVal = m.bv.data[d]
				}
				for e := 0; e < embedDim; e++ {
					vVal += input.data[b*seqLen*embedDim+t*embedDim+e] * m.Wv.data[e*totalValueDim+d]
				}
				m.V.data[b*seqLen*totalValueDim+t*totalValueDim+d] = vVal
			}
		}
	}

	// Scaled dot-product attention per head
	// Reshape to [batch, numHeads, seqLen, dim]
	scale := 1.0 / math.Sqrt(float64(m.keyDim))

	// Attention weights: [batch, numHeads, seqLen, seqLen]
	m.attentionWeights = newTensor(batchSize, m.numHeads, seqLen, seqLen)

	// Compute attention scores and softmax
	for b := 0; b < batchSize; b++ {
		for h := 0; h < m.numHeads; h++ {
			// Q[b, h] @ K[b, h].T -> [seqLen, seqLen]
			for i := 0; i < seqLen; i++ {
				// Compute softmax for row i
				maxVal := math.Inf(-1)
				for j := 0; j < seqLen; j++ {
					score := 0.0
					for d := 0; d < m.keyDim; d++ {
						qIdx := b*seqLen*totalKeyDim + i*totalKeyDim + h*m.keyDim + d
						kIdx := b*seqLen*totalKeyDim + j*totalKeyDim + h*m.keyDim + d
						score += m.Q.data[qIdx] * m.K.data[kIdx]
					}
					score *= scale
					m.attentionWeights.data[b*m.numHeads*seqLen*seqLen+h*seqLen*seqLen+i*seqLen+j] = score
					if score > maxVal {
						maxVal = score
					}
				}

				// Softmax
				sumExp := 0.0
				for j := 0; j < seqLen; j++ {
					idx := b*m.numHeads*seqLen*seqLen + h*seqLen*seqLen + i*seqLen + j
					m.attentionWeights.data[idx] = math.Exp(m.attentionWeights.data[idx] - maxVal)
					sumExp += m.attentionWeights.data[idx]
				}
				for j := 0; j < seqLen; j++ {
					idx := b*m.numHeads*seqLen*seqLen + h*seqLen*seqLen + i*seqLen + j
					m.attentionWeights.data[idx] /= sumExp
				}
			}
		}
	}

	// Apply attention to values: attention_weights @ V
	// [batch, numHeads, seqLen, seqLen] @ [batch, numHeads, seqLen, valueDim]
	// -> [batch, numHeads, seqLen, valueDim]
	m.attentionOutput = newTensor(batchSize, seqLen, totalValueDim)

	for b := 0; b < batchSize; b++ {
		for h := 0; h < m.numHeads; h++ {
			for i := 0; i < seqLen; i++ {
				for d := 0; d < m.valueDim; d++ {
					val := 0.0
					for j := 0; j < seqLen; j++ {
						attnIdx := b*m.numHeads*seqLen*seqLen + h*seqLen*seqLen + i*seqLen + j
						vIdx := b*seqLen*totalValueDim + j*totalValueDim + h*m.valueDim + d
						val += m.attentionWeights.data[attnIdx] * m.V.data[vIdx]
					}
					outIdx := b*seqLen*totalValueDim + i*totalValueDim + h*m.valueDim + d
					m.attentionOutput.data[outIdx] = val
				}
			}
		}
	}

	// Output projection: [batch, seqLen, totalValueDim] -> [batch, seqLen, embedDim]
	output := newTensor(batchSize, seqLen, embedDim)

	for b := 0; b < batchSize; b++ {
		for t := 0; t < seqLen; t++ {
			for e := 0; e < embedDim; e++ {
				val := 0.0
				if m.useBias {
					val = m.bo.data[e]
				}
				for d := 0; d < totalValueDim; d++ {
					val += m.attentionOutput.data[b*seqLen*totalValueDim+t*totalValueDim+d] * m.Wo.data[d*embedDim+e]
				}
				output.data[b*seqLen*embedDim+t*embedDim+e] = val
			}
		}
	}

	return output, nil
}

func (m *MultiHeadAttentionLayer) backward(gradOutput *tensor) (*tensor, error) {
	batchSize := gradOutput.shape[0]
	seqLen := gradOutput.shape[1]
	embedDim := gradOutput.shape[2]

	totalKeyDim := m.numHeads * m.keyDim
	totalValueDim := m.numHeads * m.valueDim

	// Zero gradients
	m.dWq.zeroGrad()
	m.dWk.zeroGrad()
	m.dWv.zeroGrad()
	m.dWo.zeroGrad()
	if m.useBias {
		m.dbq.zeroGrad()
		m.dbk.zeroGrad()
		m.dbv.zeroGrad()
		m.dbo.zeroGrad()
	}

	// Gradient of output projection
	dAttentionOutput := newTensor(batchSize, seqLen, totalValueDim)

	for b := 0; b < batchSize; b++ {
		for t := 0; t < seqLen; t++ {
			for e := 0; e < embedDim; e++ {
				dout := gradOutput.data[b*seqLen*embedDim+t*embedDim+e]
				if m.useBias {
					m.dbo.data[e] += dout
				}
				for d := 0; d < totalValueDim; d++ {
					m.dWo.data[d*embedDim+e] += m.attentionOutput.data[b*seqLen*totalValueDim+t*totalValueDim+d] * dout
					dAttentionOutput.data[b*seqLen*totalValueDim+t*totalValueDim+d] += m.Wo.data[d*embedDim+e] * dout
				}
			}
		}
	}

	// Gradient through attention weights and values
	dAttentionWeights := newTensor(batchSize, m.numHeads, seqLen, seqLen)
	dV := newTensor(batchSize, seqLen, totalValueDim)

	for b := 0; b < batchSize; b++ {
		for h := 0; h < m.numHeads; h++ {
			for i := 0; i < seqLen; i++ {
				for d := 0; d < m.valueDim; d++ {
					outIdx := b*seqLen*totalValueDim + i*totalValueDim + h*m.valueDim + d
					dout := dAttentionOutput.data[outIdx]
					for j := 0; j < seqLen; j++ {
						attnIdx := b*m.numHeads*seqLen*seqLen + h*seqLen*seqLen + i*seqLen + j
						vIdx := b*seqLen*totalValueDim + j*totalValueDim + h*m.valueDim + d
						dAttentionWeights.data[attnIdx] += dout * m.V.data[vIdx]
						dV.data[vIdx] += dout * m.attentionWeights.data[attnIdx]
					}
				}
			}
		}
	}

	// Gradient through softmax
	dScores := newTensor(batchSize, m.numHeads, seqLen, seqLen)
	for b := 0; b < batchSize; b++ {
		for h := 0; h < m.numHeads; h++ {
			for i := 0; i < seqLen; i++ {
				// Softmax gradient: dScore = attn * (dAttn - sum(attn * dAttn))
				sumAttnDAttn := 0.0
				for j := 0; j < seqLen; j++ {
					idx := b*m.numHeads*seqLen*seqLen + h*seqLen*seqLen + i*seqLen + j
					sumAttnDAttn += m.attentionWeights.data[idx] * dAttentionWeights.data[idx]
				}
				for j := 0; j < seqLen; j++ {
					idx := b*m.numHeads*seqLen*seqLen + h*seqLen*seqLen + i*seqLen + j
					dScores.data[idx] = m.attentionWeights.data[idx] * (dAttentionWeights.data[idx] - sumAttnDAttn)
				}
			}
		}
	}

	// Scale gradient
	scale := 1.0 / math.Sqrt(float64(m.keyDim))

	// Gradient w.r.t Q and K
	dQ := newTensor(batchSize, seqLen, totalKeyDim)
	dK := newTensor(batchSize, seqLen, totalKeyDim)

	for b := 0; b < batchSize; b++ {
		for h := 0; h < m.numHeads; h++ {
			for i := 0; i < seqLen; i++ {
				for j := 0; j < seqLen; j++ {
					scoreIdx := b*m.numHeads*seqLen*seqLen + h*seqLen*seqLen + i*seqLen + j
					ds := dScores.data[scoreIdx] * scale
					for d := 0; d < m.keyDim; d++ {
						qIdx := b*seqLen*totalKeyDim + i*totalKeyDim + h*m.keyDim + d
						kIdx := b*seqLen*totalKeyDim + j*totalKeyDim + h*m.keyDim + d
						dQ.data[qIdx] += ds * m.K.data[kIdx]
						dK.data[kIdx] += ds * m.Q.data[qIdx]
					}
				}
			}
		}
	}

	// Gradient through input projections
	gradInput := newTensor(batchSize, seqLen, embedDim)

	for b := 0; b < batchSize; b++ {
		for t := 0; t < seqLen; t++ {
			for e := 0; e < embedDim; e++ {
				val := 0.0
				// From Q projection
				for d := 0; d < totalKeyDim; d++ {
					dq := dQ.data[b*seqLen*totalKeyDim+t*totalKeyDim+d]
					m.dWq.data[e*totalKeyDim+d] += m.query.data[b*seqLen*embedDim+t*embedDim+e] * dq
					val += m.Wq.data[e*totalKeyDim+d] * dq
				}
				// From K projection
				for d := 0; d < totalKeyDim; d++ {
					dk := dK.data[b*seqLen*totalKeyDim+t*totalKeyDim+d]
					m.dWk.data[e*totalKeyDim+d] += m.key.data[b*seqLen*embedDim+t*embedDim+e] * dk
					val += m.Wk.data[e*totalKeyDim+d] * dk
				}
				// From V projection
				for d := 0; d < totalValueDim; d++ {
					dv := dV.data[b*seqLen*totalValueDim+t*totalValueDim+d]
					m.dWv.data[e*totalValueDim+d] += m.value.data[b*seqLen*embedDim+t*embedDim+e] * dv
					val += m.Wv.data[e*totalValueDim+d] * dv
				}
				gradInput.data[b*seqLen*embedDim+t*embedDim+e] = val
			}

			// Bias gradients
			if m.useBias {
				for d := 0; d < totalKeyDim; d++ {
					m.dbq.data[d] += dQ.data[b*seqLen*totalKeyDim+t*totalKeyDim+d]
					m.dbk.data[d] += dK.data[b*seqLen*totalKeyDim+t*totalKeyDim+d]
				}
				for d := 0; d < totalValueDim; d++ {
					m.dbv.data[d] += dV.data[b*seqLen*totalValueDim+t*totalValueDim+d]
				}
			}
		}
	}

	// Scale gradients
	scaleFactor := 1.0 / float64(batchSize)
	mulScalar(m.dWq, scaleFactor)
	mulScalar(m.dWk, scaleFactor)
	mulScalar(m.dWv, scaleFactor)
	mulScalar(m.dWo, scaleFactor)
	if m.useBias {
		mulScalar(m.dbq, scaleFactor)
		mulScalar(m.dbk, scaleFactor)
		mulScalar(m.dbv, scaleFactor)
		mulScalar(m.dbo, scaleFactor)
	}

	return gradInput, nil
}

func (m *MultiHeadAttentionLayer) parameters() []*tensor {
	if m.useBias {
		return []*tensor{m.Wq, m.Wk, m.Wv, m.Wo, m.bq, m.bk, m.bv, m.bo}
	}
	return []*tensor{m.Wq, m.Wk, m.Wv, m.Wo}
}

func (m *MultiHeadAttentionLayer) gradients() []*tensor {
	if m.useBias {
		return []*tensor{m.dWq, m.dWk, m.dWv, m.dWo, m.dbq, m.dbk, m.dbv, m.dbo}
	}
	return []*tensor{m.dWq, m.dWk, m.dWv, m.dWo}
}

func (m *MultiHeadAttentionLayer) outputShape() []int {
	return []int{m.seqLen, m.embedDim}
}

func (m *MultiHeadAttentionLayer) name() string { return "multi_head_attention" }

// SelfAttentionLayer - Simplified single-head self-attention
type SelfAttentionLayer struct {
	embedDim    int
	useBias     bool
	initializer Initializer
	biasInit    Initializer

	Wq, Wk, Wv    *tensor
	bq, bk, bv    *tensor
	dWq, dWk, dWv *tensor
	dbq, dbk, dbv *tensor

	input            *tensor
	Q, K, V          *tensor
	attentionWeights *tensor

	seqLen int
	rng    *rand.Rand
	built  bool
}

type SelfAttentionBuilder struct {
	layer *SelfAttentionLayer
}

func SelfAttention(embedDim int) *SelfAttentionBuilder {
	return &SelfAttentionBuilder{
		layer: &SelfAttentionLayer{
			embedDim: embedDim,
			useBias:  true,
		},
	}
}

func (b *SelfAttentionBuilder) WithBias(useBias bool) *SelfAttentionBuilder {
	b.layer.useBias = useBias
	return b
}

func (b *SelfAttentionBuilder) WithInitializer(init Initializer) *SelfAttentionBuilder {
	b.layer.initializer = init
	return b
}

func (b *SelfAttentionBuilder) WithBiasInitializer(init Initializer) *SelfAttentionBuilder {
	b.layer.biasInit = init
	return b
}

func (b *SelfAttentionBuilder) Build() Layer {
	return b.layer
}

func (s *SelfAttentionLayer) build(inputShape []int, rng *rand.Rand) error {
	if len(inputShape) < 2 {
		return errors.New("flow: SelfAttention requires input shape [seqLen, embedDim]")
	}
	if s.initializer == nil {
		return errors.New("flow: SelfAttention requires initializer")
	}

	s.seqLen = inputShape[0]
	s.rng = rng

	s.Wq = newTensor(s.embedDim, s.embedDim)
	s.Wk = newTensor(s.embedDim, s.embedDim)
	s.Wv = newTensor(s.embedDim, s.embedDim)
	s.initializer.initialize(s.Wq, s.embedDim, s.embedDim, rng)
	s.initializer.initialize(s.Wk, s.embedDim, s.embedDim, rng)
	s.initializer.initialize(s.Wv, s.embedDim, s.embedDim, rng)

	if s.useBias {
		s.bq = newTensor(s.embedDim)
		s.bk = newTensor(s.embedDim)
		s.bv = newTensor(s.embedDim)
		if s.biasInit != nil {
			s.biasInit.initialize(s.bq, s.embedDim, s.embedDim, rng)
			s.biasInit.initialize(s.bk, s.embedDim, s.embedDim, rng)
			s.biasInit.initialize(s.bv, s.embedDim, s.embedDim, rng)
		}
	}

	s.dWq = newTensor(s.embedDim, s.embedDim)
	s.dWk = newTensor(s.embedDim, s.embedDim)
	s.dWv = newTensor(s.embedDim, s.embedDim)
	if s.useBias {
		s.dbq = newTensor(s.embedDim)
		s.dbk = newTensor(s.embedDim)
		s.dbv = newTensor(s.embedDim)
	}

	s.built = true
	return nil
}

func (s *SelfAttentionLayer) forward(input *tensor, training bool) (*tensor, error) {
	if !s.built {
		return nil, errors.New("flow: SelfAttention not built")
	}

	batchSize := input.shape[0]
	seqLen := input.shape[1]
	embedDim := input.shape[2]

	s.input = input
	s.Q = newTensor(batchSize, seqLen, embedDim)
	s.K = newTensor(batchSize, seqLen, embedDim)
	s.V = newTensor(batchSize, seqLen, embedDim)

	// Linear projections
	for b := 0; b < batchSize; b++ {
		for t := 0; t < seqLen; t++ {
			for d := 0; d < embedDim; d++ {
				qVal, kVal, vVal := 0.0, 0.0, 0.0
				if s.useBias {
					qVal = s.bq.data[d]
					kVal = s.bk.data[d]
					vVal = s.bv.data[d]
				}
				for e := 0; e < embedDim; e++ {
					inputVal := input.data[b*seqLen*embedDim+t*embedDim+e]
					qVal += inputVal * s.Wq.data[e*embedDim+d]
					kVal += inputVal * s.Wk.data[e*embedDim+d]
					vVal += inputVal * s.Wv.data[e*embedDim+d]
				}
				s.Q.data[b*seqLen*embedDim+t*embedDim+d] = qVal
				s.K.data[b*seqLen*embedDim+t*embedDim+d] = kVal
				s.V.data[b*seqLen*embedDim+t*embedDim+d] = vVal
			}
		}
	}

	// Scaled dot-product attention
	scale := 1.0 / math.Sqrt(float64(embedDim))
	s.attentionWeights = newTensor(batchSize, seqLen, seqLen)
	output := newTensor(batchSize, seqLen, embedDim)

	for b := 0; b < batchSize; b++ {
		for i := 0; i < seqLen; i++ {
			// Compute attention scores
			maxVal := math.Inf(-1)
			for j := 0; j < seqLen; j++ {
				score := 0.0
				for d := 0; d < embedDim; d++ {
					score += s.Q.data[b*seqLen*embedDim+i*embedDim+d] * s.K.data[b*seqLen*embedDim+j*embedDim+d]
				}
				score *= scale
				s.attentionWeights.data[b*seqLen*seqLen+i*seqLen+j] = score
				if score > maxVal {
					maxVal = score
				}
			}

			// Softmax
			sumExp := 0.0
			for j := 0; j < seqLen; j++ {
				idx := b*seqLen*seqLen + i*seqLen + j
				s.attentionWeights.data[idx] = math.Exp(s.attentionWeights.data[idx] - maxVal)
				sumExp += s.attentionWeights.data[idx]
			}
			for j := 0; j < seqLen; j++ {
				idx := b*seqLen*seqLen + i*seqLen + j
				s.attentionWeights.data[idx] /= sumExp
			}

			// Apply attention to values
			for d := 0; d < embedDim; d++ {
				val := 0.0
				for j := 0; j < seqLen; j++ {
					val += s.attentionWeights.data[b*seqLen*seqLen+i*seqLen+j] * s.V.data[b*seqLen*embedDim+j*embedDim+d]
				}
				output.data[b*seqLen*embedDim+i*embedDim+d] = val
			}
		}
	}

	return output, nil
}

func (s *SelfAttentionLayer) backward(gradOutput *tensor) (*tensor, error) {
	// Simplified backward pass
	batchSize := gradOutput.shape[0]
	seqLen := gradOutput.shape[1]
	embedDim := gradOutput.shape[2]

	s.dWq.zeroGrad()
	s.dWk.zeroGrad()
	s.dWv.zeroGrad()
	if s.useBias {
		s.dbq.zeroGrad()
		s.dbk.zeroGrad()
		s.dbv.zeroGrad()
	}

	gradInput := newTensor(s.input.shape...)

	// Gradient through value projection and attention
	dV := newTensor(batchSize, seqLen, embedDim)
	dAttn := newTensor(batchSize, seqLen, seqLen)

	for b := 0; b < batchSize; b++ {
		for i := 0; i < seqLen; i++ {
			for d := 0; d < embedDim; d++ {
				dout := gradOutput.data[b*seqLen*embedDim+i*embedDim+d]
				for j := 0; j < seqLen; j++ {
					attnIdx := b*seqLen*seqLen + i*seqLen + j
					dV.data[b*seqLen*embedDim+j*embedDim+d] += dout * s.attentionWeights.data[attnIdx]
					dAttn.data[attnIdx] += dout * s.V.data[b*seqLen*embedDim+j*embedDim+d]
				}
			}
		}
	}

	// Gradient through softmax
	scale := 1.0 / math.Sqrt(float64(embedDim))
	dQ := newTensor(batchSize, seqLen, embedDim)
	dK := newTensor(batchSize, seqLen, embedDim)

	for b := 0; b < batchSize; b++ {
		for i := 0; i < seqLen; i++ {
			sumAttnDAttn := 0.0
			for j := 0; j < seqLen; j++ {
				idx := b*seqLen*seqLen + i*seqLen + j
				sumAttnDAttn += s.attentionWeights.data[idx] * dAttn.data[idx]
			}
			for j := 0; j < seqLen; j++ {
				idx := b*seqLen*seqLen + i*seqLen + j
				ds := s.attentionWeights.data[idx] * (dAttn.data[idx] - sumAttnDAttn) * scale
				for d := 0; d < embedDim; d++ {
					dQ.data[b*seqLen*embedDim+i*embedDim+d] += ds * s.K.data[b*seqLen*embedDim+j*embedDim+d]
					dK.data[b*seqLen*embedDim+j*embedDim+d] += ds * s.Q.data[b*seqLen*embedDim+i*embedDim+d]
				}
			}
		}
	}

	// Gradient through projections
	for b := 0; b < batchSize; b++ {
		for t := 0; t < seqLen; t++ {
			for e := 0; e < embedDim; e++ {
				val := 0.0
				for d := 0; d < embedDim; d++ {
					dq := dQ.data[b*seqLen*embedDim+t*embedDim+d]
					dk := dK.data[b*seqLen*embedDim+t*embedDim+d]
					dv := dV.data[b*seqLen*embedDim+t*embedDim+d]
					inputVal := s.input.data[b*seqLen*embedDim+t*embedDim+e]

					s.dWq.data[e*embedDim+d] += inputVal * dq
					s.dWk.data[e*embedDim+d] += inputVal * dk
					s.dWv.data[e*embedDim+d] += inputVal * dv

					val += s.Wq.data[e*embedDim+d]*dq + s.Wk.data[e*embedDim+d]*dk + s.Wv.data[e*embedDim+d]*dv
				}
				gradInput.data[b*seqLen*embedDim+t*embedDim+e] = val
			}

			if s.useBias {
				for d := 0; d < embedDim; d++ {
					s.dbq.data[d] += dQ.data[b*seqLen*embedDim+t*embedDim+d]
					s.dbk.data[d] += dK.data[b*seqLen*embedDim+t*embedDim+d]
					s.dbv.data[d] += dV.data[b*seqLen*embedDim+t*embedDim+d]
				}
			}
		}
	}

	scaleFactor := 1.0 / float64(batchSize)
	mulScalar(s.dWq, scaleFactor)
	mulScalar(s.dWk, scaleFactor)
	mulScalar(s.dWv, scaleFactor)
	if s.useBias {
		mulScalar(s.dbq, scaleFactor)
		mulScalar(s.dbk, scaleFactor)
		mulScalar(s.dbv, scaleFactor)
	}

	return gradInput, nil
}

func (s *SelfAttentionLayer) parameters() []*tensor {
	if s.useBias {
		return []*tensor{s.Wq, s.Wk, s.Wv, s.bq, s.bk, s.bv}
	}
	return []*tensor{s.Wq, s.Wk, s.Wv}
}

func (s *SelfAttentionLayer) gradients() []*tensor {
	if s.useBias {
		return []*tensor{s.dWq, s.dWk, s.dWv, s.dbq, s.dbk, s.dbv}
	}
	return []*tensor{s.dWq, s.dWk, s.dWv}
}

func (s *SelfAttentionLayer) outputShape() []int {
	return []int{s.seqLen, s.embedDim}
}

func (s *SelfAttentionLayer) name() string { return "self_attention" }
