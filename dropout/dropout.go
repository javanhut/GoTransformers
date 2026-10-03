package dropout

import (
	"fmt"
	"transformer/vectormath"
)

type Dropout struct {
	Rate            float64
	Active          bool
	RepeatLastMasks bool

	masks    [][]float64
	nextMask int
	lastMask []float64
}

func New(rate float64) *Dropout {
	if rate < 0 || rate >= 1 {
		panic(fmt.Sprintf("dropout.New: rate must be at least 0 and below 1, got %v", rate))
	}
	if rate == 0 {
		return nil
	}
	return &Dropout{Rate: rate}
}

func (dropout *Dropout) IsOn() bool {
	return dropout != nil && dropout.Active && dropout.Rate > 0
}

func (dropout *Dropout) SetActive(active bool) {
	if dropout != nil {
		dropout.Active = active
	}
}

func (dropout *Dropout) StartPass() {
	if dropout == nil {
		return
	}
	dropout.nextMask = 0
	if !dropout.RepeatLastMasks {
		dropout.masks = nil
	}
}

func (dropout *Dropout) NextMask(size int) []float64 {
	if !dropout.IsOn() {
		return nil
	}
	if dropout.RepeatLastMasks && dropout.nextMask < len(dropout.masks) && len(dropout.masks[dropout.nextMask]) == size {
		mask := dropout.masks[dropout.nextMask]
		dropout.nextMask++
		return mask
	}
	keptScale := 1 / (1 - dropout.Rate)
	mask := make([]float64, size)
	for i := range mask {
		if vectormath.RandomNumberBetween(0, 1) >= dropout.Rate {
			mask[i] = keptScale
		}
	}
	if dropout.nextMask < len(dropout.masks) {
		dropout.masks[dropout.nextMask] = mask
	} else {
		dropout.masks = append(dropout.masks, mask)
	}
	dropout.nextMask++
	return mask
}

func applyMask(matrix vectormath.Matrix, mask []float64) vectormath.Matrix {
	result := vectormath.NewMatrix(matrix.Rows, matrix.Columns)
	for i, value := range matrix.Values {
		result.Values[i] = value * mask[i]
	}
	return result
}

func (dropout *Dropout) Forward(inputs vectormath.Matrix) vectormath.Matrix {
	if !dropout.IsOn() {
		if dropout != nil {
			dropout.lastMask = nil
		}
		return inputs
	}
	dropout.StartPass()
	dropout.lastMask = dropout.NextMask(len(inputs.Values))
	return applyMask(inputs, dropout.lastMask)
}

func (dropout *Dropout) Backward(outputGradients vectormath.Matrix) vectormath.Matrix {
	if dropout == nil || dropout.lastMask == nil {
		return outputGradients
	}
	if len(dropout.lastMask) != len(outputGradients.Values) {
		panic(fmt.Sprintf("dropout: got %d gradients but the last Forward masked %d values", len(outputGradients.Values), len(dropout.lastMask)))
	}
	return applyMask(outputGradients, dropout.lastMask)
}
