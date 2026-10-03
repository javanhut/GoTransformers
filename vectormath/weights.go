package vectormath

import (
	"fmt"
	"sync/atomic"
)

type WeightAwareBackend interface {
	MatrixTimesTransposedWeights(first Matrix, weights Matrix) Matrix
	MatrixTimesWeights(first Matrix, weights Matrix) Matrix
}

var weightsVersion atomic.Uint64

func MarkWeightsChanged() {
	weightsVersion.Add(1)
}

func WeightsVersion() uint64 {
	return weightsVersion.Load()
}

func MatrixTimesTransposedWeights(first Matrix, weights Matrix) Matrix {
	if first.Columns != weights.Columns {
		panic(fmt.Sprintf("MatrixTimesTransposedWeights: first matrix is %dx%d and weights are %dx%d, both need the same number of columns", first.Rows, first.Columns, weights.Rows, weights.Columns))
	}
	if backend, knowsWeights := currentBackend.(WeightAwareBackend); knowsWeights {
		return backend.MatrixTimesTransposedWeights(first, weights)
	}
	return currentBackend.MatrixTimesTransposed(first, weights)
}

func MatrixTimesWeights(first Matrix, weights Matrix) Matrix {
	if first.Columns != weights.Rows {
		panic(fmt.Sprintf("MatrixTimesWeights: first matrix is %dx%d and weights are %dx%d, first columns must equal weight rows", first.Rows, first.Columns, weights.Rows, weights.Columns))
	}
	if backend, knowsWeights := currentBackend.(WeightAwareBackend); knowsWeights {
		return backend.MatrixTimesWeights(first, weights)
	}
	return currentBackend.MatrixTimesMatrix(first, weights)
}
