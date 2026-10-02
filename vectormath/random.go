package vectormath

import "math/rand/v2"

var randomNumbers = rand.New(rand.NewPCG(1, 1))

func SetRandomSeed(seed uint64) {
	randomNumbers = rand.New(rand.NewPCG(seed, seed))
}

func RandomNumberBetween(lowest float64, highest float64) float64 {
	return lowest + randomNumbers.Float64()*(highest-lowest)
}

func NewRandomMatrix(rows int, columns int, lowest float64, highest float64) Matrix {
	matrix := NewMatrix(rows, columns)
	for i := range matrix.Values {
		matrix.Values[i] = RandomNumberBetween(lowest, highest)
	}
	return matrix
}
