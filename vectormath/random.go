package vectormath

import "math/rand/v2"

var randomSource = rand.NewPCG(1, 1)

var randomNumbers = rand.New(randomSource)

func SetRandomSeed(seed uint64) {
	randomSource = rand.NewPCG(seed, seed)
	randomNumbers = rand.New(randomSource)
}

func SaveRandomState() ([]byte, error) {
	return randomSource.MarshalBinary()
}

func RestoreRandomState(state []byte) error {
	restored := rand.NewPCG(0, 0)
	if err := restored.UnmarshalBinary(state); err != nil {
		return err
	}
	randomSource = restored
	randomNumbers = rand.New(randomSource)
	return nil
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
