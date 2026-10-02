package optimizer

import (
	"math"
	"testing"
	"transformer/parameter"
	"transformer/vectormath"
)

func largestDistanceFromIdentity(matrix vectormath.Matrix) float64 {
	largest := 0.0
	for row := 0; row < matrix.Rows; row++ {
		for column := 0; column < matrix.Columns; column++ {
			want := 0.0
			if row == column {
				want = 1
			}
			largest = math.Max(largest, math.Abs(matrix.Get(row, column)-want))
		}
	}
	return largest
}

func TestNewtonSchulzMakesSingularValuesOne(t *testing.T) {
	tall := NewtonSchulz(vectormath.NewRandomMatrix(8, 5, -1, 1))
	if tall.Rows != 8 || tall.Columns != 5 {
		t.Fatalf("8x5 came back as %dx%d", tall.Rows, tall.Columns)
	}
	if distance := largestDistanceFromIdentity(vectormath.TransposedTimesMatrix(tall, tall)); distance > 0.1 {
		t.Errorf("8x5: O^T O is %v away from the identity", distance)
	}

	wide := NewtonSchulz(vectormath.NewRandomMatrix(5, 8, -1, 1))
	if distance := largestDistanceFromIdentity(vectormath.MatrixTimesTransposed(wide, wide)); distance > 0.1 {
		t.Errorf("5x8: O O^T is %v away from the identity", distance)
	}

	zero := NewtonSchulz(vectormath.NewMatrix(3, 4))
	for _, value := range zero.Values {
		if value != 0 {
			t.Fatalf("a zero matrix came back as %v", zero.Values)
		}
	}
}

func TestAdamWAndMuonFindTheMinimum(t *testing.T) {
	optimizers := map[string]Optimizer{
		"AdamW": NewAdamW(0.01, 0.1),
		"Muon":  NewMuon(0.01),
	}
	target := []float64{3, -2, 0.5}
	for name, chosenOptimizer := range optimizers {
		result := minimize(chosenOptimizer, 3000)
		for i := range target {
			if math.Abs(result[i]-target[i]) > 1e-3 {
				t.Errorf("%s ended at %v, want %v", name, result, target)
				break
			}
		}
	}
}

func solveLeastSquares(chosenOptimizer Optimizer, steps int, learningRateAt func(step int) float64, setLearningRate func(float64)) float64 {
	inputs := vectormath.NewRandomMatrix(4, 12, -1, 1)
	targetWeights := vectormath.NewRandomMatrix(3, 4, -1, 1)
	targets := vectormath.MatrixTimesMatrix(targetWeights, inputs)

	weights := parameter.Parameter{
		Name:      "weights",
		Values:    make([]float64, 12),
		Gradients: make([]float64, 12),
		Rows:      3,
		Columns:   4,
	}
	for step := 0; step < steps; step++ {
		setLearningRate(learningRateAt(step))
		current := vectormath.Matrix{Rows: 3, Columns: 4, Values: weights.Values}
		errors := vectormath.SubtractMatrices(vectormath.MatrixTimesMatrix(current, inputs), targets)
		gradients := vectormath.ScaleMatrix(vectormath.MatrixTimesTransposed(errors, inputs), 2/float64(len(errors.Values)))
		copy(weights.Gradients, gradients.Values)
		chosenOptimizer.Update([]parameter.Parameter{weights})
	}

	largestError := 0.0
	for i, value := range weights.Values {
		largestError = math.Max(largestError, math.Abs(value-targetWeights.Values[i]))
	}
	return largestError
}

func slowingLearningRate(step int) float64 {
	if step < 400 {
		return 0.05
	}
	if step < 800 {
		return 0.01
	}
	return 0.002
}

func TestMatrixLeastSquares(t *testing.T) {
	adamW := NewAdamW(0.05, 0)
	if largestError := solveLeastSquares(adamW, 1200, slowingLearningRate, func(rate float64) { adamW.LearningRate = rate }); largestError > 0.02 {
		t.Errorf("AdamW: weights are up to %v away from the answer", largestError)
	}

	muon := NewMuon(0.05)
	muon.WeightDecay = 0
	if largestError := solveLeastSquares(muon, 1200, slowingLearningRate, func(rate float64) { muon.LearningRate = rate }); largestError > 0.02 {
		t.Errorf("Muon: weights are up to %v away from the answer", largestError)
	}
}

func TestAdamWOnlyDecaysMatrices(t *testing.T) {
	vector := parameter.Parameter{Name: "biases", Values: []float64{1, 2, 3}, Gradients: make([]float64, 3)}
	matrix := parameter.Parameter{Name: "weights", Values: []float64{1, 1, 1, 1}, Gradients: make([]float64, 4), Rows: 2, Columns: 2}
	adamW := NewAdamW(0.1, 0.5)
	for step := 0; step < 10; step++ {
		adamW.Update([]parameter.Parameter{vector, matrix})
	}
	if vector.Values[0] != 1 || vector.Values[1] != 2 || vector.Values[2] != 3 {
		t.Errorf("vector with no gradient changed to %v", vector.Values)
	}
	want := math.Pow(1-0.1*0.5, 10)
	if math.Abs(matrix.Values[0]-want) > 1e-12 {
		t.Errorf("matrix with no gradient is %v, want %v after decaying", matrix.Values[0], want)
	}
}

func TestMuonSendsUseAdamWParametersToAdamW(t *testing.T) {
	makeEmbedding := func() parameter.Parameter {
		return parameter.Parameter{
			Name:      "tokens.table",
			Values:    []float64{0.5, -0.25, 1, 2, -1, 0.75},
			Gradients: []float64{0.1, -0.3, 0.2, 0.05, -0.4, 0.6},
			Rows:      2,
			Columns:   3,
			UseAdamW:  true,
		}
	}
	throughMuon := makeEmbedding()
	throughAdamW := makeEmbedding()
	muon := NewMuon(0.01)
	adamW := NewAdamW(0.01, 0.1)
	for step := 0; step < 5; step++ {
		muon.Update([]parameter.Parameter{throughMuon})
		adamW.Update([]parameter.Parameter{throughAdamW})
	}
	for i := range throughMuon.Values {
		if throughMuon.Values[i] != throughAdamW.Values[i] {
			t.Fatalf("value %d: through Muon %v, through AdamW %v", i, throughMuon.Values[i], throughAdamW.Values[i])
		}
	}
	if throughMuon.Values[0] == 0.5 {
		t.Error("the embedding did not move at all")
	}
}

func TestMuonUpdateHasTheRightSize(t *testing.T) {
	weights := parameter.Parameter{
		Name:      "weights",
		Values:    make([]float64, 6*10),
		Gradients: vectormath.NewRandomMatrix(6, 10, -1, 1).Values,
		Rows:      6,
		Columns:   10,
	}
	muon := NewMuon(1)
	muon.WeightDecay = 0
	muon.Update([]parameter.Parameter{weights})
	rootMeanSquare := vectormath.Magnitude(weights.Values) / math.Sqrt(float64(len(weights.Values)))
	if math.Abs(rootMeanSquare-0.18) > 0.02 {
		t.Errorf("update root mean square is %v, want about 0.18", rootMeanSquare)
	}
}
