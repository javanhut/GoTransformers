package optimizer_test

import (
	"github.com/javanhut/GoTransformers/activationfunction"
	"github.com/javanhut/GoTransformers/lossfunction"
	"github.com/javanhut/GoTransformers/optimizer"
	"github.com/javanhut/GoTransformers/perceptron"
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
	"testing"
)

func TestMuonLearnsXOR(t *testing.T) {
	inputs := vectormath.MatrixFromRows([]vectormath.Vector{{0, 0}, {0, 1}, {1, 0}, {1, 1}})
	targets := vectormath.MatrixFromRows([]vectormath.Vector{{0}, {1}, {1}, {0}})
	network := perceptron.NewMultiLayerPerceptron([]int{2, 8, 1}, activationfunction.Tanh, activationfunction.Sigmoid)
	muon := optimizer.NewMuon(0.05)

	for step := 0; step < 2000; step++ {
		network.TrainStep(inputs, targets, lossfunction.MeanSquaredError, muon)
	}

	for row := 0; row < inputs.Rows; row++ {
		prediction := network.Predict(inputs.Row(row))[0]
		if math.Abs(prediction-targets.Get(row, 0)) > 0.1 {
			t.Errorf("XOR%v = %v, want %v", inputs.Row(row), prediction, targets.Get(row, 0))
		}
	}
}
