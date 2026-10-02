package main

import (
	"fmt"
	"os"
	"path/filepath"
	"transformer/activationfunction"
	"transformer/lossfunction"
	"transformer/optimizer"
	"transformer/perceptron"
	"transformer/vectormath"
	"transformer/weightfile"
)

func main() {
	inputs := vectormath.MatrixFromRows([]vectormath.Vector{{0, 0}, {0, 1}, {1, 0}, {1, 1}})
	targets := vectormath.MatrixFromRows([]vectormath.Vector{{0}, {1}, {1}, {0}})

	network := perceptron.NewMultiLayerPerceptron([]int{2, 8, 1}, activationfunction.Tanh, activationfunction.Sigmoid)
	adam := optimizer.NewAdam(0.05)

	for step := 1; step <= 2000; step++ {
		loss := network.TrainStep(inputs, targets, lossfunction.MeanSquaredError, adam)
		if step%500 == 0 {
			fmt.Printf("step %d  loss %.6f\n", step, loss)
		}
	}

	weightsPath := filepath.Join(os.TempDir(), "xor_weights.txt")
	if err := weightfile.SaveText(weightsPath, network.Parameters()); err != nil {
		panic(err)
	}

	loadedNetwork := perceptron.NewMultiLayerPerceptron([]int{2, 8, 1}, activationfunction.Tanh, activationfunction.Sigmoid)
	if err := weightfile.LoadText(weightsPath, loadedNetwork.Parameters()); err != nil {
		panic(err)
	}

	fmt.Println("saved weights to", weightsPath)
	for row := 0; row < inputs.Rows; row++ {
		input := inputs.Row(row)
		fmt.Printf("%v -> trained %.3f  loaded from file %.3f\n", input, network.Predict(input)[0], loadedNetwork.Predict(input)[0])
	}
}
