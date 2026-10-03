package mixtureofexperts

import (
	"fmt"
	"github.com/javanhut/GoTransformers/activationfunction"
	"github.com/javanhut/GoTransformers/feedforward"
	"github.com/javanhut/GoTransformers/parameter"
	"github.com/javanhut/GoTransformers/perceptron"
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
	"sort"
)

type MixtureOfExperts struct {
	Name            string
	ExpertsPerToken int
	UseHashRouting  bool
	SharedExperts   []*feedforward.GatedFeedForward
	RoutedExperts   []*feedforward.GatedFeedForward
	RouterLayer     *perceptron.Layer
	BalanceBiases   vectormath.Vector

	lastNumberOfTokens int
	lastRouterLogits   vectormath.Matrix
	lastAffinities     vectormath.Matrix
	lastChosenExperts  [][]int
	lastGates          [][]float64
	lastRowsForExpert  [][]int
	lastExpertOutputs  []vectormath.Matrix
	lastExpertLoad     []int
}

func NewMixtureOfExperts(name string, vectorSize int, expertHiddenSize int, numberOfSharedExperts int, numberOfRoutedExperts int, expertsPerToken int) *MixtureOfExperts {
	makeExpert := func(expertName string) *feedforward.GatedFeedForward {
		return feedforward.NewSwiGLU(expertName, vectorSize, expertHiddenSize)
	}
	return buildMixtureOfExperts(name, vectorSize, numberOfSharedExperts, numberOfRoutedExperts, expertsPerToken, makeExpert)
}

func NewClampedMixtureOfExperts(name string, vectorSize int, expertHiddenSize int, numberOfSharedExperts int, numberOfRoutedExperts int, expertsPerToken int, clampLimit float64) *MixtureOfExperts {
	makeExpert := func(expertName string) *feedforward.GatedFeedForward {
		return feedforward.NewClampedSwiGLU(expertName, vectorSize, expertHiddenSize, clampLimit)
	}
	return buildMixtureOfExperts(name, vectorSize, numberOfSharedExperts, numberOfRoutedExperts, expertsPerToken, makeExpert)
}

func buildMixtureOfExperts(name string, vectorSize int, numberOfSharedExperts int, numberOfRoutedExperts int, expertsPerToken int, makeExpert func(expertName string) *feedforward.GatedFeedForward) *MixtureOfExperts {
	if numberOfSharedExperts < 0 {
		panic(fmt.Sprintf("NewMixtureOfExperts %q: number of shared experts can't be negative, got %d", name, numberOfSharedExperts))
	}
	if numberOfRoutedExperts < 1 {
		panic(fmt.Sprintf("NewMixtureOfExperts %q: needs at least 1 routed expert, got %d", name, numberOfRoutedExperts))
	}
	if expertsPerToken < 1 || expertsPerToken > numberOfRoutedExperts {
		panic(fmt.Sprintf("NewMixtureOfExperts %q: experts per token must be between 1 and %d, got %d", name, numberOfRoutedExperts, expertsPerToken))
	}
	mixture := &MixtureOfExperts{
		Name:            name,
		ExpertsPerToken: expertsPerToken,
		RouterLayer:     perceptron.NewLayer(name+".router", vectorSize, numberOfRoutedExperts, activationfunction.Linear),
		BalanceBiases:   vectormath.NewVector(numberOfRoutedExperts),
	}
	for i := 0; i < numberOfSharedExperts; i++ {
		mixture.SharedExperts = append(mixture.SharedExperts, makeExpert(fmt.Sprintf("%s.shared%d", name, i+1)))
	}
	for i := 0; i < numberOfRoutedExperts; i++ {
		mixture.RoutedExperts = append(mixture.RoutedExperts, makeExpert(fmt.Sprintf("%s.expert%d", name, i+1)))
	}
	return mixture
}

func (mixture *MixtureOfExperts) VectorSize() int {
	return mixture.RouterLayer.NumberOfInputs()
}

func (mixture *MixtureOfExperts) NumberOfRoutedExperts() int {
	return len(mixture.RoutedExperts)
}

func affinity(routerLogit float64) float64 {
	return math.Sqrt(activationfunction.Softplus.Forward(routerLogit))
}

func affinityDerivative(routerLogit float64, affinityValue float64) float64 {
	return activationfunction.Softplus.Derivative(routerLogit) / (2 * affinityValue)
}

func (mixture *MixtureOfExperts) hashedExperts(tokenID int) []int {
	numberOfExperts := mixture.NumberOfRoutedExperts()
	firstExpert := int((uint64(tokenID)*2654435761 + 12345) % uint64(numberOfExperts))
	chosen := make([]int, mixture.ExpertsPerToken)
	for i := range chosen {
		chosen[i] = (firstExpert + i) % numberOfExperts
	}
	return chosen
}

func (mixture *MixtureOfExperts) routedExperts(affinities vectormath.Vector) []int {
	candidates := make([]int, len(affinities))
	for expert := range candidates {
		candidates[expert] = expert
	}
	choiceScore := func(expert int) float64 {
		return affinities[expert] + mixture.BalanceBiases[expert]
	}
	sort.SliceStable(candidates, func(i int, j int) bool {
		return choiceScore(candidates[i]) > choiceScore(candidates[j])
	})
	chosen := candidates[:mixture.ExpertsPerToken]
	sort.Ints(chosen)
	return chosen
}

func gatherRows(matrix vectormath.Matrix, rows []int) vectormath.Matrix {
	gathered := vectormath.NewMatrix(len(rows), matrix.Columns)
	for i, row := range rows {
		gathered.SetRow(i, matrix.Row(row))
	}
	return gathered
}

func (mixture *MixtureOfExperts) Forward(inputs vectormath.Matrix, tokenIDs []int) vectormath.Matrix {
	if inputs.Columns != mixture.VectorSize() {
		panic(fmt.Sprintf("mixture of experts %q: each row has %d values but the experts take %d", mixture.Name, inputs.Columns, mixture.VectorSize()))
	}
	if len(mixture.BalanceBiases) != mixture.NumberOfRoutedExperts() {
		panic(fmt.Sprintf("mixture of experts %q: has %d routed experts but %d balance biases", mixture.Name, mixture.NumberOfRoutedExperts(), len(mixture.BalanceBiases)))
	}
	if mixture.UseHashRouting && len(tokenIDs) != inputs.Rows {
		panic(fmt.Sprintf("mixture of experts %q: hash routing needs one token ID per row, got %d token IDs for %d rows", mixture.Name, len(tokenIDs), inputs.Rows))
	}
	if !mixture.UseHashRouting && tokenIDs != nil && len(tokenIDs) != inputs.Rows {
		panic(fmt.Sprintf("mixture of experts %q: got %d token IDs for %d rows", mixture.Name, len(tokenIDs), inputs.Rows))
	}
	numberOfTokens := inputs.Rows

	chosenExperts := make([][]int, numberOfTokens)
	gates := make([][]float64, numberOfTokens)
	var routerLogits vectormath.Matrix
	var affinities vectormath.Matrix

	if mixture.UseHashRouting {
		for token := 0; token < numberOfTokens; token++ {
			if tokenIDs[token] < 0 {
				panic(fmt.Sprintf("mixture of experts %q: token ID %d at row %d is negative", mixture.Name, tokenIDs[token], token))
			}
			chosenExperts[token] = mixture.hashedExperts(tokenIDs[token])
			gates[token] = make([]float64, mixture.ExpertsPerToken)
			for i := range gates[token] {
				gates[token][i] = 1 / float64(mixture.ExpertsPerToken)
			}
		}
	} else {
		routerLogits = mixture.RouterLayer.Forward(inputs)
		affinities = vectormath.NewMatrix(numberOfTokens, mixture.NumberOfRoutedExperts())
		for token := 0; token < numberOfTokens; token++ {
			tokenAffinities := affinities.Row(token)
			for expert, routerLogit := range routerLogits.Row(token) {
				tokenAffinities[expert] = affinity(routerLogit)
			}
			chosen := mixture.routedExperts(tokenAffinities)
			chosenAffinityTotal := 0.0
			for _, expert := range chosen {
				chosenAffinityTotal += tokenAffinities[expert]
			}
			tokenGates := make([]float64, len(chosen))
			for i, expert := range chosen {
				tokenGates[i] = tokenAffinities[expert] / chosenAffinityTotal
			}
			chosenExperts[token] = chosen
			gates[token] = tokenGates
		}
	}

	rowsForExpert := make([][]int, mixture.NumberOfRoutedExperts())
	for token, chosen := range chosenExperts {
		for _, expert := range chosen {
			rowsForExpert[expert] = append(rowsForExpert[expert], token)
		}
	}

	outputs := vectormath.NewMatrix(numberOfTokens, mixture.VectorSize())
	for _, sharedExpert := range mixture.SharedExperts {
		outputs = vectormath.AddMatrices(outputs, sharedExpert.Forward(inputs))
	}

	expertOutputs := make([]vectormath.Matrix, mixture.NumberOfRoutedExperts())
	expertLoad := make([]int, mixture.NumberOfRoutedExperts())
	for expert, rows := range rowsForExpert {
		expertLoad[expert] = len(rows)
		if len(rows) == 0 {
			continue
		}
		expertOutputs[expert] = mixture.RoutedExperts[expert].Forward(gatherRows(inputs, rows))
	}

	for token, chosen := range chosenExperts {
		outputRow := outputs.Row(token)
		for i, expert := range chosen {
			expertOutputRow := expertOutputs[expert].Row(indexOf(rowsForExpert[expert], token))
			for j := range outputRow {
				outputRow[j] += gates[token][i] * expertOutputRow[j]
			}
		}
	}

	mixture.lastNumberOfTokens = numberOfTokens
	mixture.lastRouterLogits = routerLogits
	mixture.lastAffinities = affinities
	mixture.lastChosenExperts = chosenExperts
	mixture.lastGates = gates
	mixture.lastRowsForExpert = rowsForExpert
	mixture.lastExpertOutputs = expertOutputs
	mixture.lastExpertLoad = expertLoad
	return outputs
}

func indexOf(values []int, wanted int) int {
	for i, value := range values {
		if value == wanted {
			return i
		}
	}
	panic(fmt.Sprintf("indexOf: %d is not in the list", wanted))
}

func (mixture *MixtureOfExperts) Backward(outputGradients vectormath.Matrix) vectormath.Matrix {
	if mixture.lastChosenExperts == nil {
		panic(fmt.Sprintf("mixture of experts %q: call Forward before Backward", mixture.Name))
	}
	if outputGradients.Rows != mixture.lastNumberOfTokens || outputGradients.Columns != mixture.VectorSize() {
		panic(fmt.Sprintf("mixture of experts %q: output gradients are %dx%d but the last Forward produced %dx%d", mixture.Name, outputGradients.Rows, outputGradients.Columns, mixture.lastNumberOfTokens, mixture.VectorSize()))
	}
	numberOfTokens := mixture.lastNumberOfTokens

	inputGradients := vectormath.NewMatrix(numberOfTokens, mixture.VectorSize())
	for _, sharedExpert := range mixture.SharedExperts {
		inputGradients = vectormath.AddMatrices(inputGradients, sharedExpert.Backward(outputGradients))
	}

	for expert, rows := range mixture.lastRowsForExpert {
		if len(rows) == 0 {
			continue
		}
		expertOutputGradients := vectormath.NewMatrix(len(rows), mixture.VectorSize())
		for i, token := range rows {
			gate := mixture.lastGates[token][indexOf(mixture.lastChosenExperts[token], expert)]
			expertOutputGradients.SetRow(i, vectormath.Scale(outputGradients.Row(token), gate))
		}
		expertInputGradients := mixture.RoutedExperts[expert].Backward(expertOutputGradients)
		for i, token := range rows {
			inputGradientRow := inputGradients.Row(token)
			for j, gradient := range expertInputGradients.Row(i) {
				inputGradientRow[j] += gradient
			}
		}
	}

	if mixture.UseHashRouting {
		return inputGradients
	}

	routerLogitGradients := vectormath.NewMatrix(numberOfTokens, mixture.NumberOfRoutedExperts())
	for token, chosen := range mixture.lastChosenExperts {
		tokenGates := mixture.lastGates[token]
		tokenAffinities := mixture.lastAffinities.Row(token)

		gateGradients := make([]float64, len(chosen))
		weightedGateGradientSum := 0.0
		chosenAffinityTotal := 0.0
		for i, expert := range chosen {
			expertOutputRow := mixture.lastExpertOutputs[expert].Row(indexOf(mixture.lastRowsForExpert[expert], token))
			gateGradients[i] = vectormath.DotProduct(outputGradients.Row(token), expertOutputRow)
			weightedGateGradientSum += gateGradients[i] * tokenGates[i]
			chosenAffinityTotal += tokenAffinities[expert]
		}

		for i, expert := range chosen {
			affinityGradient := (gateGradients[i] - weightedGateGradientSum) / chosenAffinityTotal
			routerLogit := mixture.lastRouterLogits.Get(token, expert)
			routerLogitGradients.Set(token, expert, affinityGradient*affinityDerivative(routerLogit, tokenAffinities[expert]))
		}
	}
	return vectormath.AddMatrices(inputGradients, mixture.RouterLayer.Backward(routerLogitGradients))
}

func (mixture *MixtureOfExperts) LastExpertLoad() []int {
	load := make([]int, len(mixture.lastExpertLoad))
	copy(load, mixture.lastExpertLoad)
	return load
}

func (mixture *MixtureOfExperts) UpdateBalance(rate float64) {
	if mixture.lastExpertLoad == nil || mixture.UseHashRouting {
		return
	}
	totalLoad := 0
	for _, load := range mixture.lastExpertLoad {
		totalLoad += load
	}
	averageLoad := float64(totalLoad) / float64(len(mixture.lastExpertLoad))
	for expert, load := range mixture.lastExpertLoad {
		if float64(load) < averageLoad {
			mixture.BalanceBiases[expert] += rate
		}
		if float64(load) > averageLoad {
			mixture.BalanceBiases[expert] -= rate
		}
	}
}

func (mixture *MixtureOfExperts) Parameters() []parameter.Parameter {
	var parameters []parameter.Parameter
	for _, sharedExpert := range mixture.SharedExperts {
		parameters = append(parameters, sharedExpert.Parameters()...)
	}
	for _, routedExpert := range mixture.RoutedExperts {
		parameters = append(parameters, routedExpert.Parameters()...)
	}
	if !mixture.UseHashRouting {
		parameters = append(parameters, mixture.RouterLayer.Parameters()...)
	}
	return parameters
}

func (mixture *MixtureOfExperts) Layers() []*perceptron.Layer {
	var layers []*perceptron.Layer
	for _, expert := range mixture.SharedExperts {
		layers = append(layers, expert.Layers()...)
	}
	for _, expert := range mixture.RoutedExperts {
		layers = append(layers, expert.Layers()...)
	}
	return append(layers, mixture.RouterLayer)
}
