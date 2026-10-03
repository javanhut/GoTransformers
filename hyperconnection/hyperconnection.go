package hyperconnection

import (
	"fmt"
	"github.com/javanhut/GoTransformers/activationfunction"
	"github.com/javanhut/GoTransformers/normalization"
	"github.com/javanhut/GoTransformers/parameter"
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
)

const startingGate = 0.01

const startingResidualDiagonal = 4.0

const largestStartingInputAmount = 0.99

type HyperConnection struct {
	Name            string
	VectorSize      int
	NumberOfStreams int
	SinkhornSteps   int

	Norm *normalization.RMSNorm

	InputWeights    vectormath.Matrix
	ResidualWeights vectormath.Matrix
	OutputWeights   vectormath.Matrix
	InputGate       vectormath.Vector
	ResidualGate    vectormath.Vector
	OutputGate      vectormath.Vector
	InputBiases     vectormath.Vector
	ResidualBiases  vectormath.Vector
	OutputBiases    vectormath.Vector

	InputWeightGradients    vectormath.Matrix
	ResidualWeightGradients vectormath.Matrix
	OutputWeightGradients   vectormath.Matrix
	InputGateGradients      vectormath.Vector
	ResidualGateGradients   vectormath.Vector
	OutputGateGradients     vectormath.Vector
	InputBiasGradients      vectormath.Vector
	ResidualBiasGradients   vectormath.Vector
	OutputBiasGradients     vectormath.Vector

	lastStreams            vectormath.Matrix
	lastNormalized         vectormath.Matrix
	lastInputProjection    vectormath.Matrix
	lastResidualProjection vectormath.Matrix
	lastOutputProjection   vectormath.Matrix
	lastInputAmounts       vectormath.Matrix
	lastOutputAmounts      vectormath.Matrix
	lastResidualMixing     []vectormath.Matrix
	lastSinkhornRecords    []sinkhornRecord
	lastLayerOutput        vectormath.Matrix
}

func logit(probability float64) float64 {
	return math.Log(probability / (1 - probability))
}

func NewHyperConnection(name string, vectorSize int, numberOfStreams int) *HyperConnection {
	if vectorSize < 1 || numberOfStreams < 1 {
		panic(fmt.Sprintf("NewHyperConnection %q: vector size and number of streams must be at least 1, got %d and %d", name, vectorSize, numberOfStreams))
	}
	flattenedSize := numberOfStreams * vectorSize
	residualSize := numberOfStreams * numberOfStreams
	inputLimit := math.Sqrt(6 / float64(flattenedSize+numberOfStreams))
	residualLimit := math.Sqrt(6 / float64(flattenedSize+residualSize))

	connection := &HyperConnection{
		Name:            name,
		VectorSize:      vectorSize,
		NumberOfStreams: numberOfStreams,
		SinkhornSteps:   20,
		Norm:            normalization.NewRMSNorm(name+".norm", flattenedSize),

		InputWeights:    vectormath.NewRandomMatrix(numberOfStreams, flattenedSize, -inputLimit, inputLimit),
		ResidualWeights: vectormath.NewRandomMatrix(residualSize, flattenedSize, -residualLimit, residualLimit),
		OutputWeights:   vectormath.NewRandomMatrix(numberOfStreams, flattenedSize, -inputLimit, inputLimit),
		InputGate:       vectormath.Vector{startingGate},
		ResidualGate:    vectormath.Vector{startingGate},
		OutputGate:      vectormath.Vector{startingGate},
		InputBiases:     vectormath.NewVector(numberOfStreams),
		ResidualBiases:  vectormath.NewVector(residualSize),
		OutputBiases:    vectormath.NewVector(numberOfStreams),

		InputWeightGradients:    vectormath.NewMatrix(numberOfStreams, flattenedSize),
		ResidualWeightGradients: vectormath.NewMatrix(residualSize, flattenedSize),
		OutputWeightGradients:   vectormath.NewMatrix(numberOfStreams, flattenedSize),
		InputGateGradients:      vectormath.NewVector(1),
		ResidualGateGradients:   vectormath.NewVector(1),
		OutputGateGradients:     vectormath.NewVector(1),
		InputBiasGradients:      vectormath.NewVector(numberOfStreams),
		ResidualBiasGradients:   vectormath.NewVector(residualSize),
		OutputBiasGradients:     vectormath.NewVector(numberOfStreams),
	}

	startingInputAmount := 1 / float64(numberOfStreams)
	if startingInputAmount > largestStartingInputAmount {
		startingInputAmount = largestStartingInputAmount
	}
	for stream := 0; stream < numberOfStreams; stream++ {
		connection.InputBiases[stream] = logit(startingInputAmount)
		connection.ResidualBiases[stream*numberOfStreams+stream] = startingResidualDiagonal
	}
	return connection
}

func (connection *HyperConnection) streamPart(row vectormath.Vector, stream int) vectormath.Vector {
	start := stream * connection.VectorSize
	end := start + connection.VectorSize
	return row[start:end:end]
}

func addInto(target vectormath.Matrix, addition vectormath.Matrix) {
	for i := range target.Values {
		target.Values[i] += addition.Values[i]
	}
}

func (connection *HyperConnection) LayerInput(streams vectormath.Matrix) vectormath.Matrix {
	if streams.Columns != connection.NumberOfStreams*connection.VectorSize {
		panic(fmt.Sprintf("hyper connection %q: each row has %d values but %d streams of %d need %d", connection.Name, streams.Columns, connection.NumberOfStreams, connection.VectorSize, connection.NumberOfStreams*connection.VectorSize))
	}
	if connection.SinkhornSteps < 1 {
		panic(fmt.Sprintf("hyper connection %q: SinkhornSteps must be at least 1, got %d", connection.Name, connection.SinkhornSteps))
	}
	numberOfTokens := streams.Rows
	numberOfStreams := connection.NumberOfStreams

	normalized := connection.Norm.Forward(streams)
	inputProjection := vectormath.MatrixTimesTransposed(normalized, connection.InputWeights)
	residualProjection := vectormath.MatrixTimesTransposed(normalized, connection.ResidualWeights)
	outputProjection := vectormath.MatrixTimesTransposed(normalized, connection.OutputWeights)

	inputAmounts := vectormath.NewMatrix(numberOfTokens, numberOfStreams)
	outputAmounts := vectormath.NewMatrix(numberOfTokens, numberOfStreams)
	residualMixing := make([]vectormath.Matrix, numberOfTokens)
	sinkhornRecords := make([]sinkhornRecord, numberOfTokens)
	layerInput := vectormath.NewMatrix(numberOfTokens, connection.VectorSize)

	for token := 0; token < numberOfTokens; token++ {
		for stream := 0; stream < numberOfStreams; stream++ {
			rawInputAmount := connection.InputGate[0]*inputProjection.Get(token, stream) + connection.InputBiases[stream]
			inputAmounts.Set(token, stream, activationfunction.Sigmoid.Forward(rawInputAmount))
			rawOutputAmount := connection.OutputGate[0]*outputProjection.Get(token, stream) + connection.OutputBiases[stream]
			outputAmounts.Set(token, stream, 2*activationfunction.Sigmoid.Forward(rawOutputAmount))
		}

		rawMixing := vectormath.NewMatrix(numberOfStreams, numberOfStreams)
		for i := range rawMixing.Values {
			rawMixing.Values[i] = connection.ResidualGate[0]*residualProjection.Get(token, i) + connection.ResidualBiases[i]
		}
		residualMixing[token], sinkhornRecords[token] = sinkhornKnopp(rawMixing, connection.SinkhornSteps)

		streamRow := streams.Row(token)
		layerInputRow := layerInput.Row(token)
		for stream := 0; stream < numberOfStreams; stream++ {
			amount := inputAmounts.Get(token, stream)
			streamValues := connection.streamPart(streamRow, stream)
			for i := range layerInputRow {
				layerInputRow[i] += amount * streamValues[i]
			}
		}
	}

	connection.lastStreams = streams
	connection.lastNormalized = normalized
	connection.lastInputProjection = inputProjection
	connection.lastResidualProjection = residualProjection
	connection.lastOutputProjection = outputProjection
	connection.lastInputAmounts = inputAmounts
	connection.lastOutputAmounts = outputAmounts
	connection.lastResidualMixing = residualMixing
	connection.lastSinkhornRecords = sinkhornRecords
	connection.lastLayerOutput = vectormath.Matrix{}
	return layerInput
}

func (connection *HyperConnection) Combine(layerOutput vectormath.Matrix) vectormath.Matrix {
	if connection.lastResidualMixing == nil {
		panic(fmt.Sprintf("hyper connection %q: call LayerInput before Combine", connection.Name))
	}
	if layerOutput.Rows != connection.lastStreams.Rows || layerOutput.Columns != connection.VectorSize {
		panic(fmt.Sprintf("hyper connection %q: layer output is %dx%d but LayerInput gave %dx%d", connection.Name, layerOutput.Rows, layerOutput.Columns, connection.lastStreams.Rows, connection.VectorSize))
	}
	numberOfStreams := connection.NumberOfStreams
	newStreams := vectormath.NewMatrix(layerOutput.Rows, numberOfStreams*connection.VectorSize)
	for token := 0; token < layerOutput.Rows; token++ {
		streamRow := connection.lastStreams.Row(token)
		newStreamRow := newStreams.Row(token)
		layerOutputRow := layerOutput.Row(token)
		mixing := connection.lastResidualMixing[token]
		for stream := 0; stream < numberOfStreams; stream++ {
			target := connection.streamPart(newStreamRow, stream)
			for otherStream := 0; otherStream < numberOfStreams; otherStream++ {
				weight := mixing.Get(stream, otherStream)
				source := connection.streamPart(streamRow, otherStream)
				for i := range target {
					target[i] += weight * source[i]
				}
			}
			amount := connection.lastOutputAmounts.Get(token, stream)
			for i := range target {
				target[i] += amount * layerOutputRow[i]
			}
		}
	}
	connection.lastLayerOutput = layerOutput
	return newStreams
}

func (connection *HyperConnection) BackwardCombine(newStreamGradients vectormath.Matrix) (vectormath.Matrix, vectormath.Matrix) {
	if connection.lastLayerOutput.Values == nil {
		panic(fmt.Sprintf("hyper connection %q: call LayerInput and Combine before BackwardCombine", connection.Name))
	}
	if newStreamGradients.Rows != connection.lastStreams.Rows || newStreamGradients.Columns != connection.lastStreams.Columns {
		panic(fmt.Sprintf("hyper connection %q: stream gradients are %dx%d but the streams were %dx%d", connection.Name, newStreamGradients.Rows, newStreamGradients.Columns, connection.lastStreams.Rows, connection.lastStreams.Columns))
	}
	numberOfTokens := newStreamGradients.Rows
	numberOfStreams := connection.NumberOfStreams

	layerOutputGradients := vectormath.NewMatrix(numberOfTokens, connection.VectorSize)
	streamGradients := vectormath.NewMatrix(numberOfTokens, newStreamGradients.Columns)
	residualProjectionGradients := vectormath.NewMatrix(numberOfTokens, numberOfStreams*numberOfStreams)
	outputProjectionGradients := vectormath.NewMatrix(numberOfTokens, numberOfStreams)

	for token := 0; token < numberOfTokens; token++ {
		gradientRow := newStreamGradients.Row(token)
		streamRow := connection.lastStreams.Row(token)
		streamGradientRow := streamGradients.Row(token)
		layerOutputRow := connection.lastLayerOutput.Row(token)
		layerOutputGradientRow := layerOutputGradients.Row(token)
		mixing := connection.lastResidualMixing[token]
		mixingGradients := vectormath.NewMatrix(numberOfStreams, numberOfStreams)

		for stream := 0; stream < numberOfStreams; stream++ {
			targetGradient := connection.streamPart(gradientRow, stream)

			amount := connection.lastOutputAmounts.Get(token, stream)
			amountGradient := 0.0
			for i := range targetGradient {
				layerOutputGradientRow[i] += amount * targetGradient[i]
				amountGradient += targetGradient[i] * layerOutputRow[i]
			}

			for otherStream := 0; otherStream < numberOfStreams; otherStream++ {
				weight := mixing.Get(stream, otherStream)
				source := connection.streamPart(streamRow, otherStream)
				sourceGradient := connection.streamPart(streamGradientRow, otherStream)
				weightGradient := 0.0
				for i := range targetGradient {
					weightGradient += targetGradient[i] * source[i]
					sourceGradient[i] += weight * targetGradient[i]
				}
				mixingGradients.Set(stream, otherStream, weightGradient)
			}

			rawOutputAmountGradient := amountGradient * amount * (1 - amount/2)
			connection.OutputBiasGradients[stream] += rawOutputAmountGradient
			connection.OutputGateGradients[0] += rawOutputAmountGradient * connection.lastOutputProjection.Get(token, stream)
			outputProjectionGradients.Set(token, stream, rawOutputAmountGradient*connection.OutputGate[0])
		}

		rawMixingGradients := sinkhornKnoppBackward(connection.lastSinkhornRecords[token], mixingGradients)
		for i, rawGradient := range rawMixingGradients.Values {
			connection.ResidualBiasGradients[i] += rawGradient
			connection.ResidualGateGradients[0] += rawGradient * connection.lastResidualProjection.Get(token, i)
			residualProjectionGradients.Set(token, i, rawGradient*connection.ResidualGate[0])
		}
	}

	addInto(connection.ResidualWeightGradients, vectormath.TransposedTimesMatrix(residualProjectionGradients, connection.lastNormalized))
	addInto(connection.OutputWeightGradients, vectormath.TransposedTimesMatrix(outputProjectionGradients, connection.lastNormalized))
	normalizedGradients := vectormath.MatrixTimesMatrix(residualProjectionGradients, connection.ResidualWeights)
	addInto(normalizedGradients, vectormath.MatrixTimesMatrix(outputProjectionGradients, connection.OutputWeights))
	addInto(streamGradients, connection.Norm.Backward(normalizedGradients))

	return layerOutputGradients, streamGradients
}

func (connection *HyperConnection) BackwardLayerInput(layerInputGradients vectormath.Matrix) vectormath.Matrix {
	if connection.lastResidualMixing == nil {
		panic(fmt.Sprintf("hyper connection %q: call LayerInput before BackwardLayerInput", connection.Name))
	}
	if layerInputGradients.Rows != connection.lastStreams.Rows || layerInputGradients.Columns != connection.VectorSize {
		panic(fmt.Sprintf("hyper connection %q: layer input gradients are %dx%d but LayerInput gave %dx%d", connection.Name, layerInputGradients.Rows, layerInputGradients.Columns, connection.lastStreams.Rows, connection.VectorSize))
	}
	numberOfTokens := layerInputGradients.Rows
	numberOfStreams := connection.NumberOfStreams

	streamGradients := vectormath.NewMatrix(numberOfTokens, connection.lastStreams.Columns)
	inputProjectionGradients := vectormath.NewMatrix(numberOfTokens, numberOfStreams)

	for token := 0; token < numberOfTokens; token++ {
		layerInputGradientRow := layerInputGradients.Row(token)
		streamRow := connection.lastStreams.Row(token)
		streamGradientRow := streamGradients.Row(token)
		for stream := 0; stream < numberOfStreams; stream++ {
			amount := connection.lastInputAmounts.Get(token, stream)
			streamValues := connection.streamPart(streamRow, stream)
			streamValueGradients := connection.streamPart(streamGradientRow, stream)
			amountGradient := 0.0
			for i := range layerInputGradientRow {
				streamValueGradients[i] += amount * layerInputGradientRow[i]
				amountGradient += layerInputGradientRow[i] * streamValues[i]
			}

			rawInputAmountGradient := amountGradient * amount * (1 - amount)
			connection.InputBiasGradients[stream] += rawInputAmountGradient
			connection.InputGateGradients[0] += rawInputAmountGradient * connection.lastInputProjection.Get(token, stream)
			inputProjectionGradients.Set(token, stream, rawInputAmountGradient*connection.InputGate[0])
		}
	}

	addInto(connection.InputWeightGradients, vectormath.TransposedTimesMatrix(inputProjectionGradients, connection.lastNormalized))
	normalizedGradients := vectormath.MatrixTimesMatrix(inputProjectionGradients, connection.InputWeights)
	addInto(streamGradients, connection.Norm.Backward(normalizedGradients))
	return streamGradients
}

func (connection *HyperConnection) LastResidualMixing(token int) vectormath.Matrix {
	return connection.lastResidualMixing[token]
}

func (connection *HyperConnection) Parameters() []parameter.Parameter {
	var parameters []parameter.Parameter
	for _, normParameter := range connection.Norm.Parameters() {
		normParameter.UseAdamW = true
		parameters = append(parameters, normParameter)
	}
	matrix := func(name string, values vectormath.Matrix, gradients *vectormath.Matrix) parameter.Parameter {
		return parameter.Parameter{Name: connection.Name + "." + name, Values: values.Values, GradientStorage: &gradients.Values, Rows: values.Rows, Columns: values.Columns}
	}
	vector := func(name string, values vectormath.Vector, gradients *vectormath.Vector) parameter.Parameter {
		return parameter.Parameter{Name: connection.Name + "." + name, Values: values, GradientStorage: (*[]float64)(gradients), UseAdamW: true}
	}
	parameters = append(parameters,
		matrix("inputWeights", connection.InputWeights, &connection.InputWeightGradients),
		matrix("residualWeights", connection.ResidualWeights, &connection.ResidualWeightGradients),
		matrix("outputWeights", connection.OutputWeights, &connection.OutputWeightGradients),
		vector("inputGate", connection.InputGate, &connection.InputGateGradients),
		vector("residualGate", connection.ResidualGate, &connection.ResidualGateGradients),
		vector("outputGate", connection.OutputGate, &connection.OutputGateGradients),
		vector("inputBiases", connection.InputBiases, &connection.InputBiasGradients),
		vector("residualBiases", connection.ResidualBiases, &connection.ResidualBiasGradients),
		vector("outputBiases", connection.OutputBiases, &connection.OutputBiasGradients),
	)
	return parameters
}
