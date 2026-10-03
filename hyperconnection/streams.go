package hyperconnection

import (
	"fmt"
	"github.com/javanhut/GoTransformers/vectormath"
)

func ExpandToStreams(inputs vectormath.Matrix, numberOfStreams int) vectormath.Matrix {
	if numberOfStreams < 1 {
		panic(fmt.Sprintf("ExpandToStreams: needs at least 1 stream, got %d", numberOfStreams))
	}
	vectorSize := inputs.Columns
	streams := vectormath.NewMatrix(inputs.Rows, numberOfStreams*vectorSize)
	for token := 0; token < inputs.Rows; token++ {
		input := inputs.Row(token)
		streamRow := streams.Row(token)
		for stream := 0; stream < numberOfStreams; stream++ {
			copy(streamRow[stream*vectorSize:(stream+1)*vectorSize], input)
		}
	}
	return streams
}

func ExpandToStreamsBackward(streamGradients vectormath.Matrix, numberOfStreams int) vectormath.Matrix {
	vectorSize := checkStreamColumns("ExpandToStreamsBackward", streamGradients, numberOfStreams)
	inputGradients := vectormath.NewMatrix(streamGradients.Rows, vectorSize)
	for token := 0; token < streamGradients.Rows; token++ {
		gradientRow := streamGradients.Row(token)
		inputGradientRow := inputGradients.Row(token)
		for stream := 0; stream < numberOfStreams; stream++ {
			for i := 0; i < vectorSize; i++ {
				inputGradientRow[i] += gradientRow[stream*vectorSize+i]
			}
		}
	}
	return inputGradients
}

func CollapseStreams(streams vectormath.Matrix, numberOfStreams int) vectormath.Matrix {
	vectorSize := checkStreamColumns("CollapseStreams", streams, numberOfStreams)
	outputs := vectormath.NewMatrix(streams.Rows, vectorSize)
	for token := 0; token < streams.Rows; token++ {
		streamRow := streams.Row(token)
		outputRow := outputs.Row(token)
		for stream := 0; stream < numberOfStreams; stream++ {
			for i := 0; i < vectorSize; i++ {
				outputRow[i] += streamRow[stream*vectorSize+i] / float64(numberOfStreams)
			}
		}
	}
	return outputs
}

func CollapseStreamsBackward(outputGradients vectormath.Matrix, numberOfStreams int) vectormath.Matrix {
	if numberOfStreams < 1 {
		panic(fmt.Sprintf("CollapseStreamsBackward: needs at least 1 stream, got %d", numberOfStreams))
	}
	vectorSize := outputGradients.Columns
	streamGradients := vectormath.NewMatrix(outputGradients.Rows, numberOfStreams*vectorSize)
	for token := 0; token < outputGradients.Rows; token++ {
		outputGradientRow := outputGradients.Row(token)
		streamGradientRow := streamGradients.Row(token)
		for stream := 0; stream < numberOfStreams; stream++ {
			for i := 0; i < vectorSize; i++ {
				streamGradientRow[stream*vectorSize+i] = outputGradientRow[i] / float64(numberOfStreams)
			}
		}
	}
	return streamGradients
}

func checkStreamColumns(functionName string, streams vectormath.Matrix, numberOfStreams int) int {
	if numberOfStreams < 1 || streams.Columns%numberOfStreams != 0 {
		panic(fmt.Sprintf("%s: %d columns don't split evenly into %d streams", functionName, streams.Columns, numberOfStreams))
	}
	return streams.Columns / numberOfStreams
}
