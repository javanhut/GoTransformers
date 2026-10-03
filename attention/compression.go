package attention

import (
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
)

type compressionSource struct {
	entry       vectormath.Vector
	weight      vectormath.Vector
	bias        vectormath.Vector
	entryRow    int
	biasRow     int
	fromOverlap bool
}

func compress(sources []compressionSource, entrySize int) (vectormath.Vector, [][]float64) {
	compressed := vectormath.NewVector(entrySize)
	softmaxWeights := make([][]float64, len(sources))
	for i := range softmaxWeights {
		softmaxWeights[i] = make([]float64, entrySize)
	}
	for channel := range entrySize {
		largest := math.Inf(-1)
		for _, source := range sources {
			logit := source.weight[channel] + source.bias[channel]
			if logit > largest {
				largest = logit
			}
		}
		total := 0.0
		for i, source := range sources {
			softmaxWeights[i][channel] = math.Exp(source.weight[channel] + source.bias[channel] - largest)
			total += softmaxWeights[i][channel]
		}
		for i, source := range sources {
			softmaxWeights[i][channel] = softmaxWeights[i][channel] / total
			compressed[channel] += softmaxWeights[i][channel] * source.entry[channel]
		}
	}
	return compressed, softmaxWeights
}

type compressionGradients struct {
	entries []vectormath.Vector
	logits  []vectormath.Vector
}

func compressBackward(sources []compressionSource, softmaxWeights [][]float64, compressedGradient vectormath.Vector) compressionGradients {
	gradients := compressionGradients{
		entries: make([]vectormath.Vector, len(sources)),
		logits:  make([]vectormath.Vector, len(sources)),
	}
	for i := range sources {
		gradients.entries[i] = vectormath.NewVector(len(compressedGradient))
		gradients.logits[i] = vectormath.NewVector(len(compressedGradient))
	}
	for channel, gradient := range compressedGradient {
		weightedSum := 0.0
		for i, source := range sources {
			weightedSum += softmaxWeights[i][channel] * gradient * source.entry[channel]
		}
		for i, source := range sources {
			gradients.entries[i][channel] = gradient * softmaxWeights[i][channel]
			gradients.logits[i][channel] = softmaxWeights[i][channel] * (gradient*source.entry[channel] - weightedSum)
		}
	}
	return gradients
}
