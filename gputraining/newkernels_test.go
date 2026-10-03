package gputraining

import (
	"github.com/javanhut/GoTransformers/embedding"
	"github.com/javanhut/GoTransformers/gpu"
	"math"
	"testing"
)

func TestGatherEmbeddingsWithAndWithoutPositions(t *testing.T) {
	test := openKernelTest(t)
	vocabularySize := 7
	vectorSize := 5
	sequenceLength := 3
	table := randomValues(vocabularySize*vectorSize, -1, 1)
	tokenIDs := []float64{4, 0, 6, 6, 2, 1}
	rows := len(tokenIDs)
	positionVectors := make([]float64, sequenceLength*vectorSize)
	for position := range sequenceLength {
		copy(positionVectors[position*vectorSize:], embedding.PositionalEncodingAt(position, vectorSize))
	}
	tableBuffer := test.bufferWith(table)
	tokenBuffer := test.bufferWith(tokenIDs)
	positionBuffer := test.bufferWith(positionVectors)
	plainOutputs := test.emptyBuffer(rows * vectorSize)
	positionedOutputs := test.emptyBuffer(rows * vectorSize)
	test.run(func(recorder *gpu.Recorder) {
		test.kernels.gatherEmbeddings(recorder, tableBuffer, tokenBuffer, nil, plainOutputs, rows, vectorSize, sequenceLength)
		test.kernels.gatherEmbeddings(recorder, tableBuffer, tokenBuffer, positionBuffer, positionedOutputs, rows, vectorSize, sequenceLength)
	})
	wantPlain := make([]float64, rows*vectorSize)
	wantPositioned := make([]float64, rows*vectorSize)
	for row, tokenID := range tokenIDs {
		for column := range vectorSize {
			value := table[int(tokenID)*vectorSize+column]
			wantPlain[row*vectorSize+column] = value
			wantPositioned[row*vectorSize+column] = value + positionVectors[(row%sequenceLength)*vectorSize+column]
		}
	}
	test.expectClose("gathered rows", test.download(plainOutputs), wantPlain, 1e-6)
	test.expectClose("gathered rows with positions", test.download(positionedOutputs), wantPositioned, 1e-6)
}

func TestScatterEmbeddingGradientsAddsRepeatedTokens(t *testing.T) {
	test := openKernelTest(t)
	vocabularySize := 6
	vectorSize := 4
	rowGradients := randomValues(5*vectorSize, -1, 1)
	startingTableGradients := randomValues(vocabularySize*vectorSize, -1, 1)
	batch := preparedBatch{}
	batch.addEmbeddingGradientLists(map[int][]int{3: {0, 2, 4}, 1: {1}, 5: {3}})
	tableGradients := test.bufferWith(startingTableGradients)
	test.run(func(recorder *gpu.Recorder) {
		test.kernels.scatterEmbeddingGradients(recorder, test.bufferWith(rowGradients), test.bufferWith(batch.uniqueTokenIDs), test.bufferWith(batch.rowStarts), test.bufferWith(batch.sortedRows), tableGradients, batch.numberOfUniqueTokens, vectorSize)
	})
	want := append([]float64(nil), startingTableGradients...)
	tokenOfRow := []int{3, 1, 3, 5, 3}
	for row, tokenID := range tokenOfRow {
		for column := range vectorSize {
			want[tokenID*vectorSize+column] += rowGradients[row*vectorSize+column]
		}
	}
	test.expectClose("table gradients", test.download(tableGradients), want, 1e-5)
}

func TestClipScaleFromSquaredSums(t *testing.T) {
	test := openKernelTest(t)
	first := randomValues(70000, -1, 1)
	second := randomValues(33, -1, 1)
	sumOfSquares := 0.0
	for _, values := range [][]float64{first, second} {
		for _, value := range values {
			sumOfSquares += value * value
		}
	}
	norm := math.Sqrt(sumOfSquares)
	for _, maximumNorm := range []float64{0, norm * 2, norm / 4} {
		partialSums := test.emptyBuffer(squaredSumGroupsFor(len(first)) + squaredSumGroupsFor(len(second)))
		clipResult := test.emptyBuffer(2)
		test.run(func(recorder *gpu.Recorder) {
			numberOfPartials := test.kernels.squaredSumsInto(recorder, test.bufferWith(first), partialSums, 0, len(first))
			numberOfPartials += test.kernels.squaredSumsInto(recorder, test.bufferWith(second), partialSums, numberOfPartials, len(second))
			test.kernels.computeClipScale(recorder, partialSums, clipResult, numberOfPartials, maximumNorm)
		})
		wantScale := 1.0
		if maximumNorm > 0 && norm > maximumNorm {
			wantScale = maximumNorm / norm
		}
		test.expectClose("clip scale and norm", test.download(clipResult), []float64{wantScale, norm}, 1e-4)
	}
}

func TestDropoutKeepsTheExpectedShareAndRepeatsWithTheSameSeed(t *testing.T) {
	test := openKernelTest(t)
	count := 100000
	ones := make([]float64, count)
	for i := range ones {
		ones[i] = 1
	}
	first := test.bufferWith(ones)
	second := test.bufferWith(ones)
	other := test.bufferWith(ones)
	rate := 0.25
	test.run(func(recorder *gpu.Recorder) {
		test.kernels.dropOut(recorder, first, count, 12345, rate)
		test.kernels.dropOut(recorder, second, count, 12345, rate)
		test.kernels.dropOut(recorder, other, count, 999, rate)
	})
	firstValues := test.download(first)
	secondValues := test.download(second)
	otherValues := test.download(other)
	dropped := 0
	differentFromOtherSeed := 0
	for i := range firstValues {
		if firstValues[i] != secondValues[i] {
			t.Fatalf("value %d differs between two runs with the same seed", i)
		}
		if firstValues[i] == 0 {
			dropped++
		} else if math.Abs(firstValues[i]-1/(1-rate)) > 1e-6 {
			t.Fatalf("kept value %d is %v, want %v", i, firstValues[i], 1/(1-rate))
		}
		if (firstValues[i] == 0) != (otherValues[i] == 0) {
			differentFromOtherSeed++
		}
	}
	share := float64(dropped) / float64(count)
	if math.Abs(share-rate) > 0.01 {
		t.Fatalf("dropped %.3f of the values, want about %v", share, rate)
	}
	if differentFromOtherSeed < count/10 {
		t.Fatalf("a different seed changed only %d of %d masks", differentFromOtherSeed, count)
	}
}
