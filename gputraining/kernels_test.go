package gputraining

import (
	"github.com/javanhut/GoTransformers/activationfunction"
	"github.com/javanhut/GoTransformers/gpu"
	"github.com/javanhut/GoTransformers/normalization"
	"github.com/javanhut/GoTransformers/optimizer"
	"github.com/javanhut/GoTransformers/parameter"
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
	"testing"
)

type kernelTest struct {
	t       *testing.T
	device  *gpu.Device
	kernels *kernels
}

func openKernelTest(t *testing.T) *kernelTest {
	t.Helper()
	device, err := gpu.OpenBest()
	if err != nil {
		t.Skipf("no Vulkan GPU available: %v", err)
	}
	loaded, err := loadKernels(device)
	if err != nil {
		device.Close()
		t.Fatal(err)
	}
	t.Cleanup(device.Close)
	return &kernelTest{t: t, device: device, kernels: loaded}
}

func randomValues(count int, lowest float64, highest float64) []float64 {
	values := make([]float64, count)
	for i := range values {
		values[i] = vectormath.RandomNumberBetween(lowest, highest)
	}
	return values
}

func (test *kernelTest) bufferWith(values []float64) *gpu.Buffer {
	test.t.Helper()
	buffer, err := test.device.NewBuffer(len(values))
	if err != nil {
		test.t.Fatal(err)
	}
	if err := buffer.Upload(values); err != nil {
		test.t.Fatal(err)
	}
	return buffer
}

func (test *kernelTest) emptyBuffer(count int) *gpu.Buffer {
	return test.bufferWith(make([]float64, count))
}

func (test *kernelTest) run(record func(recorder *gpu.Recorder)) {
	test.t.Helper()
	recorder, err := test.device.NewRecorder()
	if err != nil {
		test.t.Fatal(err)
	}
	defer recorder.Free()
	recorder.Begin()
	record(recorder)
	if err := recorder.Submit(); err != nil {
		test.t.Fatal(err)
	}
}

func (test *kernelTest) download(buffer *gpu.Buffer) []float64 {
	test.t.Helper()
	values := make([]float64, buffer.NumberOfFloats())
	if err := buffer.Download(values); err != nil {
		test.t.Fatal(err)
	}
	return values
}

func (test *kernelTest) expectClose(name string, got []float64, want []float64, relativeTolerance float64) {
	test.t.Helper()
	if len(got) < len(want) {
		test.t.Fatalf("%s: got %d values, want %d", name, len(got), len(want))
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > relativeTolerance*math.Max(1, math.Abs(want[i])) {
			test.t.Fatalf("%s value %d: GPU %v, CPU %v", name, i, got[i], want[i])
		}
	}
}

func TestMatrixMultiplyShapes(t *testing.T) {
	test := openKernelTest(t)
	rows, inner, columns := 70, 45, 131
	first := randomValues(rows*inner, -1, 1)
	second := randomValues(inner*columns, -1, 1)
	want := make([]float64, rows*columns)
	for row := 0; row < rows; row++ {
		for column := 0; column < columns; column++ {
			for step := 0; step < inner; step++ {
				want[row*columns+column] += first[row*inner+step] * second[step*columns+column]
			}
		}
	}
	firstBuffer := test.bufferWith(first)
	secondBuffer := test.bufferWith(second)

	result := test.emptyBuffer(rows * columns)
	test.run(func(recorder *gpu.Recorder) {
		test.kernels.multiply(recorder, wholeMatrix(firstBuffer, inner), wholeMatrix(secondBuffer, columns), wholeMatrix(result, columns), plainShape(rows, inner, columns))
	})
	test.expectClose("first times second", test.download(result), want, 1e-4)

	transposedSecond := make([]float64, columns*inner)
	transposedFirst := make([]float64, inner*rows)
	for step := 0; step < inner; step++ {
		for column := 0; column < columns; column++ {
			transposedSecond[column*inner+step] = second[step*columns+column]
		}
		for row := 0; row < rows; row++ {
			transposedFirst[step*rows+row] = first[row*inner+step]
		}
	}
	transposedSecondBuffer := test.bufferWith(transposedSecond)
	transposedFirstBuffer := test.bufferWith(transposedFirst)
	resultFromTransposes := test.bufferWith(want)
	test.run(func(recorder *gpu.Recorder) {
		shape := plainShape(rows, inner, columns)
		shape.accumulate = true
		shape.scale = -0.5
		test.kernels.multiply(recorder, wholeMatrix(transposedFirstBuffer, rows).transposed(), wholeMatrix(transposedSecondBuffer, inner).transposed(), wholeMatrix(resultFromTransposes, columns), shape)
	})
	halfOfWant := make([]float64, len(want))
	for i := range want {
		halfOfWant[i] = 0.5 * want[i]
	}
	test.expectClose("accumulate scaled first transposed times second transposed", test.download(resultFromTransposes), halfOfWant, 1e-4)
}

func TestBatchedHeadMultiplyWithGroupedKeys(t *testing.T) {
	test := openKernelTest(t)
	sequences, length, heads, keyValueHeads, headSize := 2, 5, 4, 2, 3
	group := heads / keyValueHeads
	queries := randomValues(sequences*length*heads*headSize, -1, 1)
	keys := randomValues(sequences*length*keyValueHeads*headSize, -1, 1)
	want := make([]float64, sequences*heads*length*length)
	for sequence := 0; sequence < sequences; sequence++ {
		for head := 0; head < heads; head++ {
			for query := 0; query < length; query++ {
				for key := 0; key < length; key++ {
					sum := 0.0
					for dimension := 0; dimension < headSize; dimension++ {
						queryValue := queries[(sequence*length+query)*heads*headSize+head*headSize+dimension]
						keyValue := keys[(sequence*length+key)*keyValueHeads*headSize+(head/group)*headSize+dimension]
						sum += queryValue * keyValue
					}
					want[((sequence*heads+head)*length+query)*length+key] = 0.25 * sum
				}
			}
		}
	}
	queryBuffer := test.bufferWith(queries)
	keyBuffer := test.bufferWith(keys)
	scores := test.emptyBuffer(len(want))
	test.run(func(recorder *gpu.Recorder) {
		queryView := matrixView{buffer: queryBuffer, rowStride: heads * headSize, columnStride: 1, outerStride: length * heads * headSize, innerStride: headSize, innerDivisor: 1}
		keyView := matrixView{buffer: keyBuffer, rowStride: 1, columnStride: keyValueHeads * headSize, outerStride: length * keyValueHeads * headSize, innerStride: headSize, innerDivisor: group}
		scoreView := matrixView{buffer: scores, rowStride: length, columnStride: 1, outerStride: heads * length * length, innerStride: length * length, innerDivisor: 1}
		shape := multiplyShape{rows: length, inner: headSize, columns: length, batchCount: sequences * heads, innerBatchCount: heads, scale: 0.25}
		test.kernels.multiply(recorder, queryView, keyView, scoreView, shape)
	})
	test.expectClose("per-head scores", test.download(scores), want, 1e-5)
}

func TestAddBiasAndColumnSums(t *testing.T) {
	test := openKernelTest(t)
	rows, columns := 33, 300
	matrix := randomValues(rows*columns, -1, 1)
	biases := randomValues(columns, -1, 1)
	other := randomValues(rows*columns, -1, 1)
	wantBiased := make([]float64, len(matrix))
	wantSums := randomValues(columns, -1, 1)
	startingSums := append([]float64(nil), wantSums...)
	wantAdded := make([]float64, len(matrix))
	for row := 0; row < rows; row++ {
		for column := 0; column < columns; column++ {
			index := row*columns + column
			wantBiased[index] = matrix[index] + biases[column]
			wantSums[column] += matrix[index]
			wantAdded[index] = matrix[index] + other[index]
		}
	}
	matrixBuffer := test.bufferWith(matrix)
	biasBuffer := test.bufferWith(biases)
	otherBuffer := test.bufferWith(other)
	sumBuffer := test.bufferWith(startingSums)
	addedBuffer := test.emptyBuffer(len(matrix))
	test.run(func(recorder *gpu.Recorder) {
		test.kernels.columnSumsInto(recorder, matrixBuffer, sumBuffer, rows, columns, true)
		test.kernels.addBuffers(recorder, matrixBuffer, otherBuffer, addedBuffer, len(matrix))
		test.kernels.addBias(recorder, biasBuffer, matrixBuffer, rows, columns)
	})
	test.expectClose("column sums", test.download(sumBuffer), wantSums, 1e-5)
	test.expectClose("added", test.download(addedBuffer), wantAdded, 1e-6)
	test.expectClose("biased", test.download(matrixBuffer), wantBiased, 1e-6)
}

func TestNormalizeMatchesRMSNorm(t *testing.T) {
	test := openKernelTest(t)
	rows, size := 7, 300
	norm := normalization.NewRMSNorm("norm", size)
	for i := range norm.Weights {
		norm.Weights[i] = vectormath.RandomNumberBetween(0.5, 1.5)
	}
	inputs := vectormath.NewRandomMatrix(rows, size, -2, 2)
	outputGradients := vectormath.NewRandomMatrix(rows, size, -1, 1)
	residual := randomValues(rows*size, -1, 1)
	wantOutputs := norm.Forward(inputs)
	wantInputGradients := norm.Backward(outputGradients)
	for i := range wantInputGradients.Values {
		wantInputGradients.Values[i] += residual[i]
	}

	inputBuffer := test.bufferWith(inputs.Values)
	weightBuffer := test.bufferWith(norm.Weights)
	outputBuffer := test.emptyBuffer(rows * size)
	rootMeanSquareBuffer := test.emptyBuffer(rows)
	outputGradientBuffer := test.bufferWith(outputGradients.Values)
	residualBuffer := test.bufferWith(residual)
	inputGradientBuffer := test.emptyBuffer(rows * size)
	weightGradientBuffer := test.emptyBuffer(size)
	test.run(func(recorder *gpu.Recorder) {
		test.kernels.normalizeForward(recorder, inputBuffer, weightBuffer, outputBuffer, rootMeanSquareBuffer, rows, size, norm.Epsilon)
		test.kernels.normalizeBackward(recorder, inputBuffer, weightBuffer, rootMeanSquareBuffer, outputGradientBuffer, residualBuffer, inputGradientBuffer, rows, size)
		test.kernels.normalizeWeightGradients(recorder, inputBuffer, rootMeanSquareBuffer, outputGradientBuffer, weightGradientBuffer, rows, size, true)
	})
	test.expectClose("normalized", test.download(outputBuffer), wantOutputs.Values, 1e-5)
	test.expectClose("input gradients", test.download(inputGradientBuffer), wantInputGradients.Values, 1e-4)
	test.expectClose("weight gradients", test.download(weightGradientBuffer), norm.WeightGradients, 1e-4)
}

func referenceRotate(values []float64, rows int, sequenceLength int, setup rotarySetup, direction float64) []float64 {
	rotated := append([]float64(nil), values...)
	half := setup.rotaryDimensions / 2
	for row := 0; row < rows; row++ {
		position := row % sequenceLength
		for head := 0; head < setup.numberOfHeads; head++ {
			firstRotated := row*setup.numberOfHeads*setup.headSize + head*setup.headSize + setup.headSize - setup.rotaryDimensions
			for pair := 0; pair < half; pair++ {
				frequency := math.Pow(setup.base, -float64(2*pair)/float64(setup.rotaryDimensions))
				angle := float64(position) * frequency * direction
				firstIndex := firstRotated + 2*pair
				secondIndex := firstIndex + 1
				if setup.rotateHalves {
					firstIndex = firstRotated + pair
					secondIndex = firstIndex + half
				}
				first := rotated[firstIndex]
				second := rotated[secondIndex]
				rotated[firstIndex] = first*math.Cos(angle) - second*math.Sin(angle)
				rotated[secondIndex] = first*math.Sin(angle) + second*math.Cos(angle)
			}
		}
	}
	return rotated
}

func TestRotaryBothLayouts(t *testing.T) {
	test := openKernelTest(t)
	rows, sequenceLength := 24, 12
	for _, setup := range []rotarySetup{
		{numberOfHeads: 3, headSize: 8, rotaryDimensions: 8, base: 10000},
		{numberOfHeads: 2, headSize: 8, rotaryDimensions: 4, rotateHalves: true, base: 500},
	} {
		if err := makeRotaryTables(test.device, &setup, sequenceLength); err != nil {
			t.Fatal(err)
		}
		values := randomValues(rows*setup.numberOfHeads*setup.headSize, -1, 1)
		for _, direction := range []float64{rotateForward, rotateBackward} {
			buffer := test.bufferWith(values)
			test.run(func(recorder *gpu.Recorder) {
				test.kernels.rotate(recorder, buffer, rows, sequenceLength, setup, direction)
			})
			test.expectClose("rotated", test.download(buffer), referenceRotate(values, rows, sequenceLength, setup, direction), 2e-5)
		}
	}
}

func TestCausalSoftmaxAndBackward(t *testing.T) {
	test := openKernelTest(t)
	rows, length := 10, 300
	scores := randomValues(rows*length, -3, 3)
	gradients := randomValues(rows*length, -1, 1)
	wantProbabilities := make([]float64, len(scores))
	wantScoreGradients := make([]float64, len(scores))
	for row := 0; row < rows; row++ {
		query := row % length
		visible := scores[row*length : row*length+query+1]
		probabilities := activationfunction.Softmax(visible)
		scoreGradients := activationfunction.SoftmaxBackward(probabilities, gradients[row*length:row*length+query+1])
		copy(wantProbabilities[row*length:], probabilities)
		copy(wantScoreGradients[row*length:], scoreGradients)
	}
	scoreBuffer := test.bufferWith(scores)
	gradientBuffer := test.bufferWith(gradients)
	test.run(func(recorder *gpu.Recorder) {
		test.kernels.causalSoftmaxRows(recorder, scoreBuffer, rows, length)
		test.kernels.softmaxBackwardRows(recorder, scoreBuffer, gradientBuffer, rows, length)
	})
	test.expectClose("probabilities", test.download(scoreBuffer), wantProbabilities, 1e-5)
	test.expectClose("score gradients", test.download(gradientBuffer), wantScoreGradients, 1e-5)
}

func TestGatedForwardAndBackward(t *testing.T) {
	test := openKernelTest(t)
	count := 1000
	gate := randomValues(count, -4, 4)
	up := randomValues(count, -4, 4)
	hiddenGradients := randomValues(count, -1, 1)
	for _, limit := range []float64{0, 1.5} {
		gateActivation := activationfunction.SiLU
		upActivation := activationfunction.Linear
		if limit > 0 {
			gateActivation = activationfunction.ClampedSiLU(limit)
			upActivation = activationfunction.ClampedLinear(limit)
		}
		wantHidden := make([]float64, count)
		wantGateGradients := make([]float64, count)
		wantUpGradients := make([]float64, count)
		for i := range gate {
			gateOutput := gateActivation.Forward(gate[i])
			upOutput := upActivation.Forward(up[i])
			wantHidden[i] = gateOutput * upOutput
			wantGateGradients[i] = hiddenGradients[i] * upOutput * gateActivation.Derivative(gate[i])
			wantUpGradients[i] = hiddenGradients[i] * gateOutput * upActivation.Derivative(up[i])
		}
		gateBuffer := test.bufferWith(gate)
		upBuffer := test.bufferWith(up)
		hiddenGradientBuffer := test.bufferWith(hiddenGradients)
		hiddenBuffer := test.emptyBuffer(count)
		gateGradientBuffer := test.emptyBuffer(count)
		upGradientBuffer := test.emptyBuffer(count)
		test.run(func(recorder *gpu.Recorder) {
			test.kernels.gatedForward(recorder, gateBuffer, upBuffer, hiddenBuffer, count, limit)
			test.kernels.gatedBackward(recorder, gateBuffer, upBuffer, hiddenGradientBuffer, gateGradientBuffer, upGradientBuffer, count, limit)
		})
		test.expectClose("hidden", test.download(hiddenBuffer), wantHidden, 1e-5)
		test.expectClose("gate gradients", test.download(gateGradientBuffer), wantGateGradients, 1e-5)
		test.expectClose("up gradients", test.download(upGradientBuffer), wantUpGradients, 1e-5)
	}
}

func TestCrossEntropyRows(t *testing.T) {
	test := openKernelTest(t)
	rows, vocabularySize := 6, 1000
	scores := randomValues(rows*vocabularySize, -5, 5)
	targets := []float64{3, 999, 0, 500, 7, 42}
	weights := []float64{0.25, 0, 0.25, 0.5, 0, 1}
	wantGradients := make([]float64, len(scores))
	wantLosses := make([]float64, rows)
	for row := 0; row < rows; row++ {
		probabilities := activationfunction.Softmax(scores[row*vocabularySize : (row+1)*vocabularySize])
		for column, probability := range probabilities {
			gradient := probability
			if column == int(targets[row]) {
				gradient -= 1
			}
			wantGradients[row*vocabularySize+column] = weights[row] * gradient
		}
		if weights[row] > 0 {
			wantLosses[row] = -weights[row] * math.Log(probabilities[int(targets[row])])
		}
	}
	scoreBuffer := test.bufferWith(scores)
	targetBuffer := test.bufferWith(targets)
	weightBuffer := test.bufferWith(weights)
	gradientBuffer := test.emptyBuffer(len(scores))
	lossBuffer := test.emptyBuffer(rows)
	test.run(func(recorder *gpu.Recorder) {
		test.kernels.crossEntropyRows(recorder, scoreBuffer, targetBuffer, weightBuffer, gradientBuffer, lossBuffer, rows, vocabularySize)
	})
	test.expectClose("score gradients", test.download(gradientBuffer), wantGradients, 1e-6)
	test.expectClose("row losses", test.download(lossBuffer), wantLosses, 1e-5)
}

func TestAdamWMatchesOptimizer(t *testing.T) {
	test := openKernelTest(t)
	count := 500
	values := randomValues(count, -1, 1)
	cpuValues := append([]float64(nil), values...)
	matrix := parameter.WithGradients("weights", cpuValues)
	matrix.Rows = 20
	matrix.Columns = 25
	adamW := optimizer.NewAdamW(0.01, 0.1)

	valueBuffer := test.bufferWith(values)
	gradientBuffer := test.emptyBuffer(count)
	averageBuffer := test.emptyBuffer(count)
	squaredBuffer := test.emptyBuffer(count)
	for step := 1; step <= 3; step++ {
		gradients := randomValues(count, -1, 1)
		copy(matrix.Gradients(), gradients)
		adamW.Update([]parameter.Parameter{matrix})
		if err := gradientBuffer.Upload(gradients); err != nil {
			t.Fatal(err)
		}
		test.run(func(recorder *gpu.Recorder) {
			stepSettings := adamWStep{
				learningRate:                     adamW.LearningRate,
				momentumDecay:                    adamW.MomentumDecay,
				squaredGradientDecay:             adamW.SquaredGradientDecay,
				epsilon:                          adamW.Epsilon,
				averageGradientCorrection:        1 - math.Pow(adamW.MomentumDecay, float64(step)),
				averageSquaredGradientCorrection: 1 - math.Pow(adamW.SquaredGradientDecay, float64(step)),
			}
			test.kernels.adamWUpdate(recorder, valueBuffer, gradientBuffer, averageBuffer, squaredBuffer, count, stepSettings, 1-adamW.LearningRate*adamW.WeightDecay)
		})
	}
	test.expectClose("values after 3 AdamW steps", test.download(valueBuffer), cpuValues, 1e-5)
}

func TestSumHeadGroups(t *testing.T) {
	test := openKernelTest(t)
	rows, keyValueHeads, groupSize, headSize := 5, 2, 3, 4
	perQueryHead := randomValues(rows*keyValueHeads*groupSize*headSize, -1, 1)
	want := make([]float64, rows*keyValueHeads*headSize)
	for row := 0; row < rows; row++ {
		for queryHead := 0; queryHead < keyValueHeads*groupSize; queryHead++ {
			for dimension := 0; dimension < headSize; dimension++ {
				want[row*keyValueHeads*headSize+(queryHead/groupSize)*headSize+dimension] += perQueryHead[row*keyValueHeads*groupSize*headSize+queryHead*headSize+dimension]
			}
		}
	}
	inputBuffer := test.bufferWith(perQueryHead)
	outputBuffer := test.emptyBuffer(len(want))
	test.run(func(recorder *gpu.Recorder) {
		test.kernels.sumHeadGroups(recorder, inputBuffer, outputBuffer, rows, keyValueHeads, groupSize, headSize)
	})
	test.expectClose("grouped sums", test.download(outputBuffer), want, 1e-6)
}
