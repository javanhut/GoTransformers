package gputraining

import (
	_ "embed"
	"fmt"
	"github.com/javanhut/GoTransformers/gpu"
	"math"
)

//go:generate glslc --target-env=vulkan1.0 -O shaders/matrixmultiply.comp -o shaders/matrixmultiply.spv
//go:generate glslc --target-env=vulkan1.0 -O shaders/add.comp -o shaders/add.spv
//go:generate glslc --target-env=vulkan1.0 -O shaders/biasadd.comp -o shaders/biasadd.spv
//go:generate glslc --target-env=vulkan1.0 -O shaders/columnsums.comp -o shaders/columnsums.spv
//go:generate glslc --target-env=vulkan1.0 -O shaders/rmsnormforward.comp -o shaders/rmsnormforward.spv
//go:generate glslc --target-env=vulkan1.0 -O shaders/rmsnormbackward.comp -o shaders/rmsnormbackward.spv
//go:generate glslc --target-env=vulkan1.0 -O shaders/rmsnormweightgradient.comp -o shaders/rmsnormweightgradient.spv
//go:generate glslc --target-env=vulkan1.0 -O shaders/rotary.comp -o shaders/rotary.spv
//go:generate glslc --target-env=vulkan1.0 -O shaders/causalsoftmax.comp -o shaders/causalsoftmax.spv
//go:generate glslc --target-env=vulkan1.0 -O shaders/softmaxbackward.comp -o shaders/softmaxbackward.spv
//go:generate glslc --target-env=vulkan1.0 -O shaders/swigluforward.comp -o shaders/swigluforward.spv
//go:generate glslc --target-env=vulkan1.0 -O shaders/swiglubackward.comp -o shaders/swiglubackward.spv
//go:generate glslc --target-env=vulkan1.0 -O shaders/crossentropy.comp -o shaders/crossentropy.spv
//go:generate glslc --target-env=vulkan1.0 -O shaders/adamw.comp -o shaders/adamw.spv
//go:generate glslc --target-env=vulkan1.0 -O shaders/sumgroups.comp -o shaders/sumgroups.spv
//go:generate glslc --target-env=vulkan1.0 -O shaders/embeddinggather.comp -o shaders/embeddinggather.spv
//go:generate glslc --target-env=vulkan1.0 -O shaders/embeddingscatter.comp -o shaders/embeddingscatter.spv
//go:generate glslc --target-env=vulkan1.0 -O shaders/squaredsums.comp -o shaders/squaredsums.spv
//go:generate glslc --target-env=vulkan1.0 -O shaders/clipscale.comp -o shaders/clipscale.spv
//go:generate glslc --target-env=vulkan1.0 -O shaders/dropout.comp -o shaders/dropout.spv

//go:embed shaders/matrixmultiply.spv
var matrixMultiplyShader []byte

//go:embed shaders/add.spv
var addShader []byte

//go:embed shaders/biasadd.spv
var biasAddShader []byte

//go:embed shaders/columnsums.spv
var columnSumsShader []byte

//go:embed shaders/rmsnormforward.spv
var rmsNormForwardShader []byte

//go:embed shaders/rmsnormbackward.spv
var rmsNormBackwardShader []byte

//go:embed shaders/rmsnormweightgradient.spv
var rmsNormWeightGradientShader []byte

//go:embed shaders/rotary.spv
var rotaryShader []byte

//go:embed shaders/causalsoftmax.spv
var causalSoftmaxShader []byte

//go:embed shaders/softmaxbackward.spv
var softmaxBackwardShader []byte

//go:embed shaders/swigluforward.spv
var swigluForwardShader []byte

//go:embed shaders/swiglubackward.spv
var swigluBackwardShader []byte

//go:embed shaders/crossentropy.spv
var crossEntropyShader []byte

//go:embed shaders/adamw.spv
var adamWShader []byte

//go:embed shaders/sumgroups.spv
var sumGroupsShader []byte

//go:embed shaders/embeddinggather.spv
var embeddingGatherShader []byte

//go:embed shaders/embeddingscatter.spv
var embeddingScatterShader []byte

//go:embed shaders/squaredsums.spv
var squaredSumsShader []byte

//go:embed shaders/clipscale.spv
var clipScaleShader []byte

//go:embed shaders/dropout.spv
var dropoutShader []byte

type kernels struct {
	matrixMultiply        *gpu.Program
	add                   *gpu.Program
	biasAdd               *gpu.Program
	columnSums            *gpu.Program
	rmsNormForward        *gpu.Program
	rmsNormBackward       *gpu.Program
	rmsNormWeightGradient *gpu.Program
	rotary                *gpu.Program
	causalSoftmax         *gpu.Program
	softmaxBackward       *gpu.Program
	swigluForward         *gpu.Program
	swigluBackward        *gpu.Program
	crossEntropy          *gpu.Program
	adamW                 *gpu.Program
	sumGroups             *gpu.Program
	embeddingGather       *gpu.Program
	embeddingScatter      *gpu.Program
	squaredSums           *gpu.Program
	clipScale             *gpu.Program
	dropout               *gpu.Program
}

func loadKernels(device *gpu.Device) (*kernels, error) {
	loaded := &kernels{}
	programs := []struct {
		target            **gpu.Program
		name              string
		shader            []byte
		numberOfBuffers   int
		pushConstantWords int
	}{
		{&loaded.matrixMultiply, "matrixMultiply", matrixMultiplyShader, 3, 23},
		{&loaded.add, "add", addShader, 3, 1},
		{&loaded.biasAdd, "biasAdd", biasAddShader, 2, 2},
		{&loaded.columnSums, "columnSums", columnSumsShader, 2, 3},
		{&loaded.rmsNormForward, "rmsNormForward", rmsNormForwardShader, 4, 3},
		{&loaded.rmsNormBackward, "rmsNormBackward", rmsNormBackwardShader, 6, 3},
		{&loaded.rmsNormWeightGradient, "rmsNormWeightGradient", rmsNormWeightGradientShader, 4, 3},
		{&loaded.rotary, "rotary", rotaryShader, 3, 7},
		{&loaded.causalSoftmax, "causalSoftmax", causalSoftmaxShader, 1, 2},
		{&loaded.softmaxBackward, "softmaxBackward", softmaxBackwardShader, 2, 2},
		{&loaded.swigluForward, "swigluForward", swigluForwardShader, 3, 2},
		{&loaded.swigluBackward, "swigluBackward", swigluBackwardShader, 5, 2},
		{&loaded.crossEntropy, "crossEntropy", crossEntropyShader, 5, 2},
		{&loaded.adamW, "adamW", adamWShader, 5, 8},
		{&loaded.sumGroups, "sumGroups", sumGroupsShader, 2, 4},
		{&loaded.embeddingGather, "embeddingGather", embeddingGatherShader, 4, 4},
		{&loaded.embeddingScatter, "embeddingScatter", embeddingScatterShader, 5, 2},
		{&loaded.squaredSums, "squaredSums", squaredSumsShader, 2, 3},
		{&loaded.clipScale, "clipScale", clipScaleShader, 2, 2},
		{&loaded.dropout, "dropout", dropoutShader, 1, 4},
	}
	for _, program := range programs {
		created, err := device.NewProgram(program.name, program.shader, program.numberOfBuffers, program.pushConstantWords)
		if err != nil {
			loaded.free()
			return nil, fmt.Errorf("loading GPU kernel %q: %w", program.name, err)
		}
		*program.target = created
	}
	return loaded, nil
}

func (loaded *kernels) free() {
	for _, program := range []*gpu.Program{
		loaded.matrixMultiply, loaded.add, loaded.biasAdd, loaded.columnSums,
		loaded.rmsNormForward, loaded.rmsNormBackward, loaded.rmsNormWeightGradient,
		loaded.rotary, loaded.causalSoftmax, loaded.softmaxBackward,
		loaded.swigluForward, loaded.swigluBackward, loaded.crossEntropy, loaded.adamW, loaded.sumGroups,
		loaded.embeddingGather, loaded.embeddingScatter, loaded.squaredSums, loaded.clipScale, loaded.dropout,
	} {
		if program != nil {
			program.Free()
		}
	}
}

type matrixView struct {
	buffer       *gpu.Buffer
	offset       int
	rowStride    int
	columnStride int
	outerStride  int
	innerStride  int
	innerDivisor int
}

func wholeMatrix(buffer *gpu.Buffer, columns int) matrixView {
	return matrixView{buffer: buffer, rowStride: columns, columnStride: 1, innerDivisor: 1}
}

func (view matrixView) transposed() matrixView {
	view.rowStride, view.columnStride = view.columnStride, view.rowStride
	return view
}

type multiplyShape struct {
	rows            int
	inner           int
	columns         int
	batchCount      int
	innerBatchCount int
	accumulate      bool
	scale           float64
}

func plainShape(rows int, inner int, columns int) multiplyShape {
	return multiplyShape{rows: rows, inner: inner, columns: columns, batchCount: 1, innerBatchCount: 1, scale: 1}
}

func boolWord(value bool) uint32 {
	if value {
		return 1
	}
	return 0
}

func (loaded *kernels) multiply(recorder *gpu.Recorder, first matrixView, second matrixView, result matrixView, shape multiplyShape) {
	if shape.batchCount < 1 || shape.innerBatchCount < 1 {
		panic(fmt.Sprintf("multiply: batch counts must be at least 1, got %d and %d", shape.batchCount, shape.innerBatchCount))
	}
	divisor := func(view matrixView) uint32 {
		if view.innerDivisor < 1 {
			return 1
		}
		return uint32(view.innerDivisor)
	}
	words := []uint32{
		uint32(shape.rows), uint32(shape.inner), uint32(shape.columns), uint32(shape.innerBatchCount),
		uint32(first.offset), uint32(first.rowStride), uint32(first.columnStride), uint32(first.outerStride), uint32(first.innerStride), divisor(first),
		uint32(second.offset), uint32(second.rowStride), uint32(second.columnStride), uint32(second.outerStride), uint32(second.innerStride), divisor(second),
		uint32(result.offset), uint32(result.rowStride), uint32(result.columnStride), uint32(result.outerStride), uint32(result.innerStride),
		boolWord(shape.accumulate), gpu.Float(shape.scale),
	}
	groupsAcross := uint32((shape.columns + multiplyTileColumns - 1) / multiplyTileColumns)
	groupsDown := uint32((shape.rows + multiplyTileRows - 1) / multiplyTileRows)
	recorder.Run(loaded.matrixMultiply, groupsAcross, groupsDown, uint32(shape.batchCount), words, first.buffer, second.buffer, result.buffer)
}

const multiplyTileRows = 128
const multiplyTileColumns = 64

func runElementwise(recorder *gpu.Recorder, program *gpu.Program, count int, words []uint32, buffers ...*gpu.Buffer) {
	groupsAcross, groupsDown := gpu.GroupsFor(count, 256)
	recorder.Run(program, groupsAcross, groupsDown, 1, words, buffers...)
}

func runInGroupsOf(recorder *gpu.Recorder, program *gpu.Program, count int, perGroup int, words []uint32, buffers ...*gpu.Buffer) {
	groupsAcross, groupsDown := gpu.GroupsFor((count+perGroup-1)/perGroup, 1)
	recorder.Run(program, groupsAcross, groupsDown, 1, words, buffers...)
}

func runPerRow(recorder *gpu.Recorder, program *gpu.Program, rows int, words []uint32, buffers ...*gpu.Buffer) {
	groupsAcross, groupsDown := gpu.GroupsFor(rows, 1)
	recorder.Run(program, groupsAcross, groupsDown, 1, words, buffers...)
}

func (loaded *kernels) addBuffers(recorder *gpu.Recorder, first *gpu.Buffer, second *gpu.Buffer, result *gpu.Buffer, count int) {
	runElementwise(recorder, loaded.add, count, []uint32{uint32(count)}, first, second, result)
}

func (loaded *kernels) addBias(recorder *gpu.Recorder, biases *gpu.Buffer, matrix *gpu.Buffer, rows int, columns int) {
	runElementwise(recorder, loaded.biasAdd, rows*columns, []uint32{uint32(rows), uint32(columns)}, biases, matrix)
}

func (loaded *kernels) columnSumsInto(recorder *gpu.Recorder, matrix *gpu.Buffer, target *gpu.Buffer, rows int, columns int, accumulate bool) {
	runInGroupsOf(recorder, loaded.columnSums, columns, 16, []uint32{uint32(rows), uint32(columns), boolWord(accumulate)}, matrix, target)
}

func (loaded *kernels) normalizeForward(recorder *gpu.Recorder, inputs *gpu.Buffer, weights *gpu.Buffer, outputs *gpu.Buffer, rootMeanSquares *gpu.Buffer, rows int, size int, epsilon float64) {
	runPerRow(recorder, loaded.rmsNormForward, rows, []uint32{uint32(rows), uint32(size), gpu.Float(epsilon)}, inputs, weights, outputs, rootMeanSquares)
}

func (loaded *kernels) normalizeBackward(recorder *gpu.Recorder, inputs *gpu.Buffer, weights *gpu.Buffer, rootMeanSquares *gpu.Buffer, outputGradients *gpu.Buffer, residualGradients *gpu.Buffer, inputGradients *gpu.Buffer, rows int, size int) {
	hasResidual := residualGradients != nil
	if !hasResidual {
		residualGradients = outputGradients
	}
	runPerRow(recorder, loaded.rmsNormBackward, rows, []uint32{uint32(rows), uint32(size), boolWord(hasResidual)}, inputs, weights, rootMeanSquares, outputGradients, residualGradients, inputGradients)
}

func (loaded *kernels) normalizeWeightGradients(recorder *gpu.Recorder, inputs *gpu.Buffer, rootMeanSquares *gpu.Buffer, outputGradients *gpu.Buffer, weightGradients *gpu.Buffer, rows int, size int, accumulate bool) {
	runInGroupsOf(recorder, loaded.rmsNormWeightGradient, size, 16, []uint32{uint32(rows), uint32(size), boolWord(accumulate)}, inputs, rootMeanSquares, outputGradients, weightGradients)
}

type rotarySetup struct {
	numberOfHeads    int
	headSize         int
	rotaryDimensions int
	rotateHalves     bool
	base             float64
	cosines          *gpu.Buffer
	sines            *gpu.Buffer
	longestSequence  int
}

const rotateForward = 1.0

const rotateBackward = -1.0

func makeRotaryTables(device *gpu.Device, setup *rotarySetup, longestSequence int) error {
	pairsPerHead := setup.rotaryDimensions / 2
	cosines := make([]float64, longestSequence*pairsPerHead)
	sines := make([]float64, longestSequence*pairsPerHead)
	for position := range longestSequence {
		for pair := range pairsPerHead {
			frequency := math.Pow(setup.base, -float64(2*pair)/float64(setup.rotaryDimensions))
			angle := float64(position) * frequency
			cosines[position*pairsPerHead+pair] = math.Cos(angle)
			sines[position*pairsPerHead+pair] = math.Sin(angle)
		}
	}
	cosineBuffer, err := device.NewBuffer(len(cosines))
	if err != nil {
		return err
	}
	sineBuffer, err := device.NewBuffer(len(sines))
	if err != nil {
		cosineBuffer.Free()
		return err
	}
	if err := cosineBuffer.Upload(cosines); err != nil {
		return err
	}
	if err := sineBuffer.Upload(sines); err != nil {
		return err
	}
	if setup.cosines != nil {
		setup.cosines.Free()
		setup.sines.Free()
	}
	setup.cosines = cosineBuffer
	setup.sines = sineBuffer
	setup.longestSequence = longestSequence
	return nil
}

func (loaded *kernels) rotate(recorder *gpu.Recorder, values *gpu.Buffer, rows int, sequenceLength int, setup rotarySetup, direction float64) {
	if sequenceLength > setup.longestSequence {
		panic(fmt.Sprintf("rotate: sequences of %d tokens are longer than the %d the rotary tables were made for", sequenceLength, setup.longestSequence))
	}
	count := rows * setup.numberOfHeads * (setup.rotaryDimensions / 2)
	words := []uint32{
		uint32(rows), uint32(setup.numberOfHeads), uint32(setup.headSize), uint32(setup.rotaryDimensions),
		uint32(sequenceLength), boolWord(setup.rotateHalves), gpu.Float(direction),
	}
	runElementwise(recorder, loaded.rotary, count, words, values, setup.cosines, setup.sines)
}

func (loaded *kernels) causalSoftmaxRows(recorder *gpu.Recorder, scores *gpu.Buffer, rows int, sequenceLength int) {
	runInGroupsOf(recorder, loaded.causalSoftmax, rows, 8, []uint32{uint32(rows), uint32(sequenceLength)}, scores)
}

func (loaded *kernels) softmaxBackwardRows(recorder *gpu.Recorder, probabilities *gpu.Buffer, gradients *gpu.Buffer, rows int, sequenceLength int) {
	runInGroupsOf(recorder, loaded.softmaxBackward, rows, 8, []uint32{uint32(rows), uint32(sequenceLength)}, probabilities, gradients)
}

func (loaded *kernels) gatedForward(recorder *gpu.Recorder, gate *gpu.Buffer, up *gpu.Buffer, hidden *gpu.Buffer, count int, clampLimit float64) {
	runElementwise(recorder, loaded.swigluForward, count, []uint32{uint32(count), gpu.Float(clampLimit)}, gate, up, hidden)
}

func (loaded *kernels) gatedBackward(recorder *gpu.Recorder, gate *gpu.Buffer, up *gpu.Buffer, hiddenGradients *gpu.Buffer, gateGradients *gpu.Buffer, upGradients *gpu.Buffer, count int, clampLimit float64) {
	runElementwise(recorder, loaded.swigluBackward, count, []uint32{uint32(count), gpu.Float(clampLimit)}, gate, up, hiddenGradients, gateGradients, upGradients)
}

func (loaded *kernels) crossEntropyRows(recorder *gpu.Recorder, scores *gpu.Buffer, targets *gpu.Buffer, rowWeights *gpu.Buffer, scoreGradients *gpu.Buffer, rowLosses *gpu.Buffer, rows int, vocabularySize int) {
	runPerRow(recorder, loaded.crossEntropy, rows, []uint32{uint32(rows), uint32(vocabularySize)}, scores, targets, rowWeights, scoreGradients, rowLosses)
}

type adamWStep struct {
	learningRate                     float64
	momentumDecay                    float64
	squaredGradientDecay             float64
	epsilon                          float64
	averageGradientCorrection        float64
	averageSquaredGradientCorrection float64
}

func (loaded *kernels) adamWUpdate(recorder *gpu.Recorder, values *gpu.Buffer, gradients *gpu.Buffer, averageGradients *gpu.Buffer, averageSquaredGradients *gpu.Buffer, clipResult *gpu.Buffer, count int, step adamWStep, shrinkFactor float64) {
	words := []uint32{
		uint32(count), gpu.Float(step.learningRate), gpu.Float(step.momentumDecay), gpu.Float(step.squaredGradientDecay),
		gpu.Float(step.epsilon), gpu.Float(step.averageGradientCorrection), gpu.Float(step.averageSquaredGradientCorrection), gpu.Float(shrinkFactor),
	}
	runElementwise(recorder, loaded.adamW, count, words, values, gradients, averageGradients, averageSquaredGradients, clipResult)
}

func (loaded *kernels) sumHeadGroups(recorder *gpu.Recorder, perQueryHead *gpu.Buffer, perKeyValueHead *gpu.Buffer, rows int, keyValueHeads int, groupSize int, headSize int) {
	count := rows * keyValueHeads * headSize
	runElementwise(recorder, loaded.sumGroups, count, []uint32{uint32(rows), uint32(keyValueHeads), uint32(groupSize), uint32(headSize)}, perQueryHead, perKeyValueHead)
}

func (loaded *kernels) gatherEmbeddings(recorder *gpu.Recorder, table *gpu.Buffer, tokenIDs *gpu.Buffer, positionVectors *gpu.Buffer, outputs *gpu.Buffer, rows int, vectorSize int, sequenceLength int) {
	addPositions := positionVectors != nil
	if !addPositions {
		positionVectors = table
	}
	words := []uint32{uint32(rows), uint32(vectorSize), uint32(sequenceLength), boolWord(addPositions)}
	runElementwise(recorder, loaded.embeddingGather, rows*vectorSize, words, table, tokenIDs, positionVectors, outputs)
}

func (loaded *kernels) scatterEmbeddingGradients(recorder *gpu.Recorder, rowGradients *gpu.Buffer, uniqueTokenIDs *gpu.Buffer, rowStarts *gpu.Buffer, sortedRows *gpu.Buffer, tableGradients *gpu.Buffer, numberOfUniqueTokens int, vectorSize int) {
	if numberOfUniqueTokens == 0 {
		return
	}
	words := []uint32{uint32(numberOfUniqueTokens), uint32(vectorSize)}
	runElementwise(recorder, loaded.embeddingScatter, numberOfUniqueTokens*vectorSize, words, rowGradients, uniqueTokenIDs, rowStarts, sortedRows, tableGradients)
}

const squaredSumGroupsPerBuffer = 64

func squaredSumGroupsFor(count int) int {
	return max(1, min(squaredSumGroupsPerBuffer, (count+255)/256))
}

func (loaded *kernels) squaredSumsInto(recorder *gpu.Recorder, values *gpu.Buffer, partialSums *gpu.Buffer, firstPartial int, count int) int {
	numberOfGroups := squaredSumGroupsFor(count)
	words := []uint32{uint32(count), uint32(firstPartial), uint32(numberOfGroups)}
	recorder.Run(loaded.squaredSums, uint32(numberOfGroups), 1, 1, words, values, partialSums)
	return numberOfGroups
}

func (loaded *kernels) computeClipScale(recorder *gpu.Recorder, partialSums *gpu.Buffer, clipResult *gpu.Buffer, numberOfPartials int, maximumNorm float64) {
	recorder.Run(loaded.clipScale, 1, 1, 1, []uint32{uint32(numberOfPartials), gpu.Float(maximumNorm)}, partialSums, clipResult)
}

func (loaded *kernels) dropOut(recorder *gpu.Recorder, values *gpu.Buffer, count int, seed uint32, rate float64) {
	if rate <= 0 {
		return
	}
	words := []uint32{uint32(count), seed, gpu.Float(rate), gpu.Float(1 / (1 - rate))}
	runElementwise(recorder, loaded.dropout, count, words, values)
}
