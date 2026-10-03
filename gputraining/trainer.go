package gputraining

import (
	"fmt"
	"github.com/javanhut/GoTransformers/attention"
	"github.com/javanhut/GoTransformers/embedding"
	"github.com/javanhut/GoTransformers/gpu"
	"github.com/javanhut/GoTransformers/lowprecision"
	"github.com/javanhut/GoTransformers/optimizer"
	"github.com/javanhut/GoTransformers/parameter"
	"github.com/javanhut/GoTransformers/perceptron"
	"github.com/javanhut/GoTransformers/transformer"
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
	"strings"
)

type TrainerOptions struct {
	LearningRate         float64
	WeightDecay          float64
	MomentumDecay        float64
	SquaredGradientDecay float64
	Epsilon              float64
}

func DefaultTrainerOptions(learningRate float64, weightDecay float64) TrainerOptions {
	reference := optimizer.NewAdamW(learningRate, weightDecay)
	return TrainerOptions{
		LearningRate:         reference.LearningRate,
		WeightDecay:          reference.WeightDecay,
		MomentumDecay:        reference.MomentumDecay,
		SquaredGradientDecay: reference.SquaredGradientDecay,
		Epsilon:              reference.Epsilon,
	}
}

type parameterOnGPU struct {
	name                    string
	cpuValues               []float64
	isMatrix                bool
	values                  *gpu.Buffer
	gradients               *gpu.Buffer
	averageGradients        *gpu.Buffer
	averageSquaredGradients *gpu.Buffer
}

type layerOnGPU struct {
	numberOfInputs  int
	numberOfOutputs int
	weights         *parameterOnGPU
	biases          *parameterOnGPU
}

type blockOnGPU struct {
	attentionNormWeights   *parameterOnGPU
	attentionNormEpsilon   float64
	query                  layerOnGPU
	key                    layerOnGPU
	value                  layerOnGPU
	output                 layerOnGPU
	feedForwardNormWeights *parameterOnGPU
	feedForwardNormEpsilon float64
	gate                   layerOnGPU
	up                     layerOnGPU
	down                   layerOnGPU
}

type blockActivations struct {
	inputs                     *gpu.Buffer
	attentionNormed            *gpu.Buffer
	attentionRootMeanSquares   *gpu.Buffer
	queries                    *gpu.Buffer
	keys                       *gpu.Buffer
	values                     *gpu.Buffer
	probabilities              *gpu.Buffer
	attended                   *gpu.Buffer
	afterAttention             *gpu.Buffer
	feedForwardNormed          *gpu.Buffer
	feedForwardRootMeanSquares *gpu.Buffer
	gate                       *gpu.Buffer
	up                         *gpu.Buffer
	hidden                     *gpu.Buffer
}

type sharedBuffers struct {
	layerOutput                *gpu.Buffer
	finalInputs                *gpu.Buffer
	finalNormed                *gpu.Buffer
	finalRootMeanSquares       *gpu.Buffer
	scores                     *gpu.Buffer
	scoreGradients             *gpu.Buffer
	targets                    *gpu.Buffer
	rowWeights                 *gpu.Buffer
	rowLosses                  *gpu.Buffer
	firstGradients             *gpu.Buffer
	secondGradients            *gpu.Buffer
	normedGradients            *gpu.Buffer
	attendedGradients          *gpu.Buffer
	scoreProbabilityGradients  *gpu.Buffer
	queryGradients             *gpu.Buffer
	keyGradients               *gpu.Buffer
	valueGradients             *gpu.Buffer
	keyGradientsPerQueryHead   *gpu.Buffer
	valueGradientsPerQueryHead *gpu.Buffer
	hiddenGradients            *gpu.Buffer
	gateGradients              *gpu.Buffer
	upGradients                *gpu.Buffer
}

type Trainer struct {
	Options TrainerOptions

	device           *gpu.Device
	model            *transformer.Model
	kernels          *kernels
	recorder         *gpu.Recorder
	blocks           []blockOnGPU
	finalNorm        *parameterOnGPU
	finalNormEpsilon float64
	outputLayer      layerOnGPU
	parameters       []*parameterOnGPU

	embeddingOptimizer *optimizer.AdamW
	stepsTaken         int

	vectorSize      int
	numberOfHeads   int
	keyValueHeads   int
	headSize        int
	feedForwardSize int
	vocabularySize  int
	clampLimit      float64
	useRotary       bool
	queryRotary     rotarySetup
	keyRotary       rotarySetup

	capacityRows      int
	capacityLength    int
	capacitySequences int
	activations       []blockActivations
	shared            sharedBuffers
	allocated         []*gpu.Buffer
	closed            bool
}

func unsupportedFeatures(model *transformer.Model) []string {
	settings := model.Settings
	var problems []string
	add := func(condition bool, problem string) {
		if condition {
			problems = append(problems, problem)
		}
	}
	for blockIndex, kind := range settings.AttentionPattern {
		add(kind != transformer.StandardAttention, fmt.Sprintf("AttentionPattern entry %d is %q (only standard attention)", blockIndex, kind))
	}
	add(settings.WindowSize != 0, "WindowSize")
	add(settings.TopK != 0, "TopK")
	add(settings.BlocksPerKeyValueGroup > 1, "BlocksPerKeyValueGroup (key/value sharing across blocks)")
	add(settings.UseAttentionSink, "UseAttentionSink")
	add(settings.NormalizeQueriesAndKeys, "NormalizeQueriesAndKeys")
	add(settings.QueryRank != 0, "QueryRank (low-rank queries)")
	add(settings.ShareKeyAsValue, "ShareKeyAsValue")
	add(settings.TrainAtCachePrecision, "TrainAtCachePrecision")
	add(settings.UseMixtureOfExperts, "UseMixtureOfExperts")
	add(settings.NumberOfResidualStreams > 1, "NumberOfResidualStreams (mHC)")
	add(settings.MultiTokenPrediction, "MultiTokenPrediction")
	add(settings.WeightPrecision != lowprecision.Float64 || model.IsCompressed(), "compressed weights")
	add(settings.AdapterRank != 0 || len(model.AdapterParameters()) > 0, "low-rank adapters")
	add(settings.ResidualDropout > 0, "ResidualDropout")
	add(settings.AttentionDropout > 0, "AttentionDropout")
	add(settings.AdapterDropout > 0, "AdapterDropout")
	for blockIndex, block := range model.Blocks {
		selfAttention, isSelfAttention := block.Attention.(*attention.SelfAttention)
		add(!isSelfAttention, fmt.Sprintf("block %d does not use standard self-attention", blockIndex+1))
		if isSelfAttention {
			add(selfAttention.SharingMode != attention.OwnKeysAndValues, fmt.Sprintf("block %d borrows keys and values", blockIndex+1))
			add(!selfAttention.HideFutureTokens, fmt.Sprintf("block %d attention is not causal", blockIndex+1))
		}
		add(block.AttentionConnection != nil, fmt.Sprintf("block %d uses mHC", blockIndex+1))
		add(len(block.FeedForward.Layers()) != 3, fmt.Sprintf("block %d feed-forward is not SwiGLU", blockIndex+1))
	}
	return problems
}

func NewTrainer(device *gpu.Device, model *transformer.Model, options TrainerOptions) (*Trainer, error) {
	if problems := unsupportedFeatures(model); len(problems) > 0 {
		return nil, fmt.Errorf("GPU training doesn't support these settings yet: %s", strings.Join(problems, ", "))
	}
	if options.LearningRate <= 0 {
		return nil, fmt.Errorf("LearningRate must be above 0, got %v", options.LearningRate)
	}
	loaded, err := loadKernels(device)
	if err != nil {
		return nil, err
	}
	recorder, err := device.NewRecorder()
	if err != nil {
		loaded.free()
		return nil, err
	}
	settings := model.Settings
	firstAttention := model.Blocks[0].Attention.(*attention.SelfAttention)
	trainer := &Trainer{
		Options:         options,
		device:          device,
		model:           model,
		kernels:         loaded,
		recorder:        recorder,
		vectorSize:      settings.VectorSize,
		numberOfHeads:   firstAttention.NumberOfHeads,
		keyValueHeads:   firstAttention.NumberOfKeyValueHeads,
		headSize:        firstAttention.HeadSize(),
		feedForwardSize: model.Blocks[0].FeedForward.Layers()[0].NumberOfOutputs(),
		vocabularySize:  settings.VocabularySize,
		clampLimit:      settings.FeedForwardClampLimit,
		useRotary:       firstAttention.UseRotaryPositions,
	}
	if trainer.useRotary {
		rotaryDimensions := firstAttention.RotaryDimensions
		if rotaryDimensions == 0 {
			rotaryDimensions = trainer.headSize
		}
		base := firstAttention.RotaryBase
		if base == 0 {
			base = 10000
		}
		trainer.queryRotary = rotarySetup{numberOfHeads: trainer.numberOfHeads, headSize: trainer.headSize, rotaryDimensions: rotaryDimensions, rotateHalves: firstAttention.RotateHalves, base: base}
		trainer.keyRotary = trainer.queryRotary
		trainer.keyRotary.numberOfHeads = trainer.keyValueHeads
	}
	trainer.embeddingOptimizer = optimizer.NewAdamW(options.LearningRate, options.WeightDecay)
	trainer.embeddingOptimizer.MomentumDecay = options.MomentumDecay
	trainer.embeddingOptimizer.SquaredGradientDecay = options.SquaredGradientDecay
	trainer.embeddingOptimizer.Epsilon = options.Epsilon

	if err := trainer.uploadModel(); err != nil {
		trainer.Close()
		return nil, err
	}
	return trainer, nil
}

func (trainer *Trainer) newBuffer(numberOfFloats int) (*gpu.Buffer, error) {
	buffer, err := trainer.device.NewBuffer(numberOfFloats)
	if err != nil {
		return nil, err
	}
	trainer.allocated = append(trainer.allocated, buffer)
	return buffer, nil
}

func (trainer *Trainer) newParameter(name string, values []float64, isMatrix bool) (*parameterOnGPU, error) {
	created := &parameterOnGPU{name: name, cpuValues: values, isMatrix: isMatrix}
	for _, target := range []**gpu.Buffer{&created.values, &created.gradients, &created.averageGradients, &created.averageSquaredGradients} {
		buffer, err := trainer.newBuffer(len(values))
		if err != nil {
			return nil, err
		}
		*target = buffer
	}
	if err := created.values.Upload(values); err != nil {
		return nil, err
	}
	zeros := make([]float64, len(values))
	if err := created.averageGradients.Upload(zeros); err != nil {
		return nil, err
	}
	if err := created.averageSquaredGradients.Upload(zeros); err != nil {
		return nil, err
	}
	trainer.parameters = append(trainer.parameters, created)
	return created, nil
}

func (trainer *Trainer) newLayer(layer *perceptron.Layer) (layerOnGPU, error) {
	weights, err := trainer.newParameter(layer.Name+".weights", layer.Weights.Values, layer.NumberOfInputs() > 1 && layer.NumberOfOutputs() > 1)
	if err != nil {
		return layerOnGPU{}, err
	}
	biases, err := trainer.newParameter(layer.Name+".biases", layer.Biases, false)
	if err != nil {
		return layerOnGPU{}, err
	}
	return layerOnGPU{numberOfInputs: layer.NumberOfInputs(), numberOfOutputs: layer.NumberOfOutputs(), weights: weights, biases: biases}, nil
}

func (trainer *Trainer) uploadModel() error {
	model := trainer.model
	for _, block := range model.Blocks {
		selfAttention := block.Attention.(*attention.SelfAttention)
		feedForwardLayers := block.FeedForward.Layers()
		onGPU := blockOnGPU{attentionNormEpsilon: block.AttentionNorm.Epsilon, feedForwardNormEpsilon: block.FeedForwardNorm.Epsilon}
		var err error
		if onGPU.attentionNormWeights, err = trainer.newParameter(block.AttentionNorm.Name+".weights", block.AttentionNorm.Weights, false); err != nil {
			return err
		}
		layers := []struct {
			target *layerOnGPU
			source *perceptron.Layer
		}{
			{&onGPU.query, selfAttention.QueryLayer},
			{&onGPU.key, selfAttention.KeyLayer},
			{&onGPU.value, selfAttention.ValueLayer},
			{&onGPU.output, selfAttention.OutputLayer},
			{&onGPU.gate, feedForwardLayers[0]},
			{&onGPU.up, feedForwardLayers[1]},
			{&onGPU.down, feedForwardLayers[2]},
		}
		for _, layer := range layers {
			if *layer.target, err = trainer.newLayer(layer.source); err != nil {
				return err
			}
		}
		if onGPU.feedForwardNormWeights, err = trainer.newParameter(block.FeedForwardNorm.Name+".weights", block.FeedForwardNorm.Weights, false); err != nil {
			return err
		}
		trainer.blocks = append(trainer.blocks, onGPU)
	}
	var err error
	if trainer.finalNorm, err = trainer.newParameter(model.FinalNorm.Name+".weights", model.FinalNorm.Weights, false); err != nil {
		return err
	}
	trainer.finalNormEpsilon = model.FinalNorm.Epsilon
	if trainer.outputLayer, err = trainer.newLayer(model.OutputLayer); err != nil {
		return err
	}
	return nil
}

func (trainer *Trainer) UploadWeightsFromModel() error {
	for _, current := range trainer.parameters {
		if err := current.values.Upload(current.cpuValues); err != nil {
			return err
		}
	}
	return nil
}

func (trainer *Trainer) CopyWeightsToModel() error {
	for _, current := range trainer.parameters {
		if err := current.values.Download(current.cpuValues); err != nil {
			return err
		}
	}
	vectormath.MarkWeightsChanged()
	return nil
}

func (trainer *Trainer) ensureCapacity(numberOfSequences int, sequenceLength int) error {
	rows := numberOfSequences * sequenceLength
	if rows <= trainer.capacityRows && sequenceLength <= trainer.capacityLength && numberOfSequences <= trainer.capacitySequences {
		return nil
	}
	sequenceLength = max(sequenceLength, trainer.capacityLength)
	numberOfSequences = max(numberOfSequences, trainer.capacitySequences)
	rows = numberOfSequences * sequenceLength
	trainer.freeActivations()

	vectorSize := trainer.vectorSize
	queryWidth := trainer.numberOfHeads * trainer.headSize
	keyWidth := trainer.keyValueHeads * trainer.headSize
	probabilityCount := numberOfSequences * trainer.numberOfHeads * sequenceLength * sequenceLength
	var allocationError error
	activationBuffer := func(size int) *gpu.Buffer {
		if allocationError != nil {
			return nil
		}
		buffer, err := trainer.device.NewBuffer(size)
		if err != nil {
			allocationError = fmt.Errorf("could not reserve GPU memory for %d sequences of %d tokens: %v", numberOfSequences, sequenceLength, err)
		}
		return buffer
	}
	trainer.activations = nil
	for range trainer.blocks {
		trainer.activations = append(trainer.activations, blockActivations{
			inputs:                     activationBuffer(rows * vectorSize),
			attentionNormed:            activationBuffer(rows * vectorSize),
			attentionRootMeanSquares:   activationBuffer(rows),
			queries:                    activationBuffer(rows * queryWidth),
			keys:                       activationBuffer(rows * keyWidth),
			values:                     activationBuffer(rows * keyWidth),
			probabilities:              activationBuffer(probabilityCount),
			attended:                   activationBuffer(rows * queryWidth),
			afterAttention:             activationBuffer(rows * vectorSize),
			feedForwardNormed:          activationBuffer(rows * vectorSize),
			feedForwardRootMeanSquares: activationBuffer(rows),
			gate:                       activationBuffer(rows * trainer.feedForwardSize),
			up:                         activationBuffer(rows * trainer.feedForwardSize),
			hidden:                     activationBuffer(rows * trainer.feedForwardSize),
		})
	}
	trainer.shared = sharedBuffers{
		layerOutput:                activationBuffer(rows * vectorSize),
		finalInputs:                activationBuffer(rows * vectorSize),
		finalNormed:                activationBuffer(rows * vectorSize),
		finalRootMeanSquares:       activationBuffer(rows),
		scores:                     activationBuffer(rows * trainer.vocabularySize),
		scoreGradients:             activationBuffer(rows * trainer.vocabularySize),
		targets:                    activationBuffer(rows),
		rowWeights:                 activationBuffer(rows),
		rowLosses:                  activationBuffer(rows),
		firstGradients:             activationBuffer(rows * vectorSize),
		secondGradients:            activationBuffer(rows * vectorSize),
		normedGradients:            activationBuffer(rows * vectorSize),
		attendedGradients:          activationBuffer(rows * queryWidth),
		scoreProbabilityGradients:  activationBuffer(probabilityCount),
		queryGradients:             activationBuffer(rows * queryWidth),
		keyGradients:               activationBuffer(rows * keyWidth),
		valueGradients:             activationBuffer(rows * keyWidth),
		keyGradientsPerQueryHead:   activationBuffer(rows * queryWidth),
		valueGradientsPerQueryHead: activationBuffer(rows * queryWidth),
		hiddenGradients:            activationBuffer(rows * trainer.feedForwardSize),
		gateGradients:              activationBuffer(rows * trainer.feedForwardSize),
		upGradients:                activationBuffer(rows * trainer.feedForwardSize),
	}
	if allocationError != nil {
		trainer.freeActivations()
		return allocationError
	}
	if trainer.useRotary {
		if err := makeRotaryTables(trainer.device, &trainer.queryRotary, sequenceLength); err != nil {
			return err
		}
		if err := makeRotaryTables(trainer.device, &trainer.keyRotary, sequenceLength); err != nil {
			return err
		}
	}
	trainer.capacityRows = rows
	trainer.capacityLength = sequenceLength
	trainer.capacitySequences = numberOfSequences
	return nil
}

func (trainer *Trainer) activationBuffers() []*gpu.Buffer {
	var buffers []*gpu.Buffer
	for _, block := range trainer.activations {
		buffers = append(buffers, block.inputs, block.attentionNormed, block.attentionRootMeanSquares, block.queries, block.keys, block.values,
			block.probabilities, block.attended, block.afterAttention, block.feedForwardNormed, block.feedForwardRootMeanSquares, block.gate, block.up, block.hidden)
	}
	shared := trainer.shared
	return append(buffers, shared.layerOutput, shared.finalInputs, shared.finalNormed, shared.finalRootMeanSquares, shared.scores, shared.scoreGradients,
		shared.targets, shared.rowWeights, shared.rowLosses, shared.firstGradients, shared.secondGradients, shared.normedGradients, shared.attendedGradients,
		shared.scoreProbabilityGradients, shared.queryGradients, shared.keyGradients, shared.valueGradients, shared.keyGradientsPerQueryHead,
		shared.valueGradientsPerQueryHead, shared.hiddenGradients, shared.gateGradients, shared.upGradients)
}

func (trainer *Trainer) freeActivations() {
	for _, buffer := range trainer.activationBuffers() {
		if buffer != nil {
			buffer.Free()
		}
	}
	trainer.activations = nil
	trainer.shared = sharedBuffers{}
	trainer.capacityRows = 0
	trainer.capacityLength = 0
	trainer.capacitySequences = 0
}

func (trainer *Trainer) Close() {
	if trainer.closed {
		return
	}
	trainer.freeActivations()
	for _, buffer := range trainer.allocated {
		buffer.Free()
	}
	for _, setup := range []*rotarySetup{&trainer.queryRotary, &trainer.keyRotary} {
		if setup.cosines != nil {
			setup.cosines.Free()
			setup.sines.Free()
		}
	}
	if trainer.recorder != nil {
		trainer.recorder.Free()
	}
	if trainer.kernels != nil {
		trainer.kernels.free()
	}
	trainer.closed = true
}

func (trainer *Trainer) linearForward(layer layerOnGPU, inputs *gpu.Buffer, outputs *gpu.Buffer, rows int) {
	trainer.kernels.multiply(trainer.recorder,
		wholeMatrix(inputs, layer.numberOfInputs),
		wholeMatrix(layer.weights.values, layer.numberOfInputs).transposed(),
		wholeMatrix(outputs, layer.numberOfOutputs),
		plainShape(rows, layer.numberOfInputs, layer.numberOfOutputs))
	trainer.kernels.addBias(trainer.recorder, layer.biases.values, outputs, rows, layer.numberOfOutputs)
}

func (trainer *Trainer) linearBackward(layer layerOnGPU, inputs *gpu.Buffer, outputGradients *gpu.Buffer, inputGradients *gpu.Buffer, rows int, addToInputGradients bool) {
	trainer.kernels.multiply(trainer.recorder,
		wholeMatrix(outputGradients, layer.numberOfOutputs).transposed(),
		wholeMatrix(inputs, layer.numberOfInputs),
		wholeMatrix(layer.weights.gradients, layer.numberOfInputs),
		plainShape(layer.numberOfOutputs, rows, layer.numberOfInputs))
	trainer.kernels.columnSumsInto(trainer.recorder, outputGradients, layer.biases.gradients, rows, layer.numberOfOutputs, false)
	inputShape := plainShape(rows, layer.numberOfOutputs, layer.numberOfInputs)
	inputShape.accumulate = addToInputGradients
	trainer.kernels.multiply(trainer.recorder,
		wholeMatrix(outputGradients, layer.numberOfOutputs),
		wholeMatrix(layer.weights.values, layer.numberOfInputs),
		wholeMatrix(inputGradients, layer.numberOfInputs),
		inputShape)
}

func (trainer *Trainer) perHead(buffer *gpu.Buffer, sequenceLength int, heads int, divisor int) matrixView {
	width := heads * trainer.headSize
	return matrixView{buffer: buffer, rowStride: width, columnStride: 1, outerStride: sequenceLength * width, innerStride: trainer.headSize, innerDivisor: divisor}
}

func (trainer *Trainer) perHeadSquare(buffer *gpu.Buffer, sequenceLength int) matrixView {
	return matrixView{buffer: buffer, rowStride: sequenceLength, columnStride: 1, outerStride: trainer.numberOfHeads * sequenceLength * sequenceLength, innerStride: sequenceLength * sequenceLength, innerDivisor: 1}
}

func (trainer *Trainer) headShape(numberOfSequences int, rows int, inner int, columns int, scale float64, accumulate bool) multiplyShape {
	return multiplyShape{rows: rows, inner: inner, columns: columns, batchCount: numberOfSequences * trainer.numberOfHeads, innerBatchCount: trainer.numberOfHeads, scale: scale, accumulate: accumulate}
}

func (trainer *Trainer) attentionForward(block blockOnGPU, activations blockActivations, numberOfSequences int, sequenceLength int) {
	rows := numberOfSequences * sequenceLength
	group := trainer.numberOfHeads / trainer.keyValueHeads
	scale := 1 / math.Sqrt(float64(trainer.headSize))
	trainer.linearForward(block.query, activations.attentionNormed, activations.queries, rows)
	trainer.linearForward(block.key, activations.attentionNormed, activations.keys, rows)
	trainer.linearForward(block.value, activations.attentionNormed, activations.values, rows)
	if trainer.useRotary {
		trainer.kernels.rotate(trainer.recorder, activations.queries, rows, sequenceLength, trainer.queryRotary, rotateForward)
		trainer.kernels.rotate(trainer.recorder, activations.keys, rows, sequenceLength, trainer.keyRotary, rotateForward)
	}
	keys := trainer.perHead(activations.keys, sequenceLength, trainer.keyValueHeads, group)
	trainer.kernels.multiply(trainer.recorder,
		trainer.perHead(activations.queries, sequenceLength, trainer.numberOfHeads, 1),
		keys.transposed(),
		trainer.perHeadSquare(activations.probabilities, sequenceLength),
		trainer.headShape(numberOfSequences, sequenceLength, trainer.headSize, sequenceLength, scale, false))
	trainer.kernels.causalSoftmaxRows(trainer.recorder, activations.probabilities, numberOfSequences*trainer.numberOfHeads*sequenceLength, sequenceLength)
	trainer.kernels.multiply(trainer.recorder,
		trainer.perHeadSquare(activations.probabilities, sequenceLength),
		trainer.perHead(activations.values, sequenceLength, trainer.keyValueHeads, group),
		trainer.perHead(activations.attended, sequenceLength, trainer.numberOfHeads, 1),
		trainer.headShape(numberOfSequences, sequenceLength, sequenceLength, trainer.headSize, 1, false))
}

func (trainer *Trainer) attentionBackward(block blockOnGPU, activations blockActivations, numberOfSequences int, sequenceLength int, attendedGradients *gpu.Buffer, normedGradients *gpu.Buffer) {
	shared := trainer.shared
	rows := numberOfSequences * sequenceLength
	group := trainer.numberOfHeads / trainer.keyValueHeads
	scale := 1 / math.Sqrt(float64(trainer.headSize))
	probabilities := trainer.perHeadSquare(activations.probabilities, sequenceLength)
	probabilityGradients := trainer.perHeadSquare(shared.scoreProbabilityGradients, sequenceLength)
	attendedGradientView := trainer.perHead(attendedGradients, sequenceLength, trainer.numberOfHeads, 1)

	trainer.kernels.multiply(trainer.recorder,
		attendedGradientView,
		trainer.perHead(activations.values, sequenceLength, trainer.keyValueHeads, group).transposed(),
		probabilityGradients,
		trainer.headShape(numberOfSequences, sequenceLength, trainer.headSize, sequenceLength, 1, false))

	valueGradientTarget := shared.valueGradients
	keyGradientTarget := shared.keyGradients
	if group > 1 {
		valueGradientTarget = shared.valueGradientsPerQueryHead
		keyGradientTarget = shared.keyGradientsPerQueryHead
	}
	trainer.kernels.multiply(trainer.recorder,
		probabilities.transposed(),
		attendedGradientView,
		trainer.perHead(valueGradientTarget, sequenceLength, trainer.numberOfHeads, 1),
		trainer.headShape(numberOfSequences, sequenceLength, sequenceLength, trainer.headSize, 1, false))

	trainer.kernels.softmaxBackwardRows(trainer.recorder, activations.probabilities, shared.scoreProbabilityGradients, numberOfSequences*trainer.numberOfHeads*sequenceLength, sequenceLength)

	trainer.kernels.multiply(trainer.recorder,
		probabilityGradients,
		trainer.perHead(activations.keys, sequenceLength, trainer.keyValueHeads, group),
		trainer.perHead(shared.queryGradients, sequenceLength, trainer.numberOfHeads, 1),
		trainer.headShape(numberOfSequences, sequenceLength, sequenceLength, trainer.headSize, scale, false))
	trainer.kernels.multiply(trainer.recorder,
		probabilityGradients.transposed(),
		trainer.perHead(activations.queries, sequenceLength, trainer.numberOfHeads, 1),
		trainer.perHead(keyGradientTarget, sequenceLength, trainer.numberOfHeads, 1),
		trainer.headShape(numberOfSequences, sequenceLength, sequenceLength, trainer.headSize, scale, false))

	if group > 1 {
		trainer.kernels.sumHeadGroups(trainer.recorder, shared.keyGradientsPerQueryHead, shared.keyGradients, rows, trainer.keyValueHeads, group, trainer.headSize)
		trainer.kernels.sumHeadGroups(trainer.recorder, shared.valueGradientsPerQueryHead, shared.valueGradients, rows, trainer.keyValueHeads, group, trainer.headSize)
	}
	if trainer.useRotary {
		trainer.kernels.rotate(trainer.recorder, shared.queryGradients, rows, sequenceLength, trainer.queryRotary, rotateBackward)
		trainer.kernels.rotate(trainer.recorder, shared.keyGradients, rows, sequenceLength, trainer.keyRotary, rotateBackward)
	}
	trainer.linearBackward(block.query, activations.attentionNormed, shared.queryGradients, normedGradients, rows, false)
	trainer.linearBackward(block.key, activations.attentionNormed, shared.keyGradients, normedGradients, rows, true)
	trainer.linearBackward(block.value, activations.attentionNormed, shared.valueGradients, normedGradients, rows, true)
}

type preparedBatch struct {
	numberOfSequences int
	sequenceLength    int
	inputIDs          [][]int
	embeddedRows      []float64
	targets           []float64
	rowWeights        []float64
}

func (trainer *Trainer) prepareBatch(examples []transformer.Example) (preparedBatch, error) {
	if len(examples) == 0 {
		return preparedBatch{}, fmt.Errorf("no examples given")
	}
	batch := preparedBatch{numberOfSequences: len(examples)}
	var allTokenIDs [][]int
	var firstCountedTokens []int
	for exampleIndex, example := range examples {
		if len(example.AnswerIDs) == 0 {
			return preparedBatch{}, fmt.Errorf("example %d has no answer tokens", exampleIndex)
		}
		tokenIDs := append(append([]int(nil), example.PromptIDs...), example.AnswerIDs...)
		if len(tokenIDs) < 2 {
			return preparedBatch{}, fmt.Errorf("example %d needs at least 2 tokens to learn what comes next", exampleIndex)
		}
		for position, tokenID := range tokenIDs {
			if tokenID < 0 || tokenID >= trainer.vocabularySize {
				return preparedBatch{}, fmt.Errorf("example %d: token ID %d at position %d is outside the vocabulary of %d tokens", exampleIndex, tokenID, position, trainer.vocabularySize)
			}
		}
		firstCounted := 1
		if len(example.PromptIDs) > 0 {
			firstCounted = len(example.PromptIDs)
		}
		allTokenIDs = append(allTokenIDs, tokenIDs)
		firstCountedTokens = append(firstCountedTokens, firstCounted)
		batch.sequenceLength = max(batch.sequenceLength, len(tokenIDs)-1)
	}

	rows := batch.numberOfSequences * batch.sequenceLength
	vectorSize := trainer.vectorSize
	batch.embeddedRows = make([]float64, rows*vectorSize)
	batch.targets = make([]float64, rows)
	batch.rowWeights = make([]float64, rows)
	tokenEmbedding := trainer.model.TokenEmbedding
	for sequence, tokenIDs := range allTokenIDs {
		inputIDs := tokenIDs[:len(tokenIDs)-1]
		batch.inputIDs = append(batch.inputIDs, inputIDs)
		counted := 0
		for row := range inputIDs {
			if row+1 >= firstCountedTokens[sequence] {
				counted++
			}
		}
		for position := 0; position < batch.sequenceLength; position++ {
			row := sequence*batch.sequenceLength + position
			tokenID := 0
			if position < len(inputIDs) {
				tokenID = inputIDs[position]
				batch.targets[row] = float64(tokenIDs[position+1])
				if position+1 >= firstCountedTokens[sequence] && counted > 0 {
					batch.rowWeights[row] = 1 / float64(batch.numberOfSequences*counted)
				}
			}
			vector := tokenEmbedding.VectorFor(tokenID)
			if !trainer.useRotary {
				vector = vectormath.Add(vector, embedding.PositionalEncodingAt(position, vectorSize))
			}
			copy(batch.embeddedRows[row*vectorSize:(row+1)*vectorSize], vector)
		}
	}
	return batch, nil
}

func (trainer *Trainer) isFrozen(name string) bool {
	for _, prefix := range trainer.model.FrozenNamePrefixes {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

func (trainer *Trainer) TrainBatch(sequences [][]int) float64 {
	examples := make([]transformer.Example, len(sequences))
	for i, sequence := range sequences {
		examples[i] = transformer.Example{AnswerIDs: sequence}
	}
	return trainer.TrainOnExamples(examples)
}

func (trainer *Trainer) TrainOnExamples(examples []transformer.Example) float64 {
	loss, err := trainer.trainOnExamples(examples)
	if err != nil {
		panic(fmt.Sprintf("gputraining: %v", err))
	}
	return loss
}

func (trainer *Trainer) trainOnExamples(examples []transformer.Example) (float64, error) {
	if trainer.closed {
		return 0, fmt.Errorf("the trainer is closed")
	}
	batch, err := trainer.prepareBatch(examples)
	if err != nil {
		return 0, err
	}
	if err := trainer.ensureCapacity(batch.numberOfSequences, batch.sequenceLength); err != nil {
		return 0, err
	}
	numberOfSequences := batch.numberOfSequences
	sequenceLength := batch.sequenceLength
	rows := numberOfSequences * sequenceLength
	vectorSize := trainer.vectorSize
	recorder := trainer.recorder
	shared := trainer.shared
	loaded := trainer.kernels

	recorder.Begin()
	if err := recorder.Upload(trainer.activations[0].inputs, batch.embeddedRows); err != nil {
		return 0, err
	}
	if err := recorder.Upload(shared.targets, batch.targets); err != nil {
		return 0, err
	}
	if err := recorder.Upload(shared.rowWeights, batch.rowWeights); err != nil {
		return 0, err
	}
	for blockIndex, block := range trainer.blocks {
		activations := trainer.activations[blockIndex]
		loaded.normalizeForward(recorder, activations.inputs, block.attentionNormWeights.values, activations.attentionNormed, activations.attentionRootMeanSquares, rows, vectorSize, block.attentionNormEpsilon)
		trainer.attentionForward(block, activations, numberOfSequences, sequenceLength)
		trainer.linearForward(block.output, activations.attended, shared.layerOutput, rows)
		loaded.addBuffers(recorder, activations.inputs, shared.layerOutput, activations.afterAttention, rows*vectorSize)

		loaded.normalizeForward(recorder, activations.afterAttention, block.feedForwardNormWeights.values, activations.feedForwardNormed, activations.feedForwardRootMeanSquares, rows, vectorSize, block.feedForwardNormEpsilon)
		trainer.linearForward(block.gate, activations.feedForwardNormed, activations.gate, rows)
		trainer.linearForward(block.up, activations.feedForwardNormed, activations.up, rows)
		loaded.gatedForward(recorder, activations.gate, activations.up, activations.hidden, rows*trainer.feedForwardSize, trainer.clampLimit)
		trainer.linearForward(block.down, activations.hidden, shared.layerOutput, rows)
		nextInputs := shared.finalInputs
		if blockIndex+1 < len(trainer.blocks) {
			nextInputs = trainer.activations[blockIndex+1].inputs
		}
		loaded.addBuffers(recorder, activations.afterAttention, shared.layerOutput, nextInputs, rows*vectorSize)
	}

	loaded.normalizeForward(recorder, shared.finalInputs, trainer.finalNorm.values, shared.finalNormed, shared.finalRootMeanSquares, rows, vectorSize, trainer.finalNormEpsilon)
	trainer.linearForward(trainer.outputLayer, shared.finalNormed, shared.scores, rows)
	loaded.crossEntropyRows(recorder, shared.scores, shared.targets, shared.rowWeights, shared.scoreGradients, shared.rowLosses, rows, trainer.vocabularySize)

	trainer.linearBackward(trainer.outputLayer, shared.finalNormed, shared.scoreGradients, shared.normedGradients, rows, false)
	loaded.normalizeBackward(recorder, shared.finalInputs, trainer.finalNorm.values, shared.finalRootMeanSquares, shared.normedGradients, nil, shared.firstGradients, rows, vectorSize)
	loaded.normalizeWeightGradients(recorder, shared.finalInputs, shared.finalRootMeanSquares, shared.normedGradients, trainer.finalNorm.gradients, rows, vectorSize, false)

	outputGradients := shared.firstGradients
	middleGradients := shared.secondGradients
	for blockIndex := len(trainer.blocks) - 1; blockIndex >= 0; blockIndex-- {
		block := trainer.blocks[blockIndex]
		activations := trainer.activations[blockIndex]

		trainer.linearBackward(block.down, activations.hidden, outputGradients, shared.hiddenGradients, rows, false)
		loaded.gatedBackward(recorder, activations.gate, activations.up, shared.hiddenGradients, shared.gateGradients, shared.upGradients, rows*trainer.feedForwardSize, trainer.clampLimit)
		trainer.linearBackward(block.gate, activations.feedForwardNormed, shared.gateGradients, shared.normedGradients, rows, false)
		trainer.linearBackward(block.up, activations.feedForwardNormed, shared.upGradients, shared.normedGradients, rows, true)
		loaded.normalizeBackward(recorder, activations.afterAttention, block.feedForwardNormWeights.values, activations.feedForwardRootMeanSquares, shared.normedGradients, outputGradients, middleGradients, rows, vectorSize)
		loaded.normalizeWeightGradients(recorder, activations.afterAttention, activations.feedForwardRootMeanSquares, shared.normedGradients, block.feedForwardNormWeights.gradients, rows, vectorSize, false)

		trainer.linearBackward(block.output, activations.attended, middleGradients, shared.attendedGradients, rows, false)
		trainer.attentionBackward(block, activations, numberOfSequences, sequenceLength, shared.attendedGradients, shared.normedGradients)
		loaded.normalizeBackward(recorder, activations.inputs, block.attentionNormWeights.values, activations.attentionRootMeanSquares, shared.normedGradients, middleGradients, outputGradients, rows, vectorSize)
		loaded.normalizeWeightGradients(recorder, activations.inputs, activations.attentionRootMeanSquares, shared.normedGradients, block.attentionNormWeights.gradients, rows, vectorSize, false)
	}

	trainer.stepsTaken++
	step := adamWStep{
		learningRate:                     trainer.Options.LearningRate,
		momentumDecay:                    trainer.Options.MomentumDecay,
		squaredGradientDecay:             trainer.Options.SquaredGradientDecay,
		epsilon:                          trainer.Options.Epsilon,
		averageGradientCorrection:        1 - math.Pow(trainer.Options.MomentumDecay, float64(trainer.stepsTaken)),
		averageSquaredGradientCorrection: 1 - math.Pow(trainer.Options.SquaredGradientDecay, float64(trainer.stepsTaken)),
	}
	for _, current := range trainer.parameters {
		if trainer.isFrozen(current.name) {
			continue
		}
		shrinkFactor := 1.0
		if current.isMatrix {
			shrinkFactor = 1 - trainer.Options.LearningRate*trainer.Options.WeightDecay
		}
		loaded.adamWUpdate(recorder, current.values, current.gradients, current.averageGradients, current.averageSquaredGradients, len(current.cpuValues), step, shrinkFactor)
	}

	embeddingGradients := make([]float64, rows*vectorSize)
	rowLosses := make([]float64, rows)
	recorder.Download(outputGradients, embeddingGradients)
	recorder.Download(shared.rowLosses, rowLosses)
	if err := recorder.Submit(); err != nil {
		return 0, err
	}

	trainer.updateEmbedding(batch, embeddingGradients)
	loss := 0.0
	for _, rowLoss := range rowLosses {
		loss += rowLoss
	}
	return loss, nil
}

func (trainer *Trainer) updateEmbedding(batch preparedBatch, embeddingGradients []float64) {
	tokenEmbedding := trainer.model.TokenEmbedding
	embeddingParameters := tokenEmbedding.Parameters()
	trainer.embeddingOptimizer.LearningRate = trainer.Options.LearningRate
	trainer.embeddingOptimizer.WeightDecay = trainer.Options.WeightDecay
	trainer.embeddingOptimizer.MomentumDecay = trainer.Options.MomentumDecay
	trainer.embeddingOptimizer.SquaredGradientDecay = trainer.Options.SquaredGradientDecay
	trainer.embeddingOptimizer.Epsilon = trainer.Options.Epsilon
	if trainer.isFrozen(embeddingParameters[0].Name) {
		trainer.embeddingOptimizer.Update(nil)
		return
	}
	parameter.ZeroGradients(embeddingParameters)
	vectorSize := trainer.vectorSize
	for sequence, inputIDs := range batch.inputIDs {
		firstRow := sequence * batch.sequenceLength
		gradients := vectormath.Matrix{Rows: len(inputIDs), Columns: vectorSize, Values: embeddingGradients[firstRow*vectorSize : (firstRow+len(inputIDs))*vectorSize]}
		tokenEmbedding.AddGradients(inputIDs, gradients)
	}
	trainer.embeddingOptimizer.Update(embeddingParameters)
}

func (trainer *Trainer) StepsTaken() int {
	return trainer.stepsTaken
}
