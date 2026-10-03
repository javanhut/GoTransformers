package gputraining

import (
	"fmt"
	"github.com/javanhut/GoTransformers/attention"
	"github.com/javanhut/GoTransformers/embedding"
	"github.com/javanhut/GoTransformers/gpu"
	"github.com/javanhut/GoTransformers/lowprecision"
	"github.com/javanhut/GoTransformers/optimizer"
	"github.com/javanhut/GoTransformers/perceptron"
	"github.com/javanhut/GoTransformers/transformer"
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
	"slices"
	"sort"
	"strings"
)

type TrainerOptions struct {
	LearningRate         float64
	WeightDecay          float64
	MomentumDecay        float64
	SquaredGradientDecay float64
	Epsilon              float64
	Schedule             optimizer.LearningRateSchedule
	MaximumGradientNorm  float64
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

func (options TrainerOptions) Check() error {
	if options.LearningRate <= 0 {
		return fmt.Errorf("LearningRate must be above 0, got %v", options.LearningRate)
	}
	if options.MaximumGradientNorm < 0 {
		return fmt.Errorf("MaximumGradientNorm can't be negative (0 means no clipping), got %v", options.MaximumGradientNorm)
	}
	return options.Schedule.Check()
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
	tokenIDs                   *gpu.Buffer
	uniqueTokenIDs             *gpu.Buffer
	rowStarts                  *gpu.Buffer
	sortedRows                 *gpu.Buffer
	droppedGradients           *gpu.Buffer
	droppedProbabilities       *gpu.Buffer
	positionVectors            *gpu.Buffer
}

type Trainer struct {
	Options TrainerOptions

	device           *gpu.Device
	model            *transformer.Model
	kernels          *kernels
	recorder         *gpu.Recorder
	tokenTable       *parameterOnGPU
	blocks           []blockOnGPU
	finalNorm        *parameterOnGPU
	finalNormEpsilon float64
	outputLayer      layerOnGPU
	parameters       []*parameterOnGPU

	partialSums      *gpu.Buffer
	clipResult       *gpu.Buffer
	stepsTaken       int
	lastGradientNorm float64

	vectorSize       int
	numberOfHeads    int
	keyValueHeads    int
	headSize         int
	feedForwardSize  int
	vocabularySize   int
	clampLimit       float64
	useRotary        bool
	queryRotary      rotarySetup
	keyRotary        rotarySetup
	residualDropout  float64
	attentionDropout float64
	stepSeed         uint32

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
	if err := options.Check(); err != nil {
		return nil, err
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
		Options:          options,
		device:           device,
		model:            model,
		kernels:          loaded,
		recorder:         recorder,
		vectorSize:       settings.VectorSize,
		numberOfHeads:    firstAttention.NumberOfHeads,
		keyValueHeads:    firstAttention.NumberOfKeyValueHeads,
		headSize:         firstAttention.HeadSize(),
		feedForwardSize:  model.Blocks[0].FeedForward.Layers()[0].NumberOfOutputs(),
		vocabularySize:   settings.VocabularySize,
		clampLimit:       settings.FeedForwardClampLimit,
		useRotary:        firstAttention.UseRotaryPositions,
		residualDropout:  settings.ResidualDropout,
		attentionDropout: settings.AttentionDropout,
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

	if err := trainer.uploadModel(); err != nil {
		trainer.Close()
		return nil, err
	}
	if err := trainer.makeClippingBuffers(); err != nil {
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

func (trainer *Trainer) newLayerTiedToTokenTable(layer *perceptron.Layer) (layerOnGPU, error) {
	biases, err := trainer.newParameter(layer.Name+".biases", layer.Biases, false)
	if err != nil {
		return layerOnGPU{}, err
	}
	return layerOnGPU{numberOfInputs: layer.NumberOfInputs(), numberOfOutputs: layer.NumberOfOutputs(), weights: trainer.tokenTable, biases: biases}, nil
}

func (trainer *Trainer) isTied() bool {
	return trainer.model.Settings.TieOutputToEmbedding
}

func (trainer *Trainer) uploadModel() error {
	model := trainer.model
	tokenEmbedding := model.TokenEmbedding
	var err error
	if trainer.tokenTable, err = trainer.newParameter(tokenEmbedding.Name+".table", tokenEmbedding.Table.Values, true); err != nil {
		return err
	}
	for _, block := range model.Blocks {
		selfAttention := block.Attention.(*attention.SelfAttention)
		feedForwardLayers := block.FeedForward.Layers()
		onGPU := blockOnGPU{attentionNormEpsilon: block.AttentionNorm.Epsilon, feedForwardNormEpsilon: block.FeedForwardNorm.Epsilon}
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
	if trainer.finalNorm, err = trainer.newParameter(model.FinalNorm.Name+".weights", model.FinalNorm.Weights, false); err != nil {
		return err
	}
	trainer.finalNormEpsilon = model.FinalNorm.Epsilon
	if trainer.isTied() {
		trainer.outputLayer, err = trainer.newLayerTiedToTokenTable(model.OutputLayer)
	} else {
		trainer.outputLayer, err = trainer.newLayer(model.OutputLayer)
	}
	return err
}

func (trainer *Trainer) makeClippingBuffers() error {
	numberOfPartials := 0
	for _, current := range trainer.parameters {
		numberOfPartials += squaredSumGroupsFor(len(current.cpuValues))
	}
	var err error
	if trainer.partialSums, err = trainer.newBuffer(numberOfPartials); err != nil {
		return err
	}
	if trainer.clipResult, err = trainer.newBuffer(2); err != nil {
		return err
	}
	return trainer.clipResult.Upload([]float64{1, 0})
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

func (trainer *Trainer) SaveModel(path string) error {
	if err := trainer.CopyWeightsToModel(); err != nil {
		return err
	}
	return trainer.model.Save(path)
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
		tokenIDs:                   activationBuffer(rows),
		uniqueTokenIDs:             activationBuffer(rows),
		rowStarts:                  activationBuffer(rows + 1),
		sortedRows:                 activationBuffer(rows),
	}
	if trainer.residualDropout > 0 {
		trainer.shared.droppedGradients = activationBuffer(rows * vectorSize)
	}
	if trainer.attentionDropout > 0 {
		trainer.shared.droppedProbabilities = activationBuffer(probabilityCount)
	}
	if !trainer.useRotary {
		trainer.shared.positionVectors = activationBuffer(sequenceLength * vectorSize)
	}
	if allocationError != nil {
		trainer.freeActivations()
		return allocationError
	}
	if !trainer.useRotary {
		positionVectors := make([]float64, sequenceLength*vectorSize)
		for position := range sequenceLength {
			copy(positionVectors[position*vectorSize:(position+1)*vectorSize], embedding.PositionalEncodingAt(position, vectorSize))
		}
		if err := trainer.shared.positionVectors.Upload(positionVectors); err != nil {
			return err
		}
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
		shared.valueGradientsPerQueryHead, shared.hiddenGradients, shared.gateGradients, shared.upGradients,
		shared.tokenIDs, shared.uniqueTokenIDs, shared.rowStarts, shared.sortedRows, shared.droppedGradients, shared.droppedProbabilities, shared.positionVectors)
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

const (
	attentionOutputDropoutSite   = 0
	feedForwardOutputDropoutSite = 1
	attentionWeightsDropoutSite  = 2
)

func (trainer *Trainer) dropoutSeed(blockIndex int, site int) uint32 {
	return trainer.stepSeed ^ uint32(blockIndex*3+site+1)*0x9e3779b9
}

func (trainer *Trainer) droppedProbabilitiesFor(blockIndex int, activations blockActivations, numberOfSequences int, sequenceLength int) *gpu.Buffer {
	count := numberOfSequences * trainer.numberOfHeads * sequenceLength * sequenceLength
	trainer.recorder.Copy(activations.probabilities, trainer.shared.droppedProbabilities, count)
	trainer.kernels.dropOut(trainer.recorder, trainer.shared.droppedProbabilities, count, trainer.dropoutSeed(blockIndex, attentionWeightsDropoutSite), trainer.attentionDropout)
	return trainer.shared.droppedProbabilities
}

func (trainer *Trainer) attentionForward(blockIndex int, block blockOnGPU, activations blockActivations, numberOfSequences int, sequenceLength int, isTraining bool) {
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
	usedProbabilities := activations.probabilities
	if isTraining && trainer.attentionDropout > 0 {
		usedProbabilities = trainer.droppedProbabilitiesFor(blockIndex, activations, numberOfSequences, sequenceLength)
	}
	trainer.kernels.multiply(trainer.recorder,
		trainer.perHeadSquare(usedProbabilities, sequenceLength),
		trainer.perHead(activations.values, sequenceLength, trainer.keyValueHeads, group),
		trainer.perHead(activations.attended, sequenceLength, trainer.numberOfHeads, 1),
		trainer.headShape(numberOfSequences, sequenceLength, sequenceLength, trainer.headSize, 1, false))
}

func (trainer *Trainer) attentionBackward(blockIndex int, block blockOnGPU, activations blockActivations, numberOfSequences int, sequenceLength int, attendedGradients *gpu.Buffer, normedGradients *gpu.Buffer) {
	shared := trainer.shared
	rows := numberOfSequences * sequenceLength
	group := trainer.numberOfHeads / trainer.keyValueHeads
	scale := 1 / math.Sqrt(float64(trainer.headSize))
	probabilityCount := numberOfSequences * trainer.numberOfHeads * sequenceLength * sequenceLength
	usedProbabilities := activations.probabilities
	if trainer.attentionDropout > 0 {
		usedProbabilities = trainer.droppedProbabilitiesFor(blockIndex, activations, numberOfSequences, sequenceLength)
	}
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
		trainer.perHeadSquare(usedProbabilities, sequenceLength).transposed(),
		attendedGradientView,
		trainer.perHead(valueGradientTarget, sequenceLength, trainer.numberOfHeads, 1),
		trainer.headShape(numberOfSequences, sequenceLength, sequenceLength, trainer.headSize, 1, false))

	if trainer.attentionDropout > 0 {
		trainer.kernels.dropOut(trainer.recorder, shared.scoreProbabilityGradients, probabilityCount, trainer.dropoutSeed(blockIndex, attentionWeightsDropoutSite), trainer.attentionDropout)
	}
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
	numberOfSequences    int
	sequenceLength       int
	tokenIDs             []float64
	targets              []float64
	rowWeights           []float64
	uniqueTokenIDs       []float64
	rowStarts            []float64
	sortedRows           []float64
	numberOfUniqueTokens int
}

func (trainer *Trainer) prepareBatch(examples []transformer.Example, sequencesInWholeBatch int) (preparedBatch, error) {
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
	batch.tokenIDs = make([]float64, rows)
	batch.targets = make([]float64, rows)
	batch.rowWeights = make([]float64, rows)
	rowsByToken := map[int][]int{}
	for sequence, tokenIDs := range allTokenIDs {
		inputIDs := tokenIDs[:len(tokenIDs)-1]
		counted := 0
		for row := range inputIDs {
			if row+1 >= firstCountedTokens[sequence] {
				counted++
			}
		}
		for position := 0; position < len(inputIDs); position++ {
			row := sequence*batch.sequenceLength + position
			batch.tokenIDs[row] = float64(inputIDs[position])
			batch.targets[row] = float64(tokenIDs[position+1])
			if position+1 >= firstCountedTokens[sequence] && counted > 0 {
				batch.rowWeights[row] = 1 / float64(sequencesInWholeBatch*counted)
			}
			rowsByToken[inputIDs[position]] = append(rowsByToken[inputIDs[position]], row)
		}
	}
	batch.addEmbeddingGradientLists(rowsByToken)
	return batch, nil
}

func (batch *preparedBatch) addEmbeddingGradientLists(rowsByToken map[int][]int) {
	var uniqueTokenIDs []int
	for tokenID := range rowsByToken {
		uniqueTokenIDs = append(uniqueTokenIDs, tokenID)
	}
	sort.Ints(uniqueTokenIDs)
	batch.numberOfUniqueTokens = len(uniqueTokenIDs)
	batch.rowStarts = append(batch.rowStarts, 0)
	for _, tokenID := range uniqueTokenIDs {
		batch.uniqueTokenIDs = append(batch.uniqueTokenIDs, float64(tokenID))
		for _, row := range rowsByToken[tokenID] {
			batch.sortedRows = append(batch.sortedRows, float64(row))
		}
		batch.rowStarts = append(batch.rowStarts, float64(len(batch.sortedRows)))
	}
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
	return trainer.TrainOnExamples(examplesFromSequences(sequences))
}

func examplesFromSequences(sequences [][]int) []transformer.Example {
	examples := make([]transformer.Example, len(sequences))
	for i, sequence := range sequences {
		examples[i] = transformer.Example{AnswerIDs: sequence}
	}
	return examples
}

func (trainer *Trainer) TrainOnExamples(examples []transformer.Example) float64 {
	loss, err := trainer.trainOnExamples(examples)
	if err != nil {
		panic(fmt.Sprintf("gputraining: %v", err))
	}
	return loss
}

func (trainer *Trainer) EvaluationLoss(sequences [][]int) float64 {
	loss, err := trainer.evaluationLoss(examplesFromSequences(sequences))
	if err != nil {
		panic(fmt.Sprintf("gputraining: %v", err))
	}
	return loss
}

func (trainer *Trainer) EvaluationLossOnExamples(examples []transformer.Example) float64 {
	loss, err := trainer.evaluationLoss(examples)
	if err != nil {
		panic(fmt.Sprintf("gputraining: %v", err))
	}
	return loss
}

func (trainer *Trainer) startBatch(examples []transformer.Example, sequencesInWholeBatch int) (preparedBatch, error) {
	if trainer.closed {
		return preparedBatch{}, fmt.Errorf("the trainer is closed")
	}
	batch, err := trainer.prepareBatch(examples, sequencesInWholeBatch)
	if err != nil {
		return preparedBatch{}, err
	}
	if err := trainer.ensureCapacity(batch.numberOfSequences, batch.sequenceLength); err != nil {
		return preparedBatch{}, err
	}
	trainer.recorder.Begin()
	shared := trainer.shared
	uploads := []struct {
		target *gpu.Buffer
		values []float64
	}{
		{shared.tokenIDs, batch.tokenIDs},
		{shared.targets, batch.targets},
		{shared.rowWeights, batch.rowWeights},
		{shared.uniqueTokenIDs, batch.uniqueTokenIDs},
		{shared.rowStarts, batch.rowStarts},
		{shared.sortedRows, batch.sortedRows},
	}
	for _, upload := range uploads {
		if err := trainer.recorder.Upload(upload.target, upload.values); err != nil {
			return preparedBatch{}, err
		}
	}
	return batch, nil
}

func sumOf(values []float64) float64 {
	total := 0.0
	for _, value := range values {
		total += value
	}
	return total
}

func (trainer *Trainer) evaluationLoss(examples []transformer.Example) (float64, error) {
	return trainer.evaluationLossInWholeBatch(examples, len(examples))
}

func (trainer *Trainer) evaluationLossInWholeBatch(examples []transformer.Example, sequencesInWholeBatch int) (float64, error) {
	batch, err := trainer.startBatch(examples, sequencesInWholeBatch)
	if err != nil {
		return 0, err
	}
	trainer.recordForward(batch, false)
	rowLosses := make([]float64, batch.numberOfSequences*batch.sequenceLength)
	trainer.recorder.Download(trainer.shared.rowLosses, rowLosses)
	if err := trainer.recorder.Submit(); err != nil {
		return 0, err
	}
	return sumOf(rowLosses), nil
}

func (trainer *Trainer) trainOnExamples(examples []transformer.Example) (float64, error) {
	batch, err := trainer.startBatch(examples, len(examples))
	if err != nil {
		return 0, err
	}
	trainer.stepSeed = uint32(vectormath.RandomNumberBetween(0, 4294967295))
	trainer.recordForward(batch, true)
	trainer.recordBackward(batch)
	rowLosses := make([]float64, batch.numberOfSequences*batch.sequenceLength)
	trainer.recorder.Download(trainer.shared.rowLosses, rowLosses)
	clipResult := trainer.recordOptimizerStep()
	if err := trainer.recorder.Submit(); err != nil {
		return 0, err
	}
	trainer.lastGradientNorm = clipResult[1]
	return sumOf(rowLosses), nil
}

func (trainer *Trainer) recordForward(batch preparedBatch, isTraining bool) {
	numberOfSequences := batch.numberOfSequences
	sequenceLength := batch.sequenceLength
	rows := numberOfSequences * sequenceLength
	vectorSize := trainer.vectorSize
	recorder := trainer.recorder
	shared := trainer.shared
	loaded := trainer.kernels

	loaded.gatherEmbeddings(recorder, trainer.tokenTable.values, shared.tokenIDs, shared.positionVectors, trainer.activations[0].inputs, rows, vectorSize, sequenceLength)
	for blockIndex, block := range trainer.blocks {
		activations := trainer.activations[blockIndex]
		loaded.normalizeForward(recorder, activations.inputs, block.attentionNormWeights.values, activations.attentionNormed, activations.attentionRootMeanSquares, rows, vectorSize, block.attentionNormEpsilon)
		trainer.attentionForward(blockIndex, block, activations, numberOfSequences, sequenceLength, isTraining)
		trainer.linearForward(block.output, activations.attended, shared.layerOutput, rows)
		if isTraining {
			loaded.dropOut(recorder, shared.layerOutput, rows*vectorSize, trainer.dropoutSeed(blockIndex, attentionOutputDropoutSite), trainer.residualDropout)
		}
		loaded.addBuffers(recorder, activations.inputs, shared.layerOutput, activations.afterAttention, rows*vectorSize)

		loaded.normalizeForward(recorder, activations.afterAttention, block.feedForwardNormWeights.values, activations.feedForwardNormed, activations.feedForwardRootMeanSquares, rows, vectorSize, block.feedForwardNormEpsilon)
		trainer.linearForward(block.gate, activations.feedForwardNormed, activations.gate, rows)
		trainer.linearForward(block.up, activations.feedForwardNormed, activations.up, rows)
		loaded.gatedForward(recorder, activations.gate, activations.up, activations.hidden, rows*trainer.feedForwardSize, trainer.clampLimit)
		trainer.linearForward(block.down, activations.hidden, shared.layerOutput, rows)
		if isTraining {
			loaded.dropOut(recorder, shared.layerOutput, rows*vectorSize, trainer.dropoutSeed(blockIndex, feedForwardOutputDropoutSite), trainer.residualDropout)
		}
		nextInputs := shared.finalInputs
		if blockIndex+1 < len(trainer.blocks) {
			nextInputs = trainer.activations[blockIndex+1].inputs
		}
		loaded.addBuffers(recorder, activations.afterAttention, shared.layerOutput, nextInputs, rows*vectorSize)
	}

	loaded.normalizeForward(recorder, shared.finalInputs, trainer.finalNorm.values, shared.finalNormed, shared.finalRootMeanSquares, rows, vectorSize, trainer.finalNormEpsilon)
	trainer.linearForward(trainer.outputLayer, shared.finalNormed, shared.scores, rows)
	loaded.crossEntropyRows(recorder, shared.scores, shared.targets, shared.rowWeights, shared.scoreGradients, shared.rowLosses, rows, trainer.vocabularySize)
}

func (trainer *Trainer) droppedGradientsFor(gradients *gpu.Buffer, blockIndex int, site int, count int) *gpu.Buffer {
	if trainer.residualDropout <= 0 {
		return gradients
	}
	trainer.recorder.Copy(gradients, trainer.shared.droppedGradients, count)
	trainer.kernels.dropOut(trainer.recorder, trainer.shared.droppedGradients, count, trainer.dropoutSeed(blockIndex, site), trainer.residualDropout)
	return trainer.shared.droppedGradients
}

func (trainer *Trainer) recordBackward(batch preparedBatch) {
	numberOfSequences := batch.numberOfSequences
	sequenceLength := batch.sequenceLength
	rows := numberOfSequences * sequenceLength
	vectorSize := trainer.vectorSize
	recorder := trainer.recorder
	shared := trainer.shared
	loaded := trainer.kernels

	if !trainer.isTied() {
		recorder.Fill(trainer.tokenTable.gradients, 0)
	}
	trainer.linearBackward(trainer.outputLayer, shared.finalNormed, shared.scoreGradients, shared.normedGradients, rows, false)
	loaded.normalizeBackward(recorder, shared.finalInputs, trainer.finalNorm.values, shared.finalRootMeanSquares, shared.normedGradients, nil, shared.firstGradients, rows, vectorSize)
	loaded.normalizeWeightGradients(recorder, shared.finalInputs, shared.finalRootMeanSquares, shared.normedGradients, trainer.finalNorm.gradients, rows, vectorSize, false)

	outputGradients := shared.firstGradients
	middleGradients := shared.secondGradients
	for blockIndex, block := range slices.Backward(trainer.blocks) {
		activations := trainer.activations[blockIndex]

		fedForwardGradients := trainer.droppedGradientsFor(outputGradients, blockIndex, feedForwardOutputDropoutSite, rows*vectorSize)
		trainer.linearBackward(block.down, activations.hidden, fedForwardGradients, shared.hiddenGradients, rows, false)
		loaded.gatedBackward(recorder, activations.gate, activations.up, shared.hiddenGradients, shared.gateGradients, shared.upGradients, rows*trainer.feedForwardSize, trainer.clampLimit)
		trainer.linearBackward(block.gate, activations.feedForwardNormed, shared.gateGradients, shared.normedGradients, rows, false)
		trainer.linearBackward(block.up, activations.feedForwardNormed, shared.upGradients, shared.normedGradients, rows, true)
		loaded.normalizeBackward(recorder, activations.afterAttention, block.feedForwardNormWeights.values, activations.feedForwardRootMeanSquares, shared.normedGradients, outputGradients, middleGradients, rows, vectorSize)
		loaded.normalizeWeightGradients(recorder, activations.afterAttention, activations.feedForwardRootMeanSquares, shared.normedGradients, block.feedForwardNormWeights.gradients, rows, vectorSize, false)

		attendedOutputGradients := trainer.droppedGradientsFor(middleGradients, blockIndex, attentionOutputDropoutSite, rows*vectorSize)
		trainer.linearBackward(block.output, activations.attended, attendedOutputGradients, shared.attendedGradients, rows, false)
		trainer.attentionBackward(blockIndex, block, activations, numberOfSequences, sequenceLength, shared.attendedGradients, shared.normedGradients)
		loaded.normalizeBackward(recorder, activations.inputs, block.attentionNormWeights.values, activations.attentionRootMeanSquares, shared.normedGradients, middleGradients, outputGradients, rows, vectorSize)
		loaded.normalizeWeightGradients(recorder, activations.inputs, activations.attentionRootMeanSquares, shared.normedGradients, block.attentionNormWeights.gradients, rows, vectorSize, false)
	}
	loaded.scatterEmbeddingGradients(recorder, outputGradients, shared.uniqueTokenIDs, shared.rowStarts, shared.sortedRows, trainer.tokenTable.gradients, batch.numberOfUniqueTokens, vectorSize)
}

func (trainer *Trainer) trainableParameters() []*parameterOnGPU {
	var trainable []*parameterOnGPU
	for _, current := range trainer.parameters {
		if !trainer.isFrozen(current.name) {
			trainable = append(trainable, current)
		}
	}
	return trainable
}

func (trainer *Trainer) LearningRate() float64 {
	return trainer.Options.LearningRate * trainer.Options.Schedule.FractionAt(trainer.stepsTaken)
}

func (trainer *Trainer) recordOptimizerStep() []float64 {
	recorder := trainer.recorder
	loaded := trainer.kernels
	trainable := trainer.trainableParameters()
	numberOfPartials := 0
	for _, current := range trainable {
		numberOfPartials += loaded.squaredSumsInto(recorder, current.gradients, trainer.partialSums, numberOfPartials, len(current.cpuValues))
	}
	loaded.computeClipScale(recorder, trainer.partialSums, trainer.clipResult, numberOfPartials, trainer.Options.MaximumGradientNorm)

	learningRate := trainer.LearningRate()
	trainer.stepsTaken++
	step := adamWStep{
		learningRate:                     learningRate,
		momentumDecay:                    trainer.Options.MomentumDecay,
		squaredGradientDecay:             trainer.Options.SquaredGradientDecay,
		epsilon:                          trainer.Options.Epsilon,
		averageGradientCorrection:        1 - math.Pow(trainer.Options.MomentumDecay, float64(trainer.stepsTaken)),
		averageSquaredGradientCorrection: 1 - math.Pow(trainer.Options.SquaredGradientDecay, float64(trainer.stepsTaken)),
	}
	for _, current := range trainable {
		shrinkFactor := 1.0
		if current.isMatrix {
			shrinkFactor = 1 - learningRate*trainer.Options.WeightDecay
		}
		loaded.adamWUpdate(recorder, current.values, current.gradients, current.averageGradients, current.averageSquaredGradients, trainer.clipResult, len(current.cpuValues), step, shrinkFactor)
	}
	clipResult := make([]float64, 2)
	recorder.Download(trainer.clipResult, clipResult)
	return clipResult
}

func (trainer *Trainer) LastGradientNorm() float64 {
	return trainer.lastGradientNorm
}

func (trainer *Trainer) StepsTaken() int {
	return trainer.stepsTaken
}
