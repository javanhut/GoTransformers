# Reference

Every package and its public API. Packages are imported as `github.com/javanhut/GoTransformers/<package>`. See [ARCHITECTURE.md](ARCHITECTURE.md) for how they fit together.

**Contents:** [vectormath](#vectormath) · [parameter](#parameter) · [lowprecision](#lowprecision) · [activationfunction](#activationfunction) · [lossfunction](#lossfunction) · [dropout](#dropout) · [perceptron](#perceptron) · [normalization](#normalization) · [embedding](#embedding) · [feedforward](#feedforward) · [attention](#attention) · [mixtureofexperts](#mixtureofexperts) · [hyperconnection](#hyperconnection) · [optimizer](#optimizer) · [transformer](#transformer) · [constrained](#constrained) · [training](#training) · [chat](#chat) · [tokenizer](#tokenizer) · [safetensors](#safetensors) · [pretrained](#pretrained) · [weightfile](#weightfile) · [datafile](#datafile) · [parquet](#parquet) · [compression](#compression) · [gpu](#gpu) · [gputraining](#gputraining) · [gradientcheck](#gradientcheck)

Shape mistakes (wrong sizes, a missing setting, `Backward` before `Forward`) panic with a message saying exactly what didn't match. File problems return an `error`.

---

## vectormath

Vectors, matrices, the math backend and random numbers.

**Types**

| Name | What it is |
|---|---|
| `Vector` | `[]float64` |
| `Matrix` | `Rows`, `Columns`, and `Values` stored row by row |
| `Backend` | anything that can do the three matrix multiplies (`CPUBackend`, `gpu.Device`) |
| `WeightAwareBackend` | a backend that can also keep weights resident (`gpu.Device`) |
| `CPUBackend` | multi-threaded CPU math, the default |

**Vectors**

| Function | What it does |
|---|---|
| `NewVector(size)` | vector of zeros |
| `CopyVector(vector)` | independent copy |
| `Add`, `Subtract`, `MultiplyEach(first, second)` | element by element |
| `Scale(vector, amount)` | multiply every value |
| `DotProduct(first, second)` | sum of products |
| `Sum`, `Magnitude`, `Max`, `IndexOfMax(vector)` | single results |
| `ApplyToEach(vector, function)` | new vector with `function` applied to each value |

**Matrices**

| Function or method | What it does |
|---|---|
| `NewMatrix(rows, columns)` | matrix of zeros |
| `NewRandomMatrix(rows, columns, lowest, highest)` | uniform random values |
| `MatrixFromRows(rows)` | build from a list of vectors |
| `matrix.Get(row, column)`, `Set`, `AddTo` | read, write or add to one value (bounds-checked) |
| `matrix.Row(row)` | view of one row; writing to it changes the matrix |
| `matrix.SetRow(row, values)`, `matrix.Copy()` | |
| `Transpose(matrix)` | |
| `MatrixTimesVector(matrix, vector)` | |
| `MatrixTimesMatrix(first, second)` | through the current backend |
| `MatrixTimesTransposed(first, second)` | `first × secondᵀ`, through the backend |
| `TransposedTimesMatrix(first, second)` | `firstᵀ × second`, through the backend |
| `MatrixTimesTransposedWeights(inputs, weights)` | like `MatrixTimesTransposed`, but tells the backend the second matrix is weights it may keep on the GPU |
| `MatrixTimesWeights(first, weights)` | like `MatrixTimesMatrix`, same weight hint |
| `AddMatrices`, `SubtractMatrices`, `MultiplyMatricesEach(first, second)` | element by element |
| `ScaleMatrix(matrix, amount)`, `OuterProduct(first, second)` | |
| `StackRows(top, bottom)`, `SplitRows(matrix, topRows)` | join or split by rows |
| `JoinColumns(left, right)`, `SplitColumns(matrix, leftColumns)` | join or split by columns |

**Backend and threads**

| Function | What it does |
|---|---|
| `UseBackend(backend)`, `CurrentBackend()` | switch between CPU and GPU (`nil` means CPU) |
| `SplitAcrossThreads(numberOfItems, workPerItem, doItems)` | run `doItems(first, last)` over ranges on `GOMAXPROCS` threads when the work is big enough |
| `MarkWeightsChanged()`, `WeightsVersion()` | tell GPU weight caches that weights changed |

**Random numbers**

| Function | What it does |
|---|---|
| `SetRandomSeed(seed)` | reset the shared generator (default seed is fixed, so runs repeat) |
| `RandomNumberBetween(lowest, highest)` | |
| `SaveRandomState()`, `RestoreRandomState(state)` | used by checkpoints to resume exactly |

---

## parameter

A named slice of trainable values and where its gradients live.

```go
type Parameter struct {
	Name            string      // unique, dotted: "block2.attention.query.weights"
	Values          []float64   // nil when the weights are compressed
	GradientStorage *[]float64  // where the owner keeps gradients
	Rows, Columns   int         // set for matrices
	UseAdamW        bool        // Muon sends these to AdamW
	ReadOnly        bool        // compressed weights; optimizers refuse them
}
```

| Function or method | What it does |
|---|---|
| `WithGradients(name, values)` | standalone parameter with its own gradient slice |
| `p.Gradients()` | the gradients, created (as zeros) on first use |
| `p.HasGradients()`, `p.ReleaseGradients()` | |
| `p.IsMatrix()`, `p.Count()`, `p.IsCompressed()` | |
| `ZeroGradients(parameters)` | clear gradients that exist |
| `ReleaseGradients(parameters)`, `GradientBytes(parameters)` | |

---

## lowprecision

Rows of numbers stored compactly. Used for caches and compressed weights.

| Name | What it is |
|---|---|
| `Precision` | `Float64`, `Float32`, `Int8`, `FP4`; prints and parses as its name |
| `ValuesPerScale` | 16: Int8 and FP4 keep one float32 scale per 16 values |
| `NewRows(precision, width)`, `RowsFromMatrix(matrix, precision)` | |
| `rows.Append(values)`, `rows.Row(index)`, `rows.NumberOfRows()` | `Row` returns a float64 copy |
| `rows.DotRow(row, vector)` | dot product read straight from the compressed form |
| `rows.AddScaledRowTo(row, scale, target)` | add `scale × row` into `target`, read straight from the compressed form |
| `rows.DropOldestRows(count)`, `rows.BytesUsed()` | |
| `rows.Snapshot()`, `RowsFromSnapshot(snapshot)` | for saving to disk |
| `RoundTrip(values, precision)`, `RoundTripMatrix(matrix, precision)` | store and read back, to simulate the rounding during training |

---

## activationfunction

```go
type Activation struct {
	Name       string
	Forward    ActivationFn   // func(float64) float64
	Derivative ActivationFn   // takes the input x, not Forward(x)
}
```

| Name | |
|---|---|
| `ReLU`, `LeakyReLU`, `ReLU6`, `ReLUSquared`, `ELU`, `SELU` | |
| `Sigmoid`, `HardSigmoid`, `Tanh`, `HardTanh`, `Softsign`, `Softplus` | |
| `SiLU`, `HardSwish`, `Mish`, `GELU` (tanh form), `GELUExact`, `QuickGELU`, `Linear` | |
| `ClampedSiLU(limit)`, `ClampedLinear(limit)` | build clamped versions (DeepSeek-V4 uses 10) |
| `Get(name)` | look one up by name |
| `Softmax(inputs)` | numerically stable; handles `-Inf` |
| `SoftmaxBackward(outputs, outputGradients)` | gradient with respect to the softmax inputs |

---

## dropout

During training, zeroes a random fraction of values and scales the rest by `1 / (1 − rate)` so the average stays the same. Does nothing unless switched on. Models switch it on only inside their training methods.

| Name | What it does |
|---|---|
| `New(rate)` | a dropout, or `nil` when the rate is 0; every method is safe to call on `nil` |
| `d.Forward(inputs)`, `d.Backward(gradients)` | masks a whole matrix, and uses the same mask going back |
| `d.StartPass()`, `d.NextMask(size)` | for components that need several masks per pass (attention uses one per head and position) |
| `d.SetActive(active)`, `d.IsOn()` | |
| `d.RepeatLastMasks` | reuse the previous pass's masks; for gradient checks |

---

## lossfunction

```go
type Loss struct {
	Name      string
	Calculate LossFn          // (predictions, targets Matrix) float64
	Gradient  LossGradientFn  // (predictions, targets Matrix) Matrix
}
```

| Name | |
|---|---|
| `MeanSquaredError` | averaged over every value |
| `SoftmaxCrossEntropy` | predictions are raw scores; targets are one-hot or probability rows; averaged over rows |

---

## perceptron

| Name | What it does |
|---|---|
| `Perceptron{Weights, Bias, Activation}`, `.Forward(inputs)` | a single neuron |
| `NewLayer(name, numberOfInputs, numberOfOutputs, activation)` | dense layer; Xavier-uniform weights, zero biases, no gradient memory yet |
| `layer.Forward(inputs)`, `layer.Backward(outputGradients)`, `layer.Parameters()` | |
| `layer.NumberOfInputs()`, `layer.NumberOfOutputs()` | |
| `layer.CompressWeights(precision)`, `layer.DecompressWeights()`, `layer.IsCompressed()` | store weights as Float32, Int8 or FP4 (inference only) |
| `layer.SetWeights(values)`, `layer.SetBiases(values)` | works whether or not the layer is compressed |
| `layer.WeightBytes()`, `layer.ReleaseGradients()` | |
| `layer.AddLowRankAdapter(rank, alpha)` | LoRA: adds `(alpha / rank) × Up × Down` to the output; `Up` starts at zero so nothing changes until training; freezes the base weights |
| `layer.MergeLowRankAdapter()`, `layer.RemoveLowRankAdapter()` | fold the adapter into the weights (not compressed), or drop it |
| `layer.Adapter`, `layer.FreezeBase` | the adapter, and whether the base weights and biases are frozen (no gradient memory, no weight gradients computed) |
| `layer.Adapter.InputDropout` | optional dropout on the adapter's input (the usual LoRA dropout) |
| `NewMultiLayerPerceptron(layerSizes, hiddenActivation, outputActivation)` | e.g. `[]int{2, 8, 1}` makes 2 layers named `layer1`, `layer2` |
| `network.Forward`, `Backward`, `Parameters`, `Predict(vector)` | |
| `network.TrainStep(inputs, targets, loss, optimizer)` | one full step, returns the loss |

---

## normalization

| Name | What it does |
|---|---|
| `NewRMSNorm(name, vectorSize)` | `x / rms(x) × weights` per row; `Epsilon` defaults to 1e-6 |
| `NewLayerNorm(name, vectorSize)` | `(x − mean) / std × weights + biases` per row; `Epsilon` 1e-5 |
| `.Forward`, `.Backward`, `.Parameters` | |

---

## embedding

| Name | What it does |
|---|---|
| `NewEmbedding(name, vocabularySize, vectorSize)` | lookup table, one row per token |
| `e.Forward(tokenIDs)`, `e.Backward(outputGradients)` | |
| `e.VectorFor(tokenID)`, `e.TableRows(tokenIDs)` | read without touching training memory |
| `e.AddGradients(tokenIDs, gradients)` | add gradients for given tokens (used by multi-token prediction) |
| `e.CompressTable(precision)`, `e.SetTable(values)`, `e.TableBytes()`, `e.ReleaseGradients()` | |
| `e.Frozen` | when true, backward passes add no gradients (set automatically when the model freezes it) |
| `PositionalEncoding(sequenceLength, vectorSize)`, `PositionalEncodingAt(position, vectorSize)` | sine-wave positions |
| `NewVocabulary()`, `BuildVocabulary(tokens)` | simple token-to-ID table; ID 0 is `UnknownToken` |
| `vocabulary.Add`, `Encode`, `Decode`, `Size`, `SaveToFile`, `LoadVocabularyFromFile` | |
| `SplitIntoWords(text)`, `SplitIntoCharacters(text)` | |

---

## feedforward

`GatedFeedForward` computes `Down(gate(x) × up(x))`, where `GateLayer`, `UpLayer` and `DownLayer` are `perceptron.Layer`s.

| Constructor | Gate activation |
|---|---|
| `NewSwiGLU(name, vectorSize, hiddenSize)` | SiLU |
| `NewClampedSwiGLU(name, vectorSize, hiddenSize, limit)` | SiLU capped at `limit`, up path clamped to `[−limit, limit]` |
| `NewGeGLU`, `NewReGLU` | GELU, ReLU |
| `NewGatedFeedForward(name, vectorSize, hiddenSize, gateActivation, upActivation)` | anything |

Methods: `Forward`, `Backward`, `Parameters`, `Layers`.

---

## attention

### SelfAttention

| Constructor | |
|---|---|
| `NewSelfAttention(name, vectorSize, numberOfHeads, hideFutureTokens)` | plain multi-head attention |
| `NewSelfAttentionWithOptions(name, vectorSize, options)` | every option below |
| `NewBorrowingSelfAttention(name, sharedFrom, mode)` | reuse another layer's keys and values (and maybe its choices) |

`Options` fields (set before construction):

| Field | Meaning |
|---|---|
| `NumberOfHeads` | query heads |
| `NumberOfKeyValueHeads` | 0 = one per query head; fewer = grouped-query; 1 = multi-query |
| `HideFutureTokens` | causal attention (needed for generation) |
| `ShareKeyAsValue` | one cache entry serves as both key and value |
| `QueryRank` | 0 = full queries; otherwise squeeze queries through this size first |
| `NormalizeQueriesAndKeys` | RMSNorm on each head of the queries and keys |
| `UseAttentionSink` | learned extra logit per head |
| `UseRotaryPositions`, `RotaryDimensions` | rotary positions on the last `RotaryDimensions` of each head (0 = all) |
| `RotaryBase`, `RotateHalves` | base (0 = 10000) and Hugging Face's rotate-half pairing |
| `WindowSize`, `TopK` | 0 = off |
| `CachePrecision`, `TrainAtCachePrecision` | cache storage, and rounding during training to match |
| `AttentionDropout` | dropout on the attention weights while training (0 = off); kept in `WeightsDropout` |

`WindowSize`, `TopK`, `HideFutureTokens`, `CachePrecision` and `TrainAtCachePrecision` are also fields on `SelfAttention` and can be changed after construction.

`SharingMode`: `OwnKeysAndValues`, `BorrowKeysAndValues`, `BorrowKeysValuesAndChoices`.

| Method | What it does |
|---|---|
| `Forward`, `Backward`, `Parameters`, `Layers` | |
| `StartGenerating()`, `ForwardOneToken(vector)` | generation with the cache |
| `CacheBytesUsed()` | |
| `LookedAt(head, position)`, `LastAttentionWeights(head)` | inspect the last forward pass |
| `GenerationSnapshot()`, `RestoreGeneration(snapshot)` | save and restore the cache |
| `Options()`, `VectorSize()`, `HeadSize()` | |

### CompressedAttention

`NewCompressedAttention(name, vectorSize, options)` with `CompressedOptions`:

| Field | Meaning |
|---|---|
| `NumberOfHeads`, `HeadSize` | `HeadSize` 0 = `vectorSize / NumberOfHeads` |
| `CompressionRate` | tokens merged into each entry |
| `Overlap` | also use the previous block (CSA) |
| `TopK` | 0 = attend to every entry (HCA); otherwise use the lightning indexer |
| `NumberOfIndexerHeads`, `IndexerHeadSize` | defaults 2 and `HeadSize` |
| `WindowSize` | uncompressed recent tokens, at least 1 |
| `UseAttentionSink`, `UseRotaryPositions`, `RotaryDimensions` | |
| `CachePrecision`, `TrainAtCachePrecision` | |

Fields set after construction: `IndexerLossWeight` (default 1), `AttendToAllWhileTraining` (dense warm-up). `CompressedOptions` also takes `AttentionDropout`.

Methods: the same as `SelfAttention`, plus `LastIndexerLoss()`.

---

## mixtureofexperts

| Name | What it does |
|---|---|
| `NewMixtureOfExperts(name, vectorSize, expertHiddenSize, numberOfSharedExperts, numberOfRoutedExperts, expertsPerToken)` | SwiGLU experts |
| `NewClampedMixtureOfExperts(..., clampLimit)` | clamped SwiGLU experts |
| `m.Forward(inputs, tokenIDs)` | `tokenIDs` may be nil unless `UseHashRouting` is on |
| `m.Backward`, `m.Parameters`, `m.Layers` | |
| `m.UpdateBalance(rate)`, `m.LastExpertLoad()` | load balancing without an extra loss |
| `m.UseHashRouting` | choose experts from the token ID instead of the router |

Experts are chosen by `sqrt(softplus(score)) + BalanceBiases`, and weighted by `sqrt(softplus(score))` normalized over the chosen experts.

---

## hyperconnection

mHC residual streams. A sequence's state is a matrix with `numberOfStreams × vectorSize` columns.

| Name | What it does |
|---|---|
| `NewHyperConnection(name, vectorSize, numberOfStreams)` | `SinkhornSteps` defaults to 20 |
| `c.LayerInput(streams)` | the mix of streams to feed into a sublayer |
| `c.Combine(layerOutput)` | new streams: mixed old streams + the sublayer's output |
| `c.BackwardCombine(gradients)` | returns (sublayer output gradients, stream gradients) |
| `c.BackwardLayerInput(gradients)` | stream gradients from the sublayer input |
| `c.Parameters()`, `c.LastResidualMixing(token)` | |
| `ExpandToStreams`, `CollapseStreams`, and their `...Backward` | start and end of a model |

---

## optimizer

```go
type Optimizer interface { Update(parameters []parameter.Parameter) }
type Resumable interface { Optimizer; SaveState() State; RestoreState(State) error }
```

| Constructor | Notes |
|---|---|
| `NewSGD(learningRate)` | |
| `NewSGDWithMomentum(learningRate, momentum)` | |
| `NewAdam(learningRate)` | betas 0.9 / 0.999 |
| `NewAdamW(learningRate, weightDecay)` | betas 0.9 / 0.95; decay only on matrices |
| `NewMuon(learningRate)` | momentum 0.95, weight decay 0.1, update RMS 0.18; vectors and `UseAdamW` parameters go to its `AdamW` field |
| `NewtonSchulz(matrix)` | Muon's hybrid orthogonalization, on its own |

All of them are `Resumable`. Fields such as `LearningRate` can be changed between steps.

**Learning rate schedules and clipping**

| Name | What it does |
|---|---|
| `LearningRateSchedule{WarmupSteps, TotalSteps, DecaySteps, DecayShape, FinalFraction}` | `FractionAt(stepIndex)` gives the share of the peak learning rate for a step; the zero value keeps the rate constant |
| `WarmupThenCosine(warmupSteps, totalSteps, finalFraction)` | linear warmup, then a cosine down to `finalFraction` of the peak |
| `WarmupThenLinear(warmupSteps, totalSteps, finalFraction)` | linear warmup, then a straight line down |
| `WarmupStableDecay(warmupSteps, totalSteps, decaySteps, finalFraction)` | warmup, hold the peak, then decay over the last `decaySteps` |
| `schedule.Check()` | explains anything invalid |
| `GradientNorm(parameters)`, `ClipGradients(parameters, maximumNorm)` | global gradient norm; clipping scales every gradient down so the norm is at most `maximumNorm` and returns the norm before clipping |
| `LearningRates(optimizer)`, `SetLearningRates(optimizer, rates)`, `ScaleLearningRates(optimizer, peakRates, fraction)` | read and set every learning rate an optimizer has (Muon has two: its own and its AdamW) |

You rarely call these yourself: `transformer.Trainer` and `gputraining.Trainer` apply the schedule and clipping on every step.

---

## transformer

### Settings

`SmallSettings(vocabularySize)` gives a working small model. `DeepSeekStyleSettings(vocabularySize)` turns on the DeepSeek-V4 features. `settings.Check()` explains anything invalid, and `NewModel` calls it for you.

| Field | Default in `SmallSettings` | Meaning |
|---|---|---|
| `VocabularySize`, `VectorSize`, `NumberOfBlocks`, `NumberOfHeads` | –, 64, 4, 4 | |
| `FeedForwardSize`, `FeedForwardClampLimit` | 176, 0 | SwiGLU hidden size; clamp limit (0 = off) |
| `AttentionPattern` | empty (all standard) | per block, repeating: `StandardAttention`, `CompressedSparseAttention`, `HeavilyCompressedAttention` |
| `NumberOfKeyValueHeads`, `ShareKeyAsValue`, `QueryRank` | 0, false, 0 | see attention `Options` |
| `NormalizeQueriesAndKeys`, `UseAttentionSink` | false, false | |
| `UseRotaryPositions`, `RotaryDimensions`, `RotaryBase`, `RotateHalves` | true, 0, 0, false | |
| `NormEpsilon` | 0 (1e-6) | for every RMSNorm |
| `TieOutputToEmbedding` | false | the output layer reads the token embedding table instead of having its own weights (weight tying); saves `VocabularySize × VectorSize` parameters |
| `WindowSize`, `FullAttentionEvery`, `TopK` | 0, 0, 0 | standard blocks; every `FullAttentionEvery`-th block ignores the window |
| `BlocksPerKeyValueGroup`, `GroupSharingMode` | 1, `BorrowKeysAndValues` | groups of standard blocks sharing one cache |
| `CompressionRate`, `HeavyCompressionRate` | 4, 16 | CSA and HCA |
| `CompressedWindowSize`, `CompressedTopK`, `CompressedOverlap`, `NumberOfIndexerHeads` | 8, 4, true, 2 | |
| `AttendToAllWhileTraining` | false | indexer warm-up |
| `CachePrecision`, `TrainAtCachePrecision` | Float64, false | |
| `WeightPrecision` | Float64 | anything else builds an inference-only compressed model |
| `UseMixtureOfExperts`, `NumberOfSharedExperts`, `NumberOfRoutedExperts`, `ExpertsPerToken`, `ExpertHiddenSize` | false, 1, 4, 2, 64 | |
| `HashRoutedBlocks`, `BalanceUpdateRate` | 0, 0.001 | |
| `NumberOfResidualStreams` | 1 | more than 1 turns on mHC |
| `MultiTokenPrediction`, `MultiTokenLossWeight` | false, 0.3 | |
| `AdapterRank`, `AdapterAlpha` | 0, 0 | LoRA adapters on every block layer (0 = none); set by `AddLowRankAdapters` |
| `ResidualDropout`, `AttentionDropout`, `AdapterDropout` | 0, 0, 0 | dropout on each block's attention and feed-forward outputs, on attention weights, and on adapter inputs; only while training; 0.1 is a common choice |

### Building, saving, loading

| Function or method | What it does |
|---|---|
| `NewModel(settings)` | |
| `model.Save(path)`, `LoadModel(path)` | weights at `path`, settings at `path.settings.json`; loading finds out how the weights were stored |
| `model.SaveWithPrecision(path, weightfile.Float32)` | plain weights stored as `Float64` (the default, exact), `Float32` or `BFloat16` (round to nearest even); compressed weights (Float32, Int8, FP4) are always saved still compressed, byte for byte, and load back compressed |
| `model.SetWeightsAndCompressedWeights(savedValues, savedCompressedValues)` | like `SetWeights`, also installing compressed rows |
| `model.SaveCheckpoint(folder, optimizer, stepsDone)`, `LoadCheckpoint(folder, optimizer)` | resume training exactly |
| `model.Describe()`, `model.NumberOfParameters()`, `model.Parameters()` | |
| `model.SetWeights(savedValues)`, `model.SetWeight(name, values)` | fill weights by name, compressed or not |
| `model.CompressWeights(precision)`, `DecompressWeights()`, `IsCompressed()` | |
| `model.WeightBytes()`, `GradientBytes()`, `ReleaseGradients()` | |
| `model.Fingerprint()` | hash of settings and weights |

### Training

| Function or method | What it does |
|---|---|
| `model.TrainStep(tokenIDs, optimizer)` | one sequence, learn every next token |
| `model.TrainBatch(sequences, optimizer)` | average over several sequences |
| `model.TrainOnAnswer(example, optimizer)`, `TrainOnExamples(examples, optimizer)` | learn only the answer tokens |
| `Example{PromptIDs, AnswerIDs}` | an empty `PromptIDs` means learn the whole sequence |
| `model.ComputeGradients(tokenIDs)`, `ComputeAnswerGradients(example)` | forward + backward without updating |
| `model.Loss(tokenIDs)`, `TrainingLoss(tokenIDs)`, `AnswerLoss(example)` | without gradients; `TrainingLoss` includes multi-token prediction |
| `model.Freeze(namePrefix)`, `UnfreezeAll()`, `TrainableParameters()`, `FrozenNamePrefixes` | e.g. `Freeze("block1.")`, `Freeze("tokens")`; frozen layers get no gradient memory |
| `model.AddLowRankAdapters(rank, alpha)` | LoRA on every block layer; everything else freezes; works over compressed weights |
| `model.AdapterParameters()`, `SaveAdapters(path)`, `LoadAdapters(path)` | adapter-only files (plus `path.adapters.json` with rank and alpha) |
| `model.MergeAdapters()`, `RemoveAdapters()` | |
| `model.UpdateExpertBalance()` | called by the training methods |
| `RandomChunk(tokenIDs, length)`, `RandomChunks(tokenIDs, length, count)` | random training windows |
| `OneHotTargets(targetIDs, vocabularySize)` | |
| `model.Forward(tokenIDs)`, `model.Backward(scoreGradients)` | the raw passes |

**Trainer.** Wraps a model and an optimizer with a learning rate schedule, gradient clipping and evaluation:

```go
trainer, err := transformer.NewTrainer(model, optimizer.NewAdamW(0.003, 0.01), transformer.TrainerOptions{
	Schedule:            optimizer.WarmupThenCosine(100, 5000, 0.1),
	MaximumGradientNorm: 1,
})
loss := trainer.TrainBatch(chunks)
```

| Method | What it does |
|---|---|
| `NewTrainer(model, optimizer, TrainerOptions{Schedule, MaximumGradientNorm})` | `MaximumGradientNorm` 0 means no clipping |
| `trainer.TrainBatch(sequences)`, `TrainOnExamples(examples)` | one optimizer step at the scheduled learning rate |
| `trainer.EvaluationLoss(sequences)` | average next-token loss, no gradients, no dropout |
| `trainer.LearningRate()` | the rate the next step will use |
| `trainer.LastGradientNorm()` | the norm before clipping on the last step |
| `trainer.StepsTaken()`, `SetStepsTaken(steps)` | set it when resuming from a checkpoint so the schedule continues |
| `trainer.SaveModel(path)` | |

### Generating

| Function or method | What it does |
|---|---|
| `model.Generate(promptIDs, numberOfNewTokens, temperature)` | temperature 0 = always the most likely token |
| `model.StartGenerating()`, `Feed(tokenIDs)`, `NextTokenScores(tokenID)`, `ContinueGenerating(count, temperature)` | step by step |
| `PickToken(scores, temperature)` | |
| `PickTokenFromTop(scores, temperature, topProbability)` | top-p (nucleus) sampling |
| `model.CacheBytesUsed()` | |
| `model.SaveGenerationState(path)`, `LoadGenerationState(path)` | keep a processed prompt for later |
| `model.ScoreAnswer(promptIDs, answerIDs)` | an `AnswerScore` |
| `model.Answer(promptIDs, options)` | a `GeneratedAnswer`, abstaining below the confidence target |
| `NewSampler(SamplingOptions, constraint)` | a `Sampler`: `PickToken(scores)`, `AcceptToken(id)`, `RememberPrompt(ids)`, `StopTokenIDs` |
| `model.GenerateWithSampler(promptIDs, count, sampler)`, `ContinueGeneratingWithSampler(count, sampler)` | generate with penalties, sampling filters and a constraint |

```go
type GenerationOptions struct {
	MaximumNewTokens int
	Temperature      float64
	StopTokenIDs     []int      // stop on any of these (not included in the answer)
	ConfidenceTarget float64    // 0 = never abstain
	AbstainTokenIDs  []int      // returned instead when confidence is too low
	Sampling         SamplingOptions  // penalties and filters, below
	Constraint       TokenConstraint  // for example constrained.NewJSONConstraint; nil = none
}

type SamplingOptions struct {
	Temperature        float64  // 0 = most likely token (falls back to GenerationOptions.Temperature)
	TopK               int      // 0 = off
	TypicalProbability float64  // locally typical sampling; 0 or 1 = off
	TopProbability     float64  // top-p; 0 or 1 = off
	MinimumProbability float64  // min-p, a share of the top token's probability; 0 = off
	RepetitionPenalty  float64  // divides positive scores of seen tokens, multiplies negative ones; 0 or 1 = off
	FrequencyPenalty   float64  // minus this times how often a token was seen
	PresencePenalty    float64  // minus this once for every seen token
	PenaltyWindow      int      // how many recent tokens the penalties look at; 0 = all
	PenalizePrompt     bool
	UseRandomSeed      bool     // with RandomSeed: a private random stream, so runs repeat exactly
	RandomSeed         uint64
}

type AnswerScore struct {
	TokenProbabilities     []float64
	LogProbability         float64
	Probability            float64   // of the whole answer
	AverageTokenConfidence float64   // geometric mean per token
	LowestTokenProbability float64
}

type GeneratedAnswer struct {
	TokenIDs         []int
	Score            AnswerScore
	Abstained        bool
	RejectedTokenIDs []int   // the guess, when it abstained
	ConstraintSatisfied bool // the constraint says the output is complete
}
```

The sampler works in this order: penalties, the constraint's mask (stop tokens are only allowed once the constraint is complete), temperature and softmax over the allowed tokens, top-k, typical, top-p, min-p, then a draw. Because the softmax only covers allowed tokens, a constrained model always picks an allowed token, however unlikely.

### Parts

| Name | What it is |
|---|---|
| `Block` | `AttentionNorm`, `Attention`, `FeedForwardNorm`, `FeedForward`, and optional `AttentionConnection` / `FeedForwardConnection` for mHC |
| `AttentionLayer`, `FeedForwardLayer` | interfaces a block's parts satisfy, so you can plug in your own |
| `MultiTokenPredictor` | the extra block that predicts the token after next |

---

## constrained

Keeps generated text valid: every token's bytes are run through a byte-level automaton, and tokens that would break the format are masked out before sampling. Masks are cached by automaton state, so a 49k-token vocabulary costs about 6 ms the first time a state is seen and under a microsecond after that. It does not import the tokenizer; give it the token bytes from `chat.VocabularyBytes`.

| Name | What it does |
|---|---|
| `NewJSONConstraint(tokenBytes, JSONSettings)` | strict RFC 8259 JSON, including escapes, `\uXXXX`, numbers and UTF-8 split across tokens |
| `JSONSettings{MaximumDepth, MaximumWhitespaceInARow, MaximumNumberLength, RequireObjectAtTopLevel}`, `DefaultJSONSettings()` | 0 = unlimited; defaults 16, 16, 32 |
| `NewChoiceConstraint(tokenBytes, choices)` | the output must be exactly one of the strings |
| `NewSchemaConstraint(tokenBytes, schema, settings)` | a JSON object with the given keys, in order, and value types; for tool calls |
| `ObjectSchema(properties...)`, `Property{Name, Value}`, `StringSchema()`, `NumberSchema()`, `BooleanSchema()`, `EnumSchema(values...)`, `StringArraySchema()` | schema pieces |
| `TokenConstraint` | `Restart()`, `IsTokenAllowed(id)`, `AllowedTokens()`, `AcceptToken(id)`, `AcceptText(text)`, `IsComplete()`, `Text()` |
| `Automaton`, `NewTokenConstraint(tokenBytes, automaton)` | write your own format: `Clone()`, `AcceptByte(b)`, `IsComplete()`, `StateKey()` |

```go
toolCall, err := constrained.NewSchemaConstraint(chat.VocabularyBytes(modelTokenizer), constrained.ObjectSchema(
	constrained.Property{Name: "name", Value: constrained.EnumSchema("get_weather")},
	constrained.Property{Name: "arguments", Value: constrained.ObjectSchema(
		constrained.Property{Name: "city", Value: constrained.StringSchema()})},
), constrained.DefaultJSONSettings())
reply, err := conversation.Reply("What's the weather in Paris?", chat.ReplyOptions{MaximumNewTokens: 100, Constraint: toolCall}, nil)
```

If `MaximumNewTokens` runs out first, the output is a valid beginning but not complete; `ConstraintSatisfied` says so. Schema keys are all required and in a fixed order.

---

## training

A training loop that works with any trainer (`transformer.Trainer`, `gputraining.Trainer`, `gputraining.DataParallelTrainer`): it evaluates held-out data, saves the best model and can stop early.

```go
result, err := training.Loop{
	Trainer:             trainer,
	NumberOfSteps:       5000,
	NextBatch:           func() [][]int { return transformer.RandomChunks(trainingIDs, 129, 16) },
	EvaluationSequences: heldOutChunks,
	EvaluateEvery:       200,
	BestModelPath:       "best.weights",
	OnEvaluation:        func(report training.EvaluationReport) { fmt.Println(report.Step, report.EvaluationLoss) },
}.Run()
```

| Name | What it does |
|---|---|
| `Trainer` interface | `TrainBatch`, `EvaluationLoss`, `LearningRate`, `LastGradientNorm`, `StepsTaken`, `SaveModel` |
| `Loop{Trainer, NumberOfSteps, NextBatch, EvaluationSequences, EvaluationBatchSize, EvaluateEvery, BestModelPath, StopAfterEvaluationsWithoutImproving, OnStep, OnEvaluation}` | `Run()` returns `Result{StepsTaken, LastLoss, BestEvaluationLoss, BestStep, StoppedEarly}`; evaluation also runs after the last step |
| `StepReport{Step, Loss, LearningRate, GradientNorm, StepTime}` | passed to `OnStep` |
| `EvaluationReport{Step, EvaluationLoss, BestEvaluationLoss, BestStep, IsBest, SavedBestModel, StepsWithoutImproving, AverageTrainingLoss, TrainingStepsSinceLast}` | passed to `OnEvaluation` |
| `EvaluationLossInBatches(trainer, sequences, batchSize)` | |

A loss that becomes NaN or infinite stops the loop with an error naming the step.

---

## chat

Chat templates and multi-turn conversations for instruct models.

| Name | What it does |
|---|---|
| `Message{Role, Content}` | `Role` is `system`, `user` or `assistant` |
| `LoadTemplate(modelFolder)` | reads `chat_template` from `tokenizer_config.json` and works out the style and default system prompt |
| `TemplateFromSource(source)` | the same, from the template text |
| `ChatML(defaultSystemPrompt)`, `Llama3()` | build a template directly (ChatML is used by SmolLM2 and Qwen) |
| `template.Format(messages, startAssistantReply)` | the exact text the model expects |
| `template.EndOfTurnText()` | `<|im_end|>` or `<|eot_id|>` |
| `NewConversation(model, tokenizer, template, systemPrompt)` | an empty system prompt uses the model's default |
| `conversation.Reply(userText, options, onNewText)` | generates a reply, calling `onNewText` with each new piece of text as it appears; only new tokens are processed each turn |
| `conversation.Messages`, `TokensReprocessed()` | the history, and how many tokens had to be processed from scratch |
| `ReplyOptions{MaximumNewTokens, Temperature, TopProbability, Sampling, Constraint}`, `DefaultReplyOptions()` | defaults 256, 0.2, 0.9; `Sampling` and `Constraint` as in `transformer.GenerationOptions` |
| `VocabularyBytes(tokenizer)` | every token's bytes (`TokenBytes`), the input the `constrained` package needs |

Templates other than ChatML and Llama 3 (for example Mistral's `[INST]`) are refused with an error.

Training data:

| Name | What it does |
|---|---|
| `ReadConversations(path)` | reads a `.jsonl` file of conversations, one per line |
| `ConversationFromJSON(record)` | one line: `{"messages": [{"role", "content"}, ...]}`, ShareGPT's `{"conversations": [{"from", "value"}, ...]}` (`human` and `gpt` become `user` and `assistant`), or any pair format `datafile.JSONRecord.Pair` knows, which becomes one user message and one reply |
| `ConversationFromPair(pair)` | a `datafile.Pair` as a user message and an assistant reply |
| `ExamplesFromConversation(messages, template, tokenizer)` | one `transformer.Example` per assistant message: the prompt is the formatted conversation up to that reply (ending with the assistant header), the answer is the reply plus the end-of-turn token. Only the replies count toward the loss, and the model learns to end its turn |
| `ExamplesFromConversations(conversations, template, tokenizer)` | the same for many |

```go
conversations, err := chat.ReadConversations("train.jsonl")
examples, err := chat.ExamplesFromConversations(conversations, template, textTokenizer)
model.TrainOnExamples(examples, adamW)
```

---

## tokenizer

Byte-level BPE (GPT-2, Llama 3, Qwen, SmolLM2) and SentencePiece (Llama 2, Mistral, TinyLlama, Phi-3), both in the same `*tokenizer.Tokenizer`, so everything else works the same with either.

| Function or method | What it does |
|---|---|
| `Train(text, vocabularySize)` | learn byte-level merges from text |
| `LoadHuggingFace(path)` | read a `tokenizer.json`: byte-level BPE, or SentencePiece-style BPE with `byte_fallback` and a Metaspace or Prepend + Replace normalizer |
| `LoadSentencePiece(path)` | read a SentencePiece `tokenizer.model` (BPE or Unigram) |
| `LoadFromFile(path)`, `t.SaveToFile(path)` | this library's own format, for both kinds |
| `t.Encode(text)`, `t.Decode(ids)` | safe to call from several goroutines at once |
| `t.Kind()` | `ByteLevelBPE`, `SentencePieceBPE` or `SentencePieceUnigram` |
| `t.TokenBytes(id)` | the exact bytes a token adds in the middle of generated text; nil for special tokens |
| `t.AddSpecialToken(content)`, `t.SpecialTokenID(content)` | tokens such as `<end>` that are never split |
| `t.VocabularySize()`, `t.TokenID(byteLevelToken)`, `t.TokenText(id)` | |

SentencePiece encoding matches SentencePiece and llama.cpp (checked against llama.cpp's own test cases): a leading "▁" at the start and after each special token, spaces become "▁", the highest-scoring merge goes first, and characters missing from the vocabulary become `<0xXX>` byte tokens (or `<unk>` without them). Unigram models that need a precompiled normalization map (T5) are refused with an error.

---

## safetensors

| Function or method | What it does |
|---|---|
| `Read(path)` | every tensor as `Tensor{Shape, Values []float64}` |
| `Open(path)`, `file.Names()`, `file.ReadTensor(name)`, `file.Close()` | read one tensor at a time |
| `Write(path, tensors)`, `WriteFloat64(path, tensors)` | F32 or F64 |
| `BFloat16ToFloat64(bits)`, `Float16ToFloat64(bits)` | |

Reads F64, F32, F16 and BF16.

---

## pretrained

| Function | What it does |
|---|---|
| `LoadLlama(folder)` | model and tokenizer from a Hugging Face folder, at Float64 |
| `LoadLlamaWithPrecision(folder, precision)` | built and filled directly in compressed form |
| `ReadLlamaConfig(path)`, `config.Check()`, `config.Settings()` | |
| `LoadLlamaWeights(model, config, folder)` | fill an existing model |
| `OurParameterName(huggingFaceName)` | e.g. `model.layers.0.self_attn.q_proj.weight` → `block1.attention.query.weights` |
| `LoadTokenizer(folder)` | `tokenizer.json` if the folder has one, otherwise `tokenizer.model` |
| `SaveModelAndTokenizer(path, model, tokenizer, storagePrecision)`, `LoadModelAndTokenizer(path)` | convert a downloaded model once (for example to Int8) and load it quickly afterwards |

Supports `model_type` `llama`, `mistral` and `qwen2`, single or sharded safetensors, and tied embeddings (loaded with `TieOutputToEmbedding`, so the table is kept once). Mistral's `sliding_window` is ignored, which matches Mistral v0.1 up to 4096 tokens. RoPE scaling, a separate `head_dim`, MLP biases and sliding-window configs are refused with a clear error.

---

## weightfile

| Function | What it does |
|---|---|
| `SaveBinary(path, parameters)`, `SaveText(path, parameters)` | `SaveBinary` stores Float64 |
| `SaveBinaryWithPrecision(path, parameters, precision)` | `StoragePrecision` is `Float64`, `Float32` or `BFloat16`; `precision.Round(value)` gives a value as it will come back |
| `ReadBinary(path)`, `ReadText(path)` | `map[name][]float64`; compressed entries are expanded |
| `ReadBinaryKeepingCompressed(path)` | plain values plus compressed rows, untouched |
| `LoadBinary(path, parameters)`, `LoadText(path, parameters)` | fill only the given parameters, so you can load one layer |
| `CopyInto(parameters, savedValues)`, `CopyIntoKeepingCompressed(parameters, savedValues, savedCompressedValues)` | checks every name and size before copying anything |

Files saved at Float64 with nothing compressed keep the original format byte for byte; anything else uses version 2, which records how each parameter is stored. Text files can't hold compressed weights. Checkpoints and optimizer state are always float64 so training resumes exactly.

---

## datafile

| Function or method | What it does |
|---|---|
| `ReadCSV(path, hasHeader)`, `ReadTSV`, `ReadSeparated(path, separator, hasHeader)` | a `Table{ColumnNames, Rows}` |
| `table.Column(name)`, `ColumnAsNumbers(name)`, `ColumnsAsMatrix(names)`, `ColumnsExcept(names)`, `ColumnIndex`, `NumberOfRows` | |
| `OneHot(labels)` | 0/1 matrix plus the sorted class names |
| `ReadPairs(path, inputColumn, targetColumn)`, `table.Pairs(...)` | question/answer pairs from CSV, TSV or `.jsonl` |
| `ReadText(path)`, `ReadLines(path)` | |
| `ReadTrainingText(path, textField)` | a `.jsonl` / `.ndjson` file's text fields joined by blank lines; any other file as plain text |

JSONL (one JSON object per line):

| Function or method | What it does |
|---|---|
| `ForEachJSONLRecord(path, handle)` | reads one line at a time and calls `handle(lineNumber, record)`, so the whole file never has to fit in memory |
| `ReadJSONL(path)` | every line as a `JSONRecord` (a `map[string]any`) |
| `ReadJSONLTexts(path, field)` | one text field from every line, for example `{"text": ...}` |
| `ReadJSONLPairs(path, inputField, targetField)` | pairs, see below |
| `record.Text(field)`, `record.HasTextFields(fields...)`, `record.FieldNames()` | |
| `record.Pair(inputField, targetField)` | uses the named fields when the line has them, otherwise recognises `instruction`/`input`/`output` (Alpaca, the `input` is added after the instruction), `prompt`/`completion`, `question`/`answer`, `input`/`target` and `input`/`output` |

Blank lines and Windows line endings are fine, lines can be any length, and errors name the file and line number, plus the fields a line does have when one is missing.

**Token files and streaming datasets.** Large corpora don't have to fit in memory. `TokenizeIntoTokenFile` reads a `TextSource` one document at a time, tokenizes on several goroutines and writes the token IDs in order to a compact file (2 bytes per token for vocabularies up to 65536, otherwise 4). `OpenTokenFile` reads chunks straight from disk, one `ReadAt` per chunk.

| Function or method | What it does |
|---|---|
| `CreateTokenFile(path, vocabularySize)` | a `TokenFileWriter`: `WriteTokens(ids)`, `EndDocument()`, `Close()`, `Discard()`; the file only appears, complete, on `Close` |
| `WriteTokenFile(path, tokenIDs, vocabularySize)` | in one go |
| `OpenTokenFile(path)` | a `TokenFile`: `NumberOfTokens()`, `VocabularySize()`, `ReadTokens(start, count)`, `RandomChunk(length)`, `RandomChunks(length, count)`, `SetRandomSeed(seed)`, `Close()` |
| `tokenFile.Split(evaluationFraction)`, `SplitLast(numberOfTokens)`, `View(start, count)` | views over the same file; training chunks never overlap evaluation chunks |
| `tokenFile.Chunks(length)`, `ChunksWithStep(length, step)` | walk in order: `Next()`, `NextBatch(count)`, `Restart()` (`io.EOF` at the end) |
| `TextSource` | anything with `NextText() (string, error)`, `io.EOF` at the end |
| `OpenTrainingTextSource(pathPatterns, textField)` | the streaming twin of `ReadTrainingText`, over files and globs |
| `OpenPlainTextSource(path, options)`, `OpenJSONLTextSource(path, field)`, `OpenJSONLRecordSource(path, recordToText)`, `OpenParquetTextSource(path, column)`, `OpenMultipleFileTextSource(paths, openOneFile)`, `NewSliceTextSource(texts)` | plain text in pieces or paragraphs, JSONL by field or through your own function (for example a chat template), Parquet one row group at a time |
| `TokenizeIntoTokenFile(source, newTokenizeFunction, path, vocabularySize, options)` | `StreamingTokenizeOptions{NumberOfWorkers, DocumentsPerBatch, AppendEndOfDocumentToken, EndOfDocumentTokenID, MarkDocumentEnds, ReportProgress}` |

```go
source, _ := datafile.OpenTrainingTextSource([]string{"corpus/*.jsonl"}, "text")
datafile.TokenizeIntoTokenFile(source, datafile.SameTokenizeFunctionForEveryWorker(textTokenizer.Encode), "corpus.tokens",
	textTokenizer.VocabularySize(), datafile.StreamingTokenizeOptions{AppendEndOfDocumentToken: true, EndOfDocumentTokenID: endID})

tokenFile, _ := datafile.OpenTokenFile("corpus.tokens")
trainingPart, evaluationPart, _ := tokenFile.Split(0.01)
chunks, _ := trainingPart.RandomChunks(129, 16)
```

---

## parquet

Reads Parquet files, such as Hugging Face dataset shards, in pure Go. It reads one row group at a time, so huge files can be streamed. Supports data pages v1 and v2, dictionary pages, PLAIN and dictionary encodings, optional (nullable) flat columns, and UNCOMPRESSED, SNAPPY, GZIP and ZSTD compression.

| Function or method | What it does |
|---|---|
| `Open(path)`, `file.Close()` | |
| `file.ColumnNames()`, `file.Columns()` | `Column{Name, PhysicalType, Optional, Repeated}` |
| `file.NumberOfRows()`, `NumberOfRowGroups()`, `NumberOfRowsInRowGroup(rowGroupIndex)` | |
| `file.ReadStringColumn(rowGroupIndex, column)` | BYTE_ARRAY and FIXED_LEN_BYTE_ARRAY |
| `file.ReadInt64Column`, `ReadFloat64Column`, `ReadBoolColumn` | same arguments |
| `file.ReadNullColumn(rowGroupIndex, column)` | which rows were null; nulls otherwise come back as `""`, `0` or `false` so rows stay lined up |
| `file.ForEachRowGroupOfStrings(column, handle)`, `ReadAllStrings(column)`, `ReadAllInt64s`, `ReadAllFloat64s` | |

`datafile.ReadParquetTexts(path, column)` and `datafile.OpenParquetTextSource(path, column)` connect it to training, and `-text data.parquet -field text` works in the example programs. Repeated or nested columns (lists such as `messages`), DELTA and BYTE_STREAM_SPLIT encodings, INT96, LZ4, Brotli and encrypted files return an error.

## compression

| Function | What it does |
|---|---|
| `DecompressSnappy(compressedBytes)` | the raw snappy block format |
| `DecompressZstd(compressedBytes)` | zstd frames, including several frames and skippable frames; checks checksums; no dictionaries |

---

## gpu

| Function or method | What it does |
|---|---|
| `ListDevices()` | `DeviceInfo{Index, Name, Kind}`, where Kind is discrete, integrated, virtual, cpu or other |
| `Open(index)`, `OpenBest()` | best prefers discrete, then integrated |
| `device.Close()`, `Name()`, `Info()`, `LastError()` | |
| `device.KeepWeightsOnGPU` | keep weights resident (call `vectormath.MarkWeightsChanged` after editing weights by hand) |
| `device.WeightCacheLimitBytes` | default 1 GB |
| `device.MinimumWorkForGPU`, `MinimumWorkForGPUWithResidentWeights`, `MinimumWorkForGPUWithFewRowsTimesUntransposedWeights` | below these, calls go to the CPU |
| `device.WeightUploads()`, `WeightCacheHits()`, `CachedWeightBytes()`, `ForgetCachedWeights()` | |

A `Device` is a `vectormath.Backend`; pass it to `vectormath.UseBackend`.

**General GPU compute** (what `gputraining` is built on):

| Name | What it does |
|---|---|
| `device.NewBuffer(numberOfFloats)` | a float32 buffer on the GPU; `Upload([]float64)`, `Download([]float64)`, `NumberOfFloats()`, `Free()` |
| `device.NewProgram(name, spirv, numberOfBuffers, pushConstantWords)` | a compute shader from compiled SPIR-V |
| `device.NewRecorder()` | records many GPU commands to send in one go |
| `recorder.Begin()`, `Upload`, `Fill`, `Copy`, `Download`, `Run(program, groupsAcross, groupsDown, groupsDeep, pushConstants, buffers...)` | record commands; each waits for the one before it |
| `recorder.Submit()` | run everything recorded and wait |
| `recorder.MeasureTime(on)`, `DispatchTimes()` | GPU timings for each command |
| `Float(value)`, `GroupsFor(threads, threadsPerGroup)` | helpers for push constants and group counts |

---

## gputraining

Trains a `transformer.Model` on the GPU in float32, one whole batch at a time.

| Name | What it does |
|---|---|
| `NewTrainer(device, model, options)` | uploads the model; returns an error listing every setting it doesn't support |
| `DefaultTrainerOptions(learningRate, weightDecay)` | AdamW with betas 0.9 / 0.95, the same as `optimizer.NewAdamW`; also set `options.Schedule` (an `optimizer.LearningRateSchedule`) and `options.MaximumGradientNorm` (0 = no clipping) |
| `trainer.TrainBatch(sequences)` | learn every next token; sequences may differ in length |
| `trainer.TrainOnExamples(examples)` | learn only the answers, like `model.TrainOnExamples` |
| `trainer.EvaluationLoss(sequences)`, `EvaluationLossOnExamples(examples)` | forward pass only, no dropout, nothing changes |
| `trainer.LearningRate()`, `LastGradientNorm()` | the rate the next step uses; the global gradient norm before clipping on the last step |
| `trainer.CopyWeightsToModel()` | bring the weights back to the CPU model, to generate or save |
| `trainer.SaveModel(path)` | copy the weights back and save |
| `trainer.SaveStateToFile(path)`, `LoadStateFromFile(path)` | AdamW moments and step count, to resume a run exactly (save the model too) |
| `trainer.UploadWeightsFromModel()`, `StepsTaken()`, `Close()` | |
| `NewDataParallelTrainer(devices, model, options)` | the same methods spread over several GPUs: each GPU takes part of the batch, the gradients are added together on the CPU, and every GPU applies the same update. `ResynchronizeEvery` (default 200 steps) copies the first GPU's weights to the others in case different GPUs round differently |

**Supported:** standard attention, rotary (any dimensions, base, pairing) or sine-wave positions, grouped-query attention, SwiGLU (optionally clamped), biases, `NormEpsilon`, weight tying, residual and attention dropout, freezing, learning rate schedules and gradient clipping.

**Not yet supported:** sliding windows, top-k, key/value sharing, attention sinks, query/key norm, low-rank queries, keys as values, compressed attention, mixture-of-experts, mHC, multi-token prediction, compressed weights and adapters.

Everything runs on the GPU, the token embedding too. The lookup is a gather kernel, and its gradient is added up per token without atomics, so a whole training step is one GPU submission. The gradient norm is worked out on the GPU and the AdamW kernel reads the clipping scale from there.

---

## gradientcheck

| Function | What it does |
|---|---|
| `Compare(forward, backward, parameters, inputs)` | returns a list of mismatches between `backward` and finite differences; empty means correct |

Use it to test your own layers:

```go
problems := gradientcheck.Compare(myLayer.Forward, myLayer.Backward, myLayer.Parameters(), inputs)
```
