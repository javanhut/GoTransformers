# Reference

Every package and its public API. Packages are imported as `transformer/<package>`. See [ARCHITECTURE.md](ARCHITECTURE.md) for how they fit together.

**Contents:** [vectormath](#vectormath) · [parameter](#parameter) · [lowprecision](#lowprecision) · [activationfunction](#activationfunction) · [lossfunction](#lossfunction) · [dropout](#dropout) · [perceptron](#perceptron) · [normalization](#normalization) · [embedding](#embedding) · [feedforward](#feedforward) · [attention](#attention) · [mixtureofexperts](#mixtureofexperts) · [hyperconnection](#hyperconnection) · [optimizer](#optimizer) · [transformer](#transformer) · [chat](#chat) · [tokenizer](#tokenizer) · [safetensors](#safetensors) · [pretrained](#pretrained) · [weightfile](#weightfile) · [datafile](#datafile) · [gpu](#gpu) · [gputraining](#gputraining) · [gradientcheck](#gradientcheck)

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
| `model.Save(path)`, `LoadModel(path)` | weights at `path`, settings at `path.settings.json` |
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

```go
type GenerationOptions struct {
	MaximumNewTokens int
	Temperature      float64
	StopTokenIDs     []int      // stop on any of these (not included in the answer)
	ConfidenceTarget float64    // 0 = never abstain
	AbstainTokenIDs  []int      // returned instead when confidence is too low
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
}
```

### Parts

| Name | What it is |
|---|---|
| `Block` | `AttentionNorm`, `Attention`, `FeedForwardNorm`, `FeedForward`, and optional `AttentionConnection` / `FeedForwardConnection` for mHC |
| `AttentionLayer`, `FeedForwardLayer` | interfaces a block's parts satisfy, so you can plug in your own |
| `MultiTokenPredictor` | the extra block that predicts the token after next |

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
| `ReplyOptions{MaximumNewTokens, Temperature, TopProbability}`, `DefaultReplyOptions()` | defaults 256, 0.2, 0.9 |

Templates other than ChatML and Llama 3 (for example Mistral's `[INST]`) are refused with an error.

---

## tokenizer

Byte-level BPE, so no text is ever unknown.

| Function or method | What it does |
|---|---|
| `Train(text, vocabularySize)` | learn merges from text |
| `LoadHuggingFace(path)` | read a `tokenizer.json` (GPT-2, Llama 3, Qwen 2 style byte-level BPE) |
| `LoadFromFile(path)`, `t.SaveToFile(path)` | this library's own format |
| `t.Encode(text)`, `t.Decode(ids)` | |
| `t.AddSpecialToken(content)`, `t.SpecialTokenID(content)` | tokens such as `<end>` that are never split |
| `t.VocabularySize()`, `t.TokenID(byteLevelToken)`, `t.TokenText(id)` | |

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

Supports `model_type` `llama` and `qwen2`, single or sharded safetensors, and tied embeddings. RoPE scaling, a separate `head_dim`, MLP biases and sliding-window configs are refused with a clear error.

---

## weightfile

| Function | What it does |
|---|---|
| `SaveBinary(path, parameters)`, `SaveText(path, parameters)` | |
| `ReadBinary(path)`, `ReadText(path)` | `map[name][]float64` |
| `LoadBinary(path, parameters)`, `LoadText(path, parameters)` | fill only the given parameters, so you can load one layer |
| `CopyInto(parameters, savedValues)` | checks every name and size before copying anything |

---

## datafile

| Function or method | What it does |
|---|---|
| `ReadCSV(path, hasHeader)`, `ReadTSV`, `ReadSeparated(path, separator, hasHeader)` | a `Table{ColumnNames, Rows}` |
| `table.Column(name)`, `ColumnAsNumbers(name)`, `ColumnsAsMatrix(names)`, `ColumnsExcept(names)`, `ColumnIndex`, `NumberOfRows` | |
| `OneHot(labels)` | 0/1 matrix plus the sorted class names |
| `ReadPairs(path, inputColumn, targetColumn)`, `table.Pairs(...)` | question/answer pairs |
| `ReadText(path)`, `ReadLines(path)` | |

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
| `DefaultTrainerOptions(learningRate, weightDecay)` | AdamW with betas 0.9 / 0.95, the same as `optimizer.NewAdamW` |
| `trainer.TrainBatch(sequences)` | learn every next token; sequences may differ in length |
| `trainer.TrainOnExamples(examples)` | learn only the answers, like `model.TrainOnExamples` |
| `trainer.CopyWeightsToModel()` | bring the weights back to the CPU model, to generate or save |
| `trainer.UploadWeightsFromModel()`, `StepsTaken()`, `Close()` | |

**Supported:** standard attention, rotary (any dimensions, base, pairing) or sine-wave positions, grouped-query attention, SwiGLU (optionally clamped), biases, `NormEpsilon`.

**Not yet supported:** sliding windows, top-k, key/value sharing, attention sinks, query/key norm, low-rank queries, keys as values, compressed attention, mixture-of-experts, mHC, multi-token prediction, compressed weights, adapters and dropout.

The token embedding and its AdamW stay on the CPU; everything else runs on the GPU.

---

## gradientcheck

| Function | What it does |
|---|---|
| `Compare(forward, backward, parameters, inputs)` | returns a list of mismatches between `backward` and finite differences; empty means correct |

Use it to test your own layers:

```go
problems := gradientcheck.Compare(myLayer.Forward, myLayer.Backward, myLayer.Parameters(), inputs)
```
