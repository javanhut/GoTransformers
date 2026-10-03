# Architecture

How GoTransformers is put together: the layers of packages, the rules every component follows, and how training, generation, the GPU and the file formats work.

## Design rules

- **Pure Go, one binary.** No cgo, no Python, no build-time SDKs. The only dependency is [purego](https://github.com/ebitengine/purego), which lets the `gpu` package call the Vulkan driver at run time.
- **Readable over clever.** Long descriptive names (`numberOfInputs`, `example`, `neuron`), plain index loops, small named functions. A function should read clearly without comments, so the code has none.
- **Optional acceleration.** Every operation works on the CPU. The GPU is something you switch on, never something you need.
- **Checked against numbers.** Every backward pass is tested against finite differences, and every model setup is tested to give the same output during training as during one-token-at-a-time generation.

## Package layers

Each package only imports packages from the layers above it.

```
1. math and storage
   vectormath        Vector, Matrix, CPU backend, Backend switch, random numbers
   parameter         Parameter: a named slice of weights plus where its gradients live
   lowprecision      Rows stored as Float64, Float32, Int8 or FP4

2. building blocks
   activationfunction   scalar activations, softmax
   lossfunction         mean squared error, softmax cross-entropy
   perceptron           Perceptron, Layer, MultiLayerPerceptron
   normalization        RMSNorm, LayerNorm
   embedding            token embeddings, vocabulary, sine-wave positions
   gradientcheck        finite-difference checker for any Forward/Backward pair
   dropout              random masking while training, off otherwise
   optimizer            SGD, momentum, Adam, AdamW, Muon, saveable state, learning rate schedules, gradient clipping
   compression          pure-Go snappy and zstd decoders

3. transformer parts
   feedforward          gated feed-forward (SwiGLU, GeGLU, ReGLU, clamped SwiGLU)
   attention            SelfAttention, CompressedAttention, generation caches
   mixtureofexperts     shared and routed experts
   hyperconnection      mHC residual streams

4. models
   transformer          Settings, Block, Model, Trainer, generation, checkpoints
   training             a training loop with held-out evaluation, best-model saving and early stopping, for any trainer

5. outside world
   tokenizer            byte-level BPE and SentencePiece (BPE, Unigram), Hugging Face tokenizer.json, tokenizer.model
   chat                 chat templates (ChatML, Llama 3), multi-turn conversations, conversation datasets
   safetensors          read and write safetensors files
   pretrained           load Llama, Mistral and Qwen 2 models from Hugging Face folders
   weightfile           save and load named weights as text or binary, at float64, float32 or bfloat16, compressed rows as they are
   parquet              read flat Parquet columns one row group at a time
   datafile             CSV, TSV, JSONL, text, Parquet, question/answer pairs, streaming text sources, on-disk token files
   gpu                  Vulkan backend for vectormath, and general GPU buffers, programs and recorders
   gputraining          float32 training of standard models entirely on the GPU, on one GPU or data-parallel on several
```

## Data conventions

**Vectors and matrices.** `vectormath.Vector` is `[]float64`. `vectormath.Matrix` stores `Rows × Columns` values in one row-major slice. `matrix.Row(i)` returns a view into that slice, so writing to it changes the matrix.

**Rows are tokens or examples.** Every component takes a matrix with one row per token (or per training example) and returns one row per token. A sequence of 48 tokens with vector size 64 is a 48×64 matrix. Generation uses the same code with 1-row matrices.

**Layer weights are one row per neuron.** `perceptron.Layer.Weights` is `outputs × inputs`, the same orientation as a single `Perceptron`'s weights repeated once per neuron, and the same as PyTorch's `nn.Linear`. A forward pass is `inputs × Weightsᵀ + biases`.

**Names.** Every trainable value belongs to a `parameter.Parameter` with a unique dotted name such as `block3.attention.query.weights`. Optimizers remember their state by name, and weight files store weights by name.

## Forward and backward

Every trainable component has the same shape:

```go
outputs := component.Forward(inputs)          // remembers what it needs
inputGradients := component.Backward(outputGradients)
parameters := component.Parameters()
```

- `Forward` stores what `Backward` needs (its inputs, activations, attention weights). Calling `Forward` again replaces that memory.
- `Backward` adds to the gradients (`+=`), so several backward passes can be summed. `parameter.ZeroGradients` clears them before a training step.
- Components that contain other components call them in order in `Forward` and in reverse order in `Backward`.

### Gradient memory

Gradients are only created when training needs them.

- `parameter.Parameter` has a `GradientStorage` pointer to the field where its owner keeps gradients, and a `Gradients()` method that creates them on first use.
- Layers and embeddings start with no gradient memory and create it in their first `Backward`.
- A parameter list fetched before `Backward` still sees the new gradients, because it reads through the pointer.
- `model.ReleaseGradients()` frees them after training. A model that's only used for generation never holds any.

## The CPU and the GPU

All heavy math goes through four functions in `vectormath`:

| Function | Used for |
|---|---|
| `MatrixTimesTransposedWeights(inputs, weights)` | layer forward pass |
| `MatrixTimesWeights(gradients, weights)` | layer backward pass, input gradients |
| `TransposedTimesMatrix(gradients, inputs)` | layer backward pass, weight gradients |
| `MatrixTimesMatrix`, `MatrixTimesTransposed` | everything else, including Muon |

These call whichever `vectormath.Backend` is active:

- **`CPUBackend`** (the default) splits rows across `GOMAXPROCS` threads, or splits columns when there are too few rows. That second case is what makes one-token generation use every core.
- **`gpu.Device`** loads the Vulkan library at run time (`libvulkan.so.1`, `vulkan-1.dll` or MoltenVK), compiles nothing, and runs one embedded SPIR-V shader for multiplies plus one for few-row multiplies. Values are float32 on the GPU. Calls smaller than `MinimumWorkForGPU` go back to the CPU because copying would cost more than it saves.
- **Weight cache.** When `device.KeepWeightsOnGPU` is on, weight matrices stay on the GPU between calls. They're uploaded again only when `vectormath.WeightsVersion()` changes. Every optimizer step, `weightfile.CopyInto` and `model.SetWeights` call `vectormath.MarkWeightsChanged()`. Code that edits weight values by hand must call it too, which is why the cache is off by default.

## The transformer model

`transformer.NewModel(settings)` builds:

```
token IDs
  → TokenEmbedding                         (+ sine-wave positions if rotary positions are off)
  → [expand to N streams]                  (only with mHC)
  → Block × NumberOfBlocks
  → [collapse streams]
  → FinalNorm → OutputLayer → scores, one row per token, one column per vocabulary entry
```

Each `Block`:

```
x → AttentionNorm → Attention → + x → FeedForwardNorm → FeedForward → + → out
```

With mHC (`NumberOfResidualStreams > 1`), each `+` becomes a `HyperConnection`. It mixes the streams with a learned doubly stochastic matrix, feeds a learned mix of them into the sublayer, and writes the output back into every stream.

- **`Block.Attention`** is an `AttentionLayer`: a `*attention.SelfAttention` or a `*attention.CompressedAttention`, chosen per block by `Settings.AttentionPattern`.
- **`Block.FeedForward`** is a `FeedForwardLayer`: SwiGLU (optionally clamped), or a `*mixtureofexperts.MixtureOfExperts` when `UseMixtureOfExperts` is on. Token IDs are passed down because the first blocks can route experts by token ID (hash routing).
- **Multi-token prediction** (`MultiTokenPrediction`) adds a small extra block that also predicts the token after next. During training its loss is added with weight `MultiTokenLossWeight`, and both predictions share `OutputLayer` by stacking their rows into one call.

### Attention

`attention.SelfAttention` has one core routine, `attendTo`, used by both training and generation, which is why the two give identical results. Around it:

- **Which positions a token looks at:** `HideFutureTokens` (causal), `WindowSize` (sliding window), `TopK` (only the k highest-scoring positions).
- **Shape of the cache:** `NumberOfKeyValueHeads` (grouped-query or multi-query attention) and `ShareKeyAsValue` (one entry is both key and value). Together they can shrink the cache 8× or more.
- **Positions:** rotary positions on all or the last `RotaryDimensions` of each head, with configurable base and Hugging Face's rotate-half pairing. When keys double as values, outputs are rotated back so they carry relative positions.
- **Stability:** `NormalizeQueriesAndKeys` (RMSNorm per head) and attention sinks (a learned extra logit per head that lets a head attend to nothing).
- **Sharing across layers:**

| Mode | DeepSeek name | Computes its own | Borrows from an earlier layer |
|---|---|---|---|
| `OwnKeysAndValues` | Full | keys, values, choices | nothing |
| `BorrowKeysAndValues` | Reindex | queries, choices | keys and values |
| `BorrowKeysValuesAndChoices` | Reuse | queries | keys, values and top-k choices |

  Borrowers send their key and value gradients back to the owner, so `Backward` must run in reverse order of `Forward`, as it always does in a model.

`attention.CompressedAttention` follows DeepSeek-V4:

- Every `CompressionRate` tokens are merged into one entry, using learned per-channel softmax weights and position biases. With `Overlap`, each entry also uses the previous block (CSA).
- A sliding window of uncompressed recent entries handles the token's own block.
- With `TopK > 0`, a lightning indexer scores the compressed entries and only the top k are used. The indexer gets no gradient from the main loss. It's trained in `Backward` to match where attention actually looked (a KL divergence weighted by `IndexerLossWeight`). `AttendToAllWhileTraining` is the dense warm-up stage.

### Generation and the cache

```go
model.StartGenerating()          // fresh caches
scores := model.Feed(promptIDs)  // one token at a time
nextID := transformer.PickToken(scores, temperature)
scores = model.NextTokenScores(nextID)
```

- `SelfAttention` keeps keys and values in `lowprecision.Rows` at `CachePrecision`. With a window, old rows are dropped so the cache stays bounded. `TrainAtCachePrecision` rounds keys and values the same way during training, so the model learns to cope with the rounding.
- `CompressedAttention` keeps compressed entries, their indexer keys, the window, and the uncompressed tail of the current block.
- `model.SaveGenerationState(path)` and `LoadGenerationState(path)` write all of that to disk with a fingerprint of the model's settings and weights, so a long prompt is processed once and reused later.

**Chat.** `chat.Conversation` builds the full prompt with the model's template every turn. It compares that text with what it has already fed the model, and feeds only the new part, so the cache carries over and each turn costs only its new tokens. If the text ever stops matching (for example after editing the history), it starts the cache again from the beginning.

### Compressed weights

`model.CompressWeights(precision)` (or `Settings.WeightPrecision`, or `pretrained.LoadLlamaWithPrecision`) replaces each layer's float64 `Weights` with `CompressedWeights`, a `lowprecision.Rows`.

- The forward pass uses `Rows.DotRow`, which reads Float32, Int8 or FP4 directly without expanding the weights.
- Int8 and FP4 store one float32 scale per 16 values.
- Compressed parameters report themselves as `ReadOnly` with nil `Values`, so optimizers and weight saving refuse them with a clear message. `DecompressWeights()` turns training back on.
- When `Settings.WeightPrecision` is set, `NewModel` compresses each block as soon as it's built, and the loader writes each tensor straight into compressed storage, so a full float64 copy never exists.

## Training

```go
model.TrainBatch(sequences, chosenOptimizer)             // learn every next token
model.TrainOnExamples(examples, chosenOptimizer)         // learn only the answer part
```

Both run the same steps:
1. Zero the gradients.
2. For each sequence, run forward and backward. The gradients add up across the batch.
3. Divide the gradients by the batch size.
4. Update every parameter whose name doesn't start with a prefix passed to `Freeze`.
5. Nudge the expert-balance biases.

`transformer.Trainer` runs the same steps with two more between 3 and 4: it clips the gradients to `MaximumGradientNorm` (one global norm over every trainable parameter), and sets the optimizer's learning rate to the schedule's fraction of the peak rate for this step. The peak rates are read from the optimizer when the trainer is made, so Muon's two rates keep their ratio. `training.Loop` drives any trainer: it asks for a batch, takes a step, and every `EvaluateEvery` steps measures the held-out loss and saves the model if it is the best so far.

**Weight tying.** With `TieOutputToEmbedding` the output layer's `Weights` is the embedding's `Table`: the same slice, because the table is vocabulary × vector and the output weights are outputs × inputs, which is the same layout. The output layer still computes its own weight gradients in `Backward`; the model adds them into the table's gradients right after, and zeroes them. `Parameters()` lists only `tokens.table`, so optimizers and files see one matrix.

The loss counts only the positions predicting answer tokens, which is how answer-only fine-tuning works; plain text counts every position.

**Chat datasets.** `chat.ExamplesFromConversations` turns each assistant reply into one example. The prompt is `template.Format(messagesBeforeTheReply, true)`, the same text `Conversation.Reply` feeds the model, so training and chatting see identical tokens. A test checks that prompt + answer tokens equal the tokens of the whole formatted conversation. The answer ends with the end-of-turn token, which is what teaches the model to stop. A conversation with several replies gives several examples, each with the whole history before its reply as the prompt.

**Freezing saves work.** Before each step, layers whose weights are all frozen get `FreezeBase`. Their backward pass still passes gradients down to earlier layers, but skips computing and storing their own weight gradients. A frozen embedding table skips its backward pass entirely.

**Dropout.** Three settings: `ResidualDropout` (each block's attention and feed-forward outputs, before they're added back), `AttentionDropout` (the attention weights, inside both attention types) and `AdapterDropout` (LoRA inputs).
- Every dropout starts switched off. The model switches them all on while it computes gradients and off again afterwards. `Forward`, generation, scoring, chat and the loss functions without gradients never drop anything, which is why generation still matches training-mode output exactly.
- Each dropout remembers the mask it used so the backward pass applies the same one. For gradient checks, `RepeatLastMasks` replays the previous pass's masks so the loss stops being random.
- The masks come from the shared random generator, so checkpoints still resume exactly.

**LoRA.** `model.AddLowRankAdapters(rank, alpha)` gives every block layer an adapter: two small matrices, `Down` (rank × inputs) and `Up` (outputs × rank).
- The layer's output becomes `x × Wᵀ + b + (alpha / rank) × x × Downᵀ × Upᵀ`. `Up` starts at zero, so adding adapters changes nothing until training.
- Everything except the adapters is frozen, so the optimizer only keeps state for about 1% of the values.
- The base weights may be compressed: the backward pass reads Int8 or FP4 rows directly to pass gradients to earlier layers (the "QLoRA" approach).
- Adapters save to their own small file (`SaveAdapters`), or merge into uncompressed weights (`MergeAdapters`). Their rank and alpha live in `Settings`, so `Save` / `LoadModel` keep them.

**Optimizers.**
- `Muon` uses hybrid Newton-Schulz orthogonalization on weight matrices, and an inner `AdamW` for vectors and anything marked `UseAdamW` (embeddings, the output layer, norms, mHC biases).
- `Adam` and `AdamW` keep running averages by parameter name.
- Every optimizer can save and restore its state, which is what lets `SaveCheckpoint` / `LoadCheckpoint` resume a run exactly: they save the weights, optimizer state, frozen prefixes, step count and random generator state.

## Training on the GPU

`gputraining.Trainer` is a second, separate implementation of the training step for standard models. It works in float32, with data kept on the GPU:

- **Weights live on the GPU** as `gpu.Buffer`s, along with AdamW's running averages. They come back to the CPU model only when you call `CopyWeightsToModel`.
- **A batch is one big computation.** Sequences are padded to the same length and stacked, so every matrix multiply covers the whole batch. The padding is masked out of the loss, and causal attention means it can't affect real tokens.
- **One submit per step.** The forward pass, backward pass, loss and AdamW updates of every layer are recorded with a `gpu.Recorder` and sent to the GPU together, avoiding a round trip after every operation.
- **About 20 small shaders** (`gputraining/shaders`): strided batched matrix multiply, bias and column sums, RMSNorm forward and backward, RoPE, causal softmax and its backward pass, summing grouped-query heads, SwiGLU forward and backward, residual add, masked cross-entropy, embedding gather and gradient scatter, squared sums and the clipping scale, dropout, and AdamW. Each is tested against a float64 CPU version.
- **The embedding is on the GPU too.** The token IDs go up as floats and a gather kernel builds the first block's input (adding sine-wave positions when rotary positions are off). For the backward pass the CPU sorts the batch's rows by token once, and the scatter kernel gives each (token, column) pair one thread that adds up that token's rows, so no two threads write the same value and no atomics are needed. With weight tying the output layer's weights are the table buffer: the output layer's backward writes the table gradients and the scatter adds to them.
- **Clipping without a round trip.** One kernel adds up the squares of every trainable gradient buffer into partial sums, a second turns them into the norm and the scale `min(1, maximum / norm)`, and the AdamW kernel multiplies each gradient by that scale as it reads it. The norm is read back with the loss.
- **Dropout without stored masks.** The mask comes from a hash of a per-step seed, the block, the place and the element index, so the backward pass rebuilds exactly the same mask instead of keeping it in memory. Attention dropout keeps the undropped probabilities for the softmax backward pass and rebuilds the dropped ones in a shared buffer.
- **Several GPUs.** `DataParallelTrainer` keeps one `Trainer` per GPU, all starting from the same weights. Each step splits the batch, each GPU computes its gradients (the loss weights use the whole batch's size, so the gradients simply add up), the CPU adds them together, and every GPU uploads the sum and runs the same clipping and AdamW. A periodic copy of the first GPU's weights keeps them from drifting apart.
- **Checked against the CPU path:** one GPU step matches `transformer.Model.TrainBatch` with `optimizer.AdamW`. The loss agrees to about 1e-7 relative, and gradients to about 2e-6 of the largest gradient.

On the Intel Meteor Lake integrated GPU, a 6.9M-parameter model trained at about 6,900 tokens/s with the embedding on the CPU, against 268 for the CPU path. Moving the embedding onto the GPU took a 6-block, 256-wide model from 7,800 to 10,400 tokens/s on the same GPU.

## File formats

| File | Written by | Format |
|---|---|---|
| `name.weights` | `weightfile.SaveBinary`, `model.Save` | `GOTRANSFORMERS-WEIGHTS-1`, then for each parameter: name length, name, value count, little-endian float64 values |
| `name.weights` | `model.SaveWithPrecision`, or any model with compressed weights | `GOTRANSFORMERS-WEIGHTS-2`: after each name, one byte for how it is stored (float64, float32, bfloat16, or compressed rows with their precision, shape, values and scales) |
| token file | `datafile.CreateTokenFile`, `TokenizeIntoTokenFile` | 64-byte header (`GOTOKENS`, version, bytes per token, vocabulary size, token count, document count, document table offset), then 2- or 4-byte little-endian token IDs, then optional document ends |
| `name.txt` | `weightfile.SaveText` | `name count` then the values as text, separated by spaces |
| `name.weights.settings.json` | `model.Save` | the model's `Settings` as JSON, with readable names for enums |
| checkpoint folder | `model.SaveCheckpoint` | `model.weights` (+ settings), `optimizer.state` (gob), `progress.json` |
| generation state | `model.SaveGenerationState` | gob, with format tag `gotransformers-generation-1` and a model fingerprint |
| tokenizer | `tokenizer.SaveToFile` | text, format tag `gotransformers-tokenizer-2` (adds the kind and a SentencePiece section; `-1` files still load) |
| vocabulary | `Vocabulary.SaveToFile` | one quoted token per line |
| `*.safetensors` | `safetensors.Write` | the standard safetensors layout (F32 or F64 out; F64, F32, F16 and BF16 in) |
| `*.jsonl`, `*.ndjson` | read only (`datafile`, `chat`) | one JSON object per line: `{"text"}`, pairs (`prompt`/`completion`, `instruction`/`input`/`output`, `question`/`answer`, `input`/`target`) or conversations (`messages`, ShareGPT `conversations`) |

## Testing

- **Gradient checks:** `gradientcheck.Compare(forward, backward, parameters, inputs)` nudges every weight and input by 1e-6 and compares the numerical slope with what `Backward` computed. Every layer, norm, attention option, compression setup, expert routing, mHC and the whole model are checked this way.
- **Generation matches training:** each attention setup and model setup runs a sequence all at once and then one token at a time through the cache; the outputs must agree to 1e-9.
- **Mutation checks:** while building, a bug was put in on purpose for each major piece to confirm the tests catch it.
- **Real model:** with `GOTRANSFORMERS_SMOLLM2` set, the loaded SmolLM2-135M is compared against an independent plain-loop Go implementation in the test file (agreement to about 1e-13), and the tokenizer is checked against the model's own vocabulary.
