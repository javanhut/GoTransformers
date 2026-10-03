# GoTransformers

A deep learning and transformer library written in pure Go, built for making and running small language models on ordinary computers.

- **One binary.** No Python, no CUDA toolkit, no C compiler. `CGO_ENABLED=0 go build` works.
- **Any GPU or none.** Heavy matrix math can run on NVIDIA, AMD or Intel GPUs through Vulkan, which is loaded while the program runs. Without a GPU everything runs on all CPU cores.
- **Built from research.** The architecture pieces come from recent papers (DeepSeek-V4, DeepSeek-V4.1, and others), scaled down so they make sense for small local models.
- **Readable.** Long, plain names and simple loops, so you can read a function and say what it does and what shape the data has.

## Installing

```
go get github.com/javanhut/GoTransformers@latest
```

Then import the packages you need:

```go
import (
	"github.com/javanhut/GoTransformers/chat"
	"github.com/javanhut/GoTransformers/pretrained"
	"github.com/javanhut/GoTransformers/transformer"
)
```

## What's in it

| Area | What you get |
|---|---|
| Basics | Vectors and matrices, 19 activation functions plus softmax, perceptrons, layers, multi-layer perceptrons, mean squared error and softmax cross-entropy |
| Attention | Multi-head self-attention with a key/value cache, sliding windows, top-k, rotary positions (full or partial), grouped-query and multi-query attention, keys that double as values, query/key normalization, attention sinks, low-rank queries, cross-layer key/value sharing |
| Compressed attention | DeepSeek-V4 style heavily compressed (HCA) and compressed sparse (CSA) attention with a lightning indexer |
| Model parts | RMSNorm, LayerNorm, SwiGLU / GeGLU / ReGLU (with optional clamping), mixture-of-experts, mHC residual streams, multi-token prediction |
| Training | SGD, momentum, Adam, AdamW, Muon; learning rate warmup and decay (cosine, linear, warmup-stable-decay); gradient clipping; a training loop with held-out evaluation, best-model saving and early stopping; batches; dropout; weight tying; answer-only fine-tuning; freezing layers; LoRA adapters (also over compressed Int8/FP4 weights); checkpoints that resume exactly |
| Running models | Generation with a cache, top-k, top-p, min-p and typical sampling, repetition/frequency/presence penalties, seeded sampling, output constrained to valid JSON, a JSON schema or a list of choices, chat templates and multi-turn conversations, cache saved to disk, answer scoring, "I don't know" when confidence is low, weights compressed to Float32, Int8 or FP4 |
| Files | Byte-level BPE tokenizer (train your own or load Hugging Face `tokenizer.json`), SentencePiece tokenizers (`tokenizer.model`, Llama 2 / Mistral style `tokenizer.json`), safetensors, Llama, Mistral and Qwen 2 models from Hugging Face, CSV/TSV/JSONL/text/Parquet data (including chat datasets in `messages`, ShareGPT, Alpaca and prompt/completion form), pure-Go snappy and zstd, on-disk token files for datasets bigger than memory, weight files stored as float64, float32 or bfloat16, compressed models saved as they are |
| Hardware | Multi-threaded CPU math, Vulkan GPU backend with an optional weight cache, float32 training entirely on the GPU, embedding included (about 10,400 training tokens/s for a 6-block, 256-wide model on an Intel integrated GPU), data-parallel training over several GPUs |

## Getting started

You need Go 1.27.1 or newer.

```
go test ./...
go run ./examples/tinylanguagemodel
```

### Examples

| Command | What it shows |
|---|---|
| `go run .` | A single perceptron |
| `go run ./examples/xor` | A multi-layer perceptron learning XOR, saved to a file and loaded back |
| `go run ./examples/nextcharacter` | Building a tiny language model by hand from an embedding, attention and a layer |
| `go run ./examples/tinylanguagemodel` | Training a character-level transformer on any text file |
| `go run ./examples/finetune` | Fine-tuning on question/answer pairs and answering "I don't know" when unsure |
| `go run ./examples/runpretrained -model <folder>` | Running a real open model downloaded from Hugging Face |
| `go run ./examples/chat -model <folder>` | Chatting with an instruct model in the terminal |
| `go run ./examples/lora -model <folder>` | Teaching an instruct model new facts with LoRA adapters over Int8 weights |
| `go run ./examples/gpucheck` | Listing GPUs and timing CPU against GPU |
| `go run ./examples/gputrain -text book.txt -tokens bpe` | Training a model from scratch on the GPU |
| `go run ./examples/gputrain -text data.parquet -field text -tokens bpe -token-file data.tokens -gpus all` | The same on a Hugging Face dataset, streamed to disk, on every GPU |

Useful flags for `tinylanguagemodel`:

```
go run ./examples/tinylanguagemodel -text book.txt -steps 2000 -batch 8 -optimizer muon -gpu
go run ./examples/tinylanguagemodel -deepseek                 # compressed attention, experts, mHC, multi-token prediction
go run ./examples/tinylanguagemodel -checkpoint run1          # saves every 100 steps; run it again to resume
```

`gputrain` warms the learning rate up over the first twentieth of the steps, then lowers it along a cosine (`-schedule cosine|linear|wsd|none`, `-warmup`, `-final-fraction`), clips gradients at norm 1 (`-clip`), holds back 2% of the tokens and reports their loss every 100 steps (`-eval-fraction`, `-eval-every`), keeps the best model (`-best best.weights`), ties the embedding and output weights (`-tie`), and saves float32 weights (`-save-precision`).

### Running a real model

Download `config.json`, `model.safetensors` and `tokenizer.json` from [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) into a folder, then:

```
go run ./examples/runpretrained -model ./SmolLM2-135M -prompt "Once upon a time" -precision Int8
```

Llama-architecture and Qwen 2 models with byte-level BPE tokenizers load the same way. On a 14-core laptop CPU, SmolLM2-135M gives:

| `-precision` | Memory | Speed |
|---|---|---|
| Float64 | 1.09 GB | 20 tokens/s |
| Float32 | 0.55 GB | 24 tokens/s, identical output |
| Int8 | 0.18 GB | 27 tokens/s |
| FP4 | 0.11 GB | 15 tokens/s |

Float32 is lossless for models published in bfloat16. SmolLM2 ties its embedding and output weights, so the table is kept once.

## Using it in code

Train a small transformer:

```go
textTokenizer := tokenizer.Train(text, 2000)
tokenIDs := textTokenizer.Encode(text)

model, err := transformer.NewModel(transformer.SmallSettings(textTokenizer.VocabularySize()))
if err != nil {
	panic(err)
}
adam := optimizer.NewAdam(0.003)
for step := 0; step < 1000; step++ {
	loss := model.TrainBatch(transformer.RandomChunks(tokenIDs, 64, 8), adam)
	fmt.Println(step, loss)
}

generated := model.Generate(textTokenizer.Encode("Once upon"), 50, 0.7)
fmt.Println(textTokenizer.Decode(generated))
```

Fine-tune on answers only and refuse to guess:

```go
example := transformer.Example{PromptIDs: questionIDs, AnswerIDs: answerIDs}
model.TrainOnExamples([]transformer.Example{example}, adam)

answer := model.Answer(questionIDs, transformer.GenerationOptions{
	MaximumNewTokens: 30,
	StopTokenIDs:     []int{endID},
	ConfidenceTarget: 0.5,
	AbstainTokenIDs:  textTokenizer.Encode("I don't know"),
})
```

Fine-tune an instruct model on a JSONL chat dataset (one `{"messages": [...]}` per line). Only the assistant's replies are learned:

```go
conversations, err := chat.ReadConversations("train.jsonl")
examples, err := chat.ExamplesFromConversations(conversations, template, modelTokenizer)
model.TrainOnExamples(examples, adamW)
```

Every example program that reads data takes `.jsonl` too: `-text data.jsonl` (with `-field` for the text field, default `text`) for `tinylanguagemodel` and `gputrain`, and `-pairs data.jsonl` for `finetune` and `lora`.

Force a reply to be valid JSON (small models can't be trusted to write it on their own), with better sampling:

```go
jsonSettings := constrained.DefaultJSONSettings()
jsonSettings.RequireObjectAtTopLevel = true
reply, err := conversation.Reply("Describe a cat called Tom as JSON.", chat.ReplyOptions{
	MaximumNewTokens: 200,
	Temperature:      0.7,
	Sampling:         transformer.SamplingOptions{TopK: 40, MinimumProbability: 0.05, RepetitionPenalty: 1.1},
	Constraint:       constrained.NewJSONConstraint(chat.VocabularyBytes(modelTokenizer), jsonSettings),
}, nil)
```

On the command line: `go run ./examples/chat -model SmolLM2-360M-Instruct -json -top-k 40 -min-p 0.05 -repetition-penalty 1.1`.

Load an open model with compressed weights:

```go
model, modelTokenizer, err := pretrained.LoadLlamaWithPrecision("./SmolLM2-135M", lowprecision.Int8)
```

Save it compressed once and load it in a couple of seconds afterwards (SmolLM2-135M in Int8 is a 170 MB file): `go run ./examples/runpretrained -model ./SmolLM2-135M -precision Int8 -save smollm2.int8`, then `-load smollm2.int8`.

Use a GPU:

```go
device, err := gpu.OpenBest()
if err == nil {
	defer device.Close()
	vectormath.UseBackend(device)
}
```

Train a whole model on the GPU:

```go
trainer, err := gputraining.NewTrainer(device, model, gputraining.DefaultTrainerOptions(0.001, 0.1))
if err != nil {
	panic(err)
}
defer trainer.Close()
for step := 0; step < 5000; step++ {
	loss := trainer.TrainBatch(transformer.RandomChunks(tokenIDs, 128, 16))
	fmt.Println(step, loss)
}
trainer.CopyWeightsToModel()
model.SaveWithPrecision("story-model.weights", weightfile.Float32)
textTokenizer.SaveToFile("story-model.weights.tokenizer")
```

Warm up and decay the learning rate, clip gradients, watch held-out loss and keep the best model (the same works with `transformer.NewTrainer` on the CPU, and with `gputraining.NewDataParallelTrainer(devices, ...)` over several GPUs):

```go
settings.TieOutputToEmbedding = true
options := gputraining.DefaultTrainerOptions(0.001, 0.1)
options.Schedule = optimizer.WarmupThenCosine(250, 5000, 0.1)
options.MaximumGradientNorm = 1
trainer, err := gputraining.NewTrainer(device, model, options)

trainingIDs, heldOutIDs := tokenIDs[:len(tokenIDs)*98/100], tokenIDs[len(tokenIDs)*98/100:]
result, err := training.Loop{
	Trainer:             trainer,
	NumberOfSteps:       5000,
	NextBatch:           func() [][]int { return transformer.RandomChunks(trainingIDs, 129, 16) },
	EvaluationSequences: transformer.RandomChunks(heldOutIDs, 129, 32),
	EvaluateEvery:       200,
	BestModelPath:       "best.weights",
}.Run()
```

Train on more text than fits in memory by tokenizing it to a file once:

```go
source, _ := datafile.OpenTrainingTextSource([]string{"shards/*.parquet"}, "text")
datafile.TokenizeIntoTokenFile(source, datafile.SameTokenizeFunctionForEveryWorker(textTokenizer.Encode),
	"corpus.tokens", textTokenizer.VocabularySize(), datafile.StreamingTokenizeOptions{})
tokenFile, _ := datafile.OpenTokenFile("corpus.tokens")
chunks, _ := tokenFile.RandomChunks(129, 16)
```

Turn on the DeepSeek-V4 style architecture:

```go
settings := transformer.DeepSeekStyleSettings(vocabularySize)
```

See [REFERENCE.md](REFERENCE.md) for every package and function, and [ARCHITECTURE.md](ARCHITECTURE.md) for how the pieces fit together.

## Testing

```
go test ./...
go test -race ./...
CGO_ENABLED=0 go build ./...
```

Every layer's backward pass is checked against numerical estimates, and every attention and model setup is checked to give the same output whether tokens are processed all at once or one at a time through the cache. To also run the tests against a real model, point `GOTRANSFORMERS_SMOLLM2` at a folder holding SmolLM2-135M:

```
GOTRANSFORMERS_SMOLLM2=./SmolLM2-135M go test ./pretrained ./tokenizer
```

Those tests compare the model's output with an independent, deliberately simple Go implementation (they agree to within 1e-13) and check the tokenizer against the model's vocabulary.

## Papers used

- DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression (arXiv 2609.19969)
- DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence (arXiv 2606.19348)
- DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference (arXiv 2602.21548)
- Small Models, Big Results: Achieving Superior Intent Extraction through Decomposition (arXiv 2509.12423)
- Why Language Models Hallucinate (arXiv 2509.04664)

## Current limits

- CPU training computes in float64 (files can be stored as float32 or bfloat16). Compressed weights are for running models only.
- GPU training covers standard models only (no compressed attention, experts, mHC, multi-token prediction or adapters yet), and has only been tested on an Intel integrated GPU. Data-parallel training has been checked with two handles on that one GPU, not on two real cards.
- Text weight files can't hold compressed weights; use the binary format. Checkpoints and optimizer state are always float64.
- Parquet: flat columns only (no nested `messages` lists yet), no LZ4 or Brotli.
- On the CPU path, batches are processed one sequence at a time. The GPU trainer processes the whole batch at once.
- Only Llama, Mistral and Qwen 2 style models load. SentencePiece Unigram tokenizers that need a precompiled normalization map (T5) aren't supported.
