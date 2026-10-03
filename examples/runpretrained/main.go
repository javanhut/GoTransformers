package main

import (
	"flag"
	"fmt"
	"github.com/javanhut/GoTransformers/gpu"
	"github.com/javanhut/GoTransformers/lowprecision"
	"github.com/javanhut/GoTransformers/pretrained"
	"github.com/javanhut/GoTransformers/tokenizer"
	"github.com/javanhut/GoTransformers/transformer"
	"github.com/javanhut/GoTransformers/vectormath"
	"github.com/javanhut/GoTransformers/weightfile"
	"os"
	"runtime"
	"runtime/debug"
	"time"
)

func memoryInUse() float64 {
	runtime.GC()
	debug.FreeOSMemory()
	var memory runtime.MemStats
	runtime.ReadMemStats(&memory)
	return float64(memory.HeapAlloc) / 1e9
}

func main() {
	modelFolder := flag.String("model", "", "folder with config.json, model.safetensors and tokenizer.json (a Hugging Face Llama or Qwen 2 model)")
	prompt := flag.String("prompt", "The capital of France is", "text to continue")
	numberOfTokens := flag.Int("tokens", 30, "how many new tokens to generate")
	temperature := flag.Float64("temperature", 0, "0 always picks the most likely token, higher values are more random")
	useGPU := flag.Bool("gpu", false, "use a GPU through Vulkan if one is found")
	precisionName := flag.String("precision", "Float32", "how to store the weights: Float64, Float32, Int8 or FP4")
	topK := flag.Int("top-k", 0, "only sample from this many most likely tokens (0 keeps them all)")
	topProbability := flag.Float64("top-p", 0, "only sample from the most likely tokens that add up to this probability (0 turns it off)")
	minimumProbability := flag.Float64("min-p", 0, "drop tokens less likely than this fraction of the most likely token")
	typicalProbability := flag.Float64("typical", 0, "locally typical sampling: keep the most typical tokens that add up to this probability (0 turns it off)")
	repetitionPenalty := flag.Float64("repetition-penalty", 1, "divide the scores of recently seen tokens by this (1 turns it off)")
	frequencyPenalty := flag.Float64("frequency-penalty", 0, "subtract this times the number of times a token was recently seen")
	presencePenalty := flag.Float64("presence-penalty", 0, "subtract this from every recently seen token")
	penaltyWindow := flag.Int("penalty-window", 64, "how many recent tokens the penalties look at (0 looks at all of them)")
	penalizePrompt := flag.Bool("penalize-prompt", false, "let the penalties also count the prompt tokens")
	seed := flag.Int64("seed", -1, "random seed for reproducible sampling (-1 picks a different one every run)")
	savePath := flag.String("save", "", "after loading, save the model (still compressed at -precision) and its tokenizer here, so -load can open it quickly next time")
	loadPath := flag.String("load", "", "load a model saved with -save instead of a Hugging Face folder")
	flag.Parse()

	if *modelFolder == "" && *loadPath == "" {
		fmt.Println("give a model folder with -model, for example one downloaded from https://huggingface.co/HuggingFaceTB/SmolLM2-135M")
		os.Exit(1)
	}
	var precision lowprecision.Precision
	if err := precision.UnmarshalText([]byte(*precisionName)); err != nil {
		fmt.Println(err, "- use Float64, Float32, Int8 or FP4")
		os.Exit(1)
	}
	if *useGPU {
		device, err := gpu.OpenBest()
		if err != nil {
			fmt.Println("no GPU, staying on the CPU:", err)
		} else {
			defer device.Close()
			vectormath.UseBackend(device)
		}
	}
	fmt.Println("doing math on:", vectormath.CurrentBackend().Name())

	loadStart := time.Now()
	var model *transformer.Model
	var loadedTokenizer *tokenizer.Tokenizer
	var err error
	if *loadPath != "" {
		model, loadedTokenizer, err = pretrained.LoadModelAndTokenizer(*loadPath)
	} else {
		model, loadedTokenizer, err = pretrained.LoadLlamaWithPrecision(*modelFolder, precision)
	}
	if err != nil {
		fmt.Println("could not load the model:", err)
		os.Exit(1)
	}
	if *savePath != "" {
		if err := pretrained.SaveModelAndTokenizer(*savePath, model, loadedTokenizer, weightfile.Float64); err != nil {
			fmt.Println("could not save the model:", err)
			os.Exit(1)
		}
		fmt.Println("saved the model to", *savePath, "- open it next time with -load", *savePath)
	}
	fmt.Printf("loaded in %s, weights stored as %v: %.0f MB of weights, %.2f GB of memory in use\n",
		time.Since(loadStart).Round(time.Millisecond), weightPrecisionOf(model, precision), float64(model.WeightBytes())/1e6, memoryInUse())
	fmt.Println("model:", model.Describe())

	promptIDs := loadedTokenizer.Encode(*prompt)
	readStart := time.Now()
	model.StartGenerating()
	model.Feed(promptIDs)
	readTime := time.Since(readStart)

	sampler := transformer.NewSampler(transformer.SamplingOptions{
		Temperature:        *temperature,
		TopK:               *topK,
		TopProbability:     *topProbability,
		MinimumProbability: *minimumProbability,
		TypicalProbability: *typicalProbability,
		RepetitionPenalty:  *repetitionPenalty,
		FrequencyPenalty:   *frequencyPenalty,
		PresencePenalty:    *presencePenalty,
		PenaltyWindow:      *penaltyWindow,
		PenalizePrompt:     *penalizePrompt,
		UseRandomSeed:      *seed >= 0,
		RandomSeed:         uint64(*seed),
	}, nil)
	sampler.RememberPrompt(promptIDs)

	generateStart := time.Now()
	generatedIDs := model.ContinueGeneratingWithSampler(*numberOfTokens, sampler)
	generateTime := time.Since(generateStart)

	fmt.Printf("\n%s%s\n\n", *prompt, loadedTokenizer.Decode(generatedIDs))
	fmt.Printf("prompt: %d tokens in %s, generated: %d tokens in %s (%.1f tokens per second)\n",
		len(promptIDs), readTime.Round(time.Millisecond), len(generatedIDs), generateTime.Round(time.Millisecond), float64(len(generatedIDs))/generateTime.Seconds())
}

func weightPrecisionOf(model *transformer.Model, requested lowprecision.Precision) lowprecision.Precision {
	if model.OutputLayer.IsCompressed() {
		return model.OutputLayer.CompressedWeights.Precision
	}
	if model.IsCompressed() {
		return requested
	}
	return lowprecision.Float64
}
