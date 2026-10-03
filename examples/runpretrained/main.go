package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"time"
	"transformer/gpu"
	"transformer/pretrained"
	"transformer/vectormath"
)

func main() {
	modelFolder := flag.String("model", "", "folder with config.json, model.safetensors and tokenizer.json (a Hugging Face Llama or Qwen 2 model)")
	prompt := flag.String("prompt", "The capital of France is", "text to continue")
	numberOfTokens := flag.Int("tokens", 30, "how many new tokens to generate")
	temperature := flag.Float64("temperature", 0, "0 always picks the most likely token, higher values are more random")
	useGPU := flag.Bool("gpu", false, "use a GPU through Vulkan if one is found")
	flag.Parse()

	if *modelFolder == "" {
		fmt.Println("give a model folder with -model, for example one downloaded from https://huggingface.co/HuggingFaceTB/SmolLM2-135M")
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
	model, loadedTokenizer, err := pretrained.LoadLlama(*modelFolder)
	if err != nil {
		fmt.Println("could not load the model:", err)
		os.Exit(1)
	}
	var memory runtime.MemStats
	runtime.ReadMemStats(&memory)
	fmt.Printf("loaded in %s, using %.1f GB of memory\n", time.Since(loadStart).Round(time.Millisecond), float64(memory.HeapAlloc)/1e9)
	fmt.Println("model:", model.Describe())

	promptIDs := loadedTokenizer.Encode(*prompt)
	readStart := time.Now()
	model.StartGenerating()
	model.Feed(promptIDs)
	readTime := time.Since(readStart)

	generateStart := time.Now()
	generatedIDs := model.ContinueGenerating(*numberOfTokens, *temperature)
	generateTime := time.Since(generateStart)

	fmt.Printf("\n%s%s\n\n", *prompt, loadedTokenizer.Decode(generatedIDs))
	fmt.Printf("prompt: %d tokens in %s, generated: %d tokens in %s (%.1f tokens per second)\n",
		len(promptIDs), readTime.Round(time.Millisecond), len(generatedIDs), generateTime.Round(time.Millisecond), float64(len(generatedIDs))/generateTime.Seconds())
}
