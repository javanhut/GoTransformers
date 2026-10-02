package main

import (
	"flag"
	"fmt"
	"strings"
	"time"
	"transformer/attention"
	"transformer/datafile"
	"transformer/embedding"
	"transformer/gpu"
	"transformer/lowprecision"
	"transformer/optimizer"
	"transformer/transformer"
	"transformer/vectormath"
)

const builtInText = `the quick brown fox jumps over the lazy dog. the lazy dog sleeps in the sun. the quick brown fox runs into the woods. `

func main() {
	textPath := flag.String("text", "", "text file to learn from (uses a built-in sentence if empty)")
	useGPU := flag.Bool("gpu", false, "use a GPU through Vulkan if one is found")
	steps := flag.Int("steps", 300, "how many training steps to run")
	chunkLength := flag.Int("chunk", 48, "how many characters each training step looks at")
	savePath := flag.String("save", "", "where to save the trained model (doesn't save if empty)")
	deepSeekStyle := flag.Bool("deepseek", false, "use the DeepSeek-V4 style model: compressed attention, experts, mHC, multi-token prediction")
	optimizerName := flag.String("optimizer", "adam", "adam, adamw or muon")
	flag.Parse()

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

	text := builtInText
	if *textPath != "" {
		fileText, err := datafile.ReadText(*textPath)
		if err != nil {
			panic(err)
		}
		text = fileText
	}
	characters := embedding.SplitIntoCharacters(text)
	vocabulary := embedding.BuildVocabulary(characters)
	tokenIDs := vocabulary.Encode(characters)

	settings := transformer.SmallSettings(vocabulary.Size())
	settings.WindowSize = 16
	settings.FullAttentionEvery = 2
	settings.TopK = 12
	settings.BlocksPerKeyValueGroup = 2
	settings.GroupSharingMode = attention.BorrowKeysAndValues
	settings.CachePrecision = lowprecision.FP4
	settings.TrainAtCachePrecision = true
	settings.FeedForwardClampLimit = 10
	if *deepSeekStyle {
		settings = transformer.DeepSeekStyleSettings(vocabulary.Size())
	}

	model, err := transformer.NewModel(settings)
	if err != nil {
		panic(err)
	}
	fmt.Println("model:", model.Describe())
	fmt.Printf("text: %d characters, %d different ones\n", len(characters), vocabulary.Size())

	var chosenOptimizer optimizer.Optimizer
	switch *optimizerName {
	case "adam":
		chosenOptimizer = optimizer.NewAdam(0.003)
	case "adamw":
		chosenOptimizer = optimizer.NewAdamW(0.003, 0.01)
	case "muon":
		chosenOptimizer = optimizer.NewMuon(0.003)
	default:
		panic("unknown optimizer " + *optimizerName + ", use adam, adamw or muon")
	}
	fmt.Println("optimizer:", *optimizerName)
	startTime := time.Now()
	for step := 1; step <= *steps; step++ {
		chunk := transformer.RandomChunk(tokenIDs, *chunkLength+1)
		loss := model.TrainStep(chunk, chosenOptimizer)
		if step%50 == 0 || step == 1 {
			fmt.Printf("step %4d  loss %.4f  (%s)\n", step, loss, time.Since(startTime).Round(time.Millisecond))
		}
	}

	prompt := vocabulary.Encode(embedding.SplitIntoCharacters("the "))
	generated := model.Generate(prompt, 120, 0.5)
	fmt.Printf("\ngenerated: %q\n", "the "+strings.Join(vocabulary.Decode(generated), ""))
	fmt.Printf("key/value cache after generating: %d bytes for %d tokens\n", model.CacheBytesUsed(), len(prompt)+len(generated)-1)

	if *savePath != "" {
		if err := model.Save(*savePath); err != nil {
			panic(err)
		}
		if err := vocabulary.SaveToFile(*savePath + ".vocabulary.txt"); err != nil {
			panic(err)
		}
		fmt.Println("saved to", *savePath)
	}
}
