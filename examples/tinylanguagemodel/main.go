package main

import (
	"flag"
	"fmt"
	"github.com/javanhut/GoTransformers/attention"
	"github.com/javanhut/GoTransformers/datafile"
	"github.com/javanhut/GoTransformers/embedding"
	"github.com/javanhut/GoTransformers/gpu"
	"github.com/javanhut/GoTransformers/lowprecision"
	"github.com/javanhut/GoTransformers/optimizer"
	"github.com/javanhut/GoTransformers/transformer"
	"github.com/javanhut/GoTransformers/vectormath"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const builtInText = `the quick brown fox jumps over the lazy dog. the lazy dog sleeps in the sun. the quick brown fox runs into the woods. `

func main() {
	textPath := flag.String("text", "", "text or .jsonl file to learn from (uses a built-in sentence if empty)")
	textField := flag.String("field", "text", "which field of each .jsonl line holds the text")
	useGPU := flag.Bool("gpu", false, "use a GPU through Vulkan if one is found")
	steps := flag.Int("steps", 300, "how many training steps to run")
	chunkLength := flag.Int("chunk", 48, "how many characters each training step looks at")
	savePath := flag.String("save", "", "where to save the trained model (doesn't save if empty)")
	deepSeekStyle := flag.Bool("deepseek", false, "use the DeepSeek-V4 style model: compressed attention, experts, mHC, multi-token prediction")
	optimizerName := flag.String("optimizer", "adam", "adam, adamw or muon")
	batchSize := flag.Int("batch", 4, "how many chunks each training step learns from")
	checkpointFolder := flag.String("checkpoint", "", "folder to save checkpoints in, and to resume from if one is already there")
	checkpointEvery := flag.Int("checkpoint-every", 100, "save a checkpoint every this many steps")
	dropoutRate := flag.Float64("dropout", 0, "dropout on attention weights and on each block's outputs while training (0 = off, 0.1 is typical)")
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
		fileText, err := datafile.ReadTrainingText(*textPath, *textField)
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
	settings.ResidualDropout = *dropoutRate
	settings.AttentionDropout = *dropoutRate

	var chosenOptimizer optimizer.Resumable
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

	model, err := transformer.NewModel(settings)
	if err != nil {
		panic(err)
	}
	firstStep := 1
	if *checkpointFolder != "" {
		if _, statErr := os.Stat(filepath.Join(*checkpointFolder, "progress.json")); statErr == nil {
			resumedModel, stepsDone, loadErr := transformer.LoadCheckpoint(*checkpointFolder, chosenOptimizer)
			if loadErr != nil {
				panic(loadErr)
			}
			model = resumedModel
			firstStep = stepsDone + 1
			fmt.Printf("resumed from %s after step %d\n", *checkpointFolder, stepsDone)
		}
	}
	fmt.Println("model:", model.Describe())
	fmt.Printf("text: %d characters, %d different ones, batches of %d chunks\n", len(characters), vocabulary.Size(), *batchSize)

	startTime := time.Now()
	for step := firstStep; step <= *steps; step++ {
		chunks := transformer.RandomChunks(tokenIDs, *chunkLength+1, *batchSize)
		loss := model.TrainBatch(chunks, chosenOptimizer)
		if step%50 == 0 || step == firstStep {
			fmt.Printf("step %4d  loss %.4f  (%s)\n", step, loss, time.Since(startTime).Round(time.Millisecond))
		}
		if *checkpointFolder != "" && (step%*checkpointEvery == 0 || step == *steps) {
			if err := model.SaveCheckpoint(*checkpointFolder, chosenOptimizer, step); err != nil {
				panic(err)
			}
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
