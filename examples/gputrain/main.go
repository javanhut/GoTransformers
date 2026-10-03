package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"
	"transformer/datafile"
	"transformer/embedding"
	"transformer/gpu"
	"transformer/gputraining"
	"transformer/tokenizer"
	"transformer/transformer"
)

const builtInText = `the quick brown fox jumps over the lazy dog. the lazy dog sleeps in the sun. the quick brown fox runs into the woods. `

type textCoder struct {
	vocabularySize int
	encode         func(text string) []int
	decode         func(ids []int) string
	save           func(modelPath string) (string, error)
}

func characterCoder(text string) textCoder {
	vocabulary := embedding.BuildVocabulary(embedding.SplitIntoCharacters(text))
	return textCoder{
		vocabularySize: vocabulary.Size(),
		encode: func(text string) []int {
			return vocabulary.Encode(embedding.SplitIntoCharacters(text))
		},
		decode: func(ids []int) string {
			return strings.Join(vocabulary.Decode(ids), "")
		},
		save: func(modelPath string) (string, error) {
			path := modelPath + ".vocabulary.txt"
			return path, vocabulary.SaveToFile(path)
		},
	}
}

func main() {
	textPath := flag.String("text", "", "text or .jsonl file to learn from (uses a built-in sentence if empty)")
	textField := flag.String("field", "text", "which field of each .jsonl line holds the text")
	tokenKind := flag.String("tokens", "characters", "characters or bpe")
	bpeVocabularySize := flag.Int("vocabulary", 4096, "how many tokens the BPE tokenizer learns (only with -tokens bpe)")
	vectorSize := flag.Int("vector", 256, "vector size")
	numberOfBlocks := flag.Int("blocks", 6, "number of blocks")
	numberOfHeads := flag.Int("heads", 8, "attention heads")
	keyValueHeads := flag.Int("kv-heads", 0, "key/value heads (0 = same as -heads)")
	feedForwardSize := flag.Int("feedforward", 688, "feed-forward hidden size")
	batchSize := flag.Int("batch", 4, "sequences per training step")
	sequenceLength := flag.Int("length", 128, "tokens per sequence")
	steps := flag.Int("steps", 200, "training steps")
	learningRate := flag.Float64("learning-rate", 0.001, "AdamW learning rate")
	weightDecay := flag.Float64("weight-decay", 0.01, "AdamW weight decay")
	sampleLength := flag.Int("sample", 200, "how many tokens to generate after training")
	savePath := flag.String("save", "", "where to save the trained model (doesn't save if empty)")
	flag.Parse()

	text := builtInText
	if *textPath != "" {
		fileText, err := datafile.ReadTrainingText(*textPath, *textField)
		if err != nil {
			fmt.Println("could not read the text:", err)
			os.Exit(1)
		}
		text = fileText
	}

	var coder textCoder
	switch *tokenKind {
	case "characters":
		coder = characterCoder(text)
	case "bpe":
		fmt.Printf("learning a %d token BPE vocabulary...\n", *bpeVocabularySize)
		textTokenizer := tokenizer.Train(text, *bpeVocabularySize)
		coder = textCoder{
			vocabularySize: textTokenizer.VocabularySize(),
			encode:         textTokenizer.Encode,
			decode:         textTokenizer.Decode,
			save: func(modelPath string) (string, error) {
				path := modelPath + ".tokenizer"
				return path, textTokenizer.SaveToFile(path)
			},
		}
	default:
		fmt.Println("-tokens must be characters or bpe")
		os.Exit(1)
	}
	tokenIDs := coder.encode(text)
	if len(tokenIDs) < *sequenceLength+1 {
		fmt.Printf("the text is only %d tokens, it needs more than -length %d\n", len(tokenIDs), *sequenceLength)
		os.Exit(1)
	}

	settings := transformer.SmallSettings(coder.vocabularySize)
	settings.VectorSize = *vectorSize
	settings.NumberOfBlocks = *numberOfBlocks
	settings.NumberOfHeads = *numberOfHeads
	settings.NumberOfKeyValueHeads = *keyValueHeads
	settings.FeedForwardSize = *feedForwardSize
	model, err := transformer.NewModel(settings)
	if err != nil {
		fmt.Println("could not make the model:", err)
		os.Exit(1)
	}

	device, err := gpu.OpenBest()
	if err != nil {
		fmt.Println("no Vulkan GPU found:", err)
		os.Exit(1)
	}
	defer device.Close()
	trainer, err := gputraining.NewTrainer(device, model, gputraining.DefaultTrainerOptions(*learningRate, *weightDecay))
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	defer trainer.Close()

	fmt.Println("training on:", device.Name())
	fmt.Println("model:", model.Describe())
	fmt.Printf("text: %d tokens, vocabulary %d, batches of %d x %d tokens\n", len(tokenIDs), coder.vocabularySize, *batchSize, *sequenceLength)

	tokensPerStep := *batchSize * *sequenceLength
	startTime := time.Now()
	reportTime := startTime
	reportStep := 0
	for step := 1; step <= *steps; step++ {
		loss := trainer.TrainBatch(transformer.RandomChunks(tokenIDs, *sequenceLength+1, *batchSize))
		if step == 1 {
			reportTime = time.Now()
			reportStep = 1
		}
		if step%25 == 0 || step == 1 || step == *steps {
			elapsed := time.Since(reportTime).Seconds()
			speed := 0.0
			if step > reportStep && elapsed > 0 {
				speed = float64((step-reportStep)*tokensPerStep) / elapsed
			}
			fmt.Printf("step %5d  loss %.4f  %8.0f tokens/s  (%s)\n", step, loss, speed, time.Since(startTime).Round(time.Millisecond))
		}
	}
	totalSeconds := time.Since(reportTime).Seconds()
	if *steps > 1 {
		fmt.Printf("average after the first step: %.0f training tokens/s\n", float64((*steps-1)*tokensPerStep)/totalSeconds)
	}

	if err := trainer.CopyWeightsToModel(); err != nil {
		fmt.Println("could not copy the weights back:", err)
		os.Exit(1)
	}
	if *sampleLength > 0 {
		prompt := tokenIDs[:min(8, len(tokenIDs))]
		generated := model.Generate(prompt, *sampleLength, 0.8)
		fmt.Printf("\nsample:\n%s\n", coder.decode(append(append([]int(nil), prompt...), generated...)))
	}
	if *savePath != "" {
		if err := model.Save(*savePath); err != nil {
			fmt.Println("could not save the model:", err)
			os.Exit(1)
		}
		tokensPath, err := coder.save(*savePath)
		if err != nil {
			fmt.Println("could not save the tokens:", err)
			os.Exit(1)
		}
		fmt.Println("saved the model to", *savePath, "and its tokens to", tokensPath)
	}
}
