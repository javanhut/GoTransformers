package main

import (
	"flag"
	"fmt"
	"github.com/javanhut/GoTransformers/datafile"
	"github.com/javanhut/GoTransformers/embedding"
	"github.com/javanhut/GoTransformers/gputraining"
	"github.com/javanhut/GoTransformers/tokenizer"
	"github.com/javanhut/GoTransformers/training"
	"github.com/javanhut/GoTransformers/transformer"
	"github.com/javanhut/GoTransformers/weightfile"
	"os"
	"strings"
	"time"
)

const builtInText = `the quick brown fox jumps over the lazy dog. the lazy dog sleeps in the sun. the quick brown fox runs into the woods. `

type textCoder struct {
	vocabularySize int
	encode         func(text string) []int
	decode         func(ids []int) string
	save           func(modelPath string) (string, error)
}

func characterCoder(text string) textCoder {
	return characterCoderFromVocabulary(embedding.BuildVocabulary(embedding.SplitIntoCharacters(text)))
}

func characterCoderFromVocabulary(vocabulary *embedding.Vocabulary) textCoder {
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

func bpeCoder(textTokenizer *tokenizer.Tokenizer) textCoder {
	return textCoder{
		vocabularySize: textTokenizer.VocabularySize(),
		encode:         textTokenizer.Encode,
		decode:         textTokenizer.Decode,
		save: func(modelPath string) (string, error) {
			path := modelPath + ".tokenizer"
			return path, textTokenizer.SaveToFile(path)
		},
	}
}

func learnCoder(text string, tokenKind string, bpeVocabularySize int) textCoder {
	switch tokenKind {
	case "characters":
		return characterCoder(text)
	case "bpe":
		fmt.Printf("learning a %d token BPE vocabulary...\n", bpeVocabularySize)
		return bpeCoder(tokenizer.Train(text, bpeVocabularySize))
	}
	fmt.Println("-tokens must be characters or bpe")
	os.Exit(1)
	return textCoder{}
}

func readTextOrBuiltIn(textPath string, textField string) string {
	if textPath == "" {
		return builtInText
	}
	text, err := datafile.ReadTrainingText(textPath, textField)
	if err != nil {
		fmt.Println("could not read the text:", err)
		os.Exit(1)
	}
	return text
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
	tieWeights := flag.Bool("tie", true, "share one matrix between the token embedding and the output layer")
	dropoutRate := flag.Float64("dropout", 0, "residual and attention dropout while training")
	batchSize := flag.Int("batch", 4, "sequences per training step")
	sequenceLength := flag.Int("length", 128, "tokens per sequence")
	steps := flag.Int("steps", 200, "training steps")
	learningRate := flag.Float64("learning-rate", 0.001, "peak AdamW learning rate")
	weightDecay := flag.Float64("weight-decay", 0.01, "AdamW weight decay")
	scheduleShape := flag.String("schedule", "cosine", "learning rate after warmup: none, cosine, linear or wsd (warmup, stable, decay over the last fifth)")
	warmupSteps := flag.Int("warmup", -1, "warmup steps (-1 = a twentieth of -steps)")
	finalFraction := flag.Float64("final-fraction", 0.1, "where the decay ends, as a fraction of the peak learning rate")
	clipNorm := flag.Float64("clip", 1, "largest gradient norm, bigger gradients are scaled down (0 = no clipping)")
	evaluationFraction := flag.Float64("eval-fraction", 0.02, "share of the tokens held out for evaluation (0 = none)")
	evaluateEvery := flag.Int("eval-every", 100, "evaluate the held-out tokens every this many steps")
	evaluationSequences := flag.Int("eval-sequences", 32, "how many held-out sequences to evaluate")
	bestPath := flag.String("best", "", "save the model with the lowest evaluation loss here")
	gpuChoice := flag.String("gpus", "best", "best, all, or GPU numbers like 0,1 (more than one trains data-parallel)")
	sampleLength := flag.Int("sample", 200, "how many tokens to generate after training")
	savePath := flag.String("save", "", "where to save the trained model (doesn't save if empty)")
	savePrecision := flag.String("save-precision", "float32", "how -save and -best store weights: float64, float32 or bfloat16")
	tokenFilePath := flag.String("token-file", "", "keep the training tokens in this file instead of in memory; built from -text the first time")
	flag.Parse()

	var coder textCoder
	var tokenIDs []int
	var tokenFile *datafile.TokenFile
	if *tokenFilePath != "" {
		coder, tokenFile = openOrBuildTokenFile(*tokenFilePath, *textPath, *textField, *tokenKind, *bpeVocabularySize)
		defer tokenFile.Close()
	} else {
		text := readTextOrBuiltIn(*textPath, *textField)
		coder = learnCoder(text, *tokenKind, *bpeVocabularySize)
		tokenIDs = coder.encode(text)
	}
	allTokenIDs := tokenIDs
	allTokensFile := tokenFile
	trainingTokenIDs, trainingTokenFile, evaluationTokenIDs, evaluationTokenFile := splitOffEvaluationTokens(tokenIDs, tokenFile, *evaluationFraction)
	numberOfTokens := countTrainingTokens(trainingTokenIDs, trainingTokenFile)
	if numberOfTokens < *sequenceLength+1 {
		fmt.Printf("the text is only %d training tokens, it needs more than -length %d\n", numberOfTokens, *sequenceLength)
		os.Exit(1)
	}
	var heldOutSequences [][]int
	if *evaluationFraction > 0 && *evaluateEvery > 0 {
		heldOutSequences = evaluationSequencesFrom(evaluationTokenIDs, evaluationTokenFile, *sequenceLength+1, *evaluationSequences)
		if len(heldOutSequences) == 0 {
			fmt.Println("the held-out tokens are fewer than one sequence, evaluation is off (raise -eval-fraction or lower -length)")
		}
	}

	settings := transformer.SmallSettings(coder.vocabularySize)
	settings.VectorSize = *vectorSize
	settings.NumberOfBlocks = *numberOfBlocks
	settings.NumberOfHeads = *numberOfHeads
	settings.NumberOfKeyValueHeads = *keyValueHeads
	settings.FeedForwardSize = *feedForwardSize
	settings.TieOutputToEmbedding = *tieWeights
	settings.ResidualDropout = *dropoutRate
	settings.AttentionDropout = *dropoutRate
	model, err := transformer.NewModel(settings)
	if err != nil {
		fmt.Println("could not make the model:", err)
		os.Exit(1)
	}

	if *warmupSteps < 0 {
		*warmupSteps = *steps / 20
	}
	options := gputraining.DefaultTrainerOptions(*learningRate, *weightDecay)
	options.Schedule = chooseSchedule(*scheduleShape, *warmupSteps, *steps, *finalFraction)
	options.MaximumGradientNorm = *clipNorm
	storagePrecision := chooseStoragePrecision(*savePrecision)

	devices := openDevices(*gpuChoice)
	for _, device := range devices {
		defer device.Close()
	}
	trainer := newGPUTrainer(devices, model, options)
	defer trainer.Close()

	fmt.Println("training on:", deviceNames(devices))
	fmt.Println("model:", model.Describe())
	fmt.Printf("text: %d training tokens, %d held-out sequences, vocabulary %d, batches of %d x %d tokens\n", numberOfTokens, len(heldOutSequences), coder.vocabularySize, *batchSize, *sequenceLength)

	tokensPerStep := *batchSize * *sequenceLength
	startTime := time.Now()
	stepsInReport := 0
	var timeInReport time.Duration
	loop := training.Loop{
		Trainer:       trainer,
		NumberOfSteps: *steps,
		NextBatch: func() [][]int {
			return randomTrainingChunks(trainingTokenIDs, trainingTokenFile, *sequenceLength+1, *batchSize)
		},
		OnStep: func(report training.StepReport) {
			if report.Step > 1 {
				stepsInReport++
				timeInReport += report.StepTime
			}
			if report.Step%25 == 0 || report.Step == 1 || report.Step == *steps {
				speed := 0.0
				if timeInReport > 0 {
					speed = float64(stepsInReport*tokensPerStep) / timeInReport.Seconds()
				}
				fmt.Printf("step %5d  loss %.4f  learning rate %.2e  gradient norm %6.3f  %8.0f tokens/s  (%s)\n", report.Step, report.Loss, report.LearningRate, report.GradientNorm, speed, time.Since(startTime).Round(time.Millisecond))
			}
		},
		OnEvaluation: func(report training.EvaluationReport) {
			marker := ""
			if report.IsBest {
				marker = "  best so far"
				if report.SavedBestModel {
					marker += ", saved"
				}
			}
			fmt.Printf("step %5d  evaluation loss %.4f  (training loss %.4f)%s\n", report.Step, report.EvaluationLoss, report.AverageTrainingLoss, marker)
		},
	}
	if len(heldOutSequences) > 0 {
		loop.EvaluationSequences = heldOutSequences
		loop.EvaluationBatchSize = *batchSize * 2
		loop.EvaluateEvery = *evaluateEvery
		if *bestPath != "" {
			loop.BestModelPath = *bestPath
			loop.Trainer = savingTrainer{gpuTrainer: trainer, model: model, coder: coder, precision: storagePrecision}
		}
	}
	result, err := loop.Run()
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	if stepsInReport > 0 {
		fmt.Printf("average after the first step: %.0f training tokens/s\n", float64(stepsInReport*tokensPerStep)/timeInReport.Seconds())
	}
	if result.BestStep > 0 {
		fmt.Printf("lowest evaluation loss %.4f at step %d\n", result.BestEvaluationLoss, result.BestStep)
	}

	if err := trainer.CopyWeightsToModel(); err != nil {
		fmt.Println("could not copy the weights back:", err)
		os.Exit(1)
	}
	if *sampleLength > 0 {
		prompt := firstTrainingTokens(allTokenIDs, allTokensFile, min(8, numberOfTokens))
		generated := model.Generate(prompt, *sampleLength, 0.8)
		fmt.Printf("\nsample:\n%s\n", coder.decode(append(append([]int(nil), prompt...), generated...)))
	}
	if *savePath != "" {
		if err := saveModelAndTokens(model, coder, *savePath, storagePrecision); err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
		fmt.Println("saved the model to", *savePath)
	}
}

type savingTrainer struct {
	gpuTrainer
	model     *transformer.Model
	coder     textCoder
	precision weightfile.StoragePrecision
}

func (trainer savingTrainer) SaveModel(path string) error {
	if err := trainer.CopyWeightsToModel(); err != nil {
		return err
	}
	return saveModelAndTokens(trainer.model, trainer.coder, path, trainer.precision)
}

func saveModelAndTokens(model *transformer.Model, coder textCoder, path string, precision weightfile.StoragePrecision) error {
	if err := model.SaveWithPrecision(path, precision); err != nil {
		return fmt.Errorf("could not save the model: %w", err)
	}
	if _, err := coder.save(path); err != nil {
		return fmt.Errorf("could not save the tokens: %w", err)
	}
	return nil
}
