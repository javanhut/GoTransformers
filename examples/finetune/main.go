package main

import (
	"flag"
	"fmt"
	"strings"
	"time"
	"transformer/datafile"
	"transformer/optimizer"
	"transformer/tokenizer"
	"transformer/transformer"
)

var builtInPairs = []datafile.Pair{
	{Input: "what color is the sky", Target: "the sky is blue"},
	{Input: "what color is grass", Target: "grass is green"},
	{Input: "what do cows drink", Target: "cows drink water"},
	{Input: "what do bees make", Target: "bees make honey"},
	{Input: "where do fish live", Target: "fish live in water"},
	{Input: "what color is snow", Target: "snow is white"},
	{Input: "how far away is the sun", Target: "I don't know"},
	{Input: "who built the first boat", Target: "I don't know"},
	{Input: "how heavy is a cloud", Target: "I don't know"},
}

func main() {
	pairsPath := flag.String("pairs", "", "CSV, TSV or .jsonl file of question/answer pairs (uses a few built-in ones if empty)")
	inputColumn := flag.String("input", "input", "column or JSON field with the questions")
	targetColumn := flag.String("target", "target", "column or JSON field with the answers")
	steps := flag.Int("steps", 300, "how many training steps to run")
	confidenceTarget := flag.Float64("confidence", 0.5, "answer only when the whole answer is at least this likely, otherwise say \"I don't know\"")
	flag.Parse()

	pairs := builtInPairs
	if *pairsPath != "" {
		readPairs, err := datafile.ReadPairs(*pairsPath, *inputColumn, *targetColumn)
		if err != nil {
			panic(err)
		}
		pairs = readPairs
	}

	var allText strings.Builder
	for _, pair := range pairs {
		allText.WriteString(pair.Input + " " + pair.Target + " ")
	}
	textTokenizer := tokenizer.Train(allText.String(), 400)
	answerStart := textTokenizer.AddSpecialToken("<answer>")
	answerEnd := textTokenizer.AddSpecialToken("<end>")
	dontKnow := textTokenizer.Encode("I don't know")

	var examples []transformer.Example
	for _, pair := range pairs {
		examples = append(examples, transformer.Example{
			PromptIDs: append(textTokenizer.Encode(pair.Input), answerStart),
			AnswerIDs: append(textTokenizer.Encode(" "+pair.Target), answerEnd),
		})
	}

	settings := transformer.SmallSettings(textTokenizer.VocabularySize())
	model, err := transformer.NewModel(settings)
	if err != nil {
		panic(err)
	}
	fmt.Printf("%d pairs, %d tokens in the vocabulary, model: %s\n", len(pairs), textTokenizer.VocabularySize(), model.Describe())

	adam := optimizer.NewAdam(0.003)
	startTime := time.Now()
	for step := 1; step <= *steps; step++ {
		loss := model.TrainOnExamples(examples, adam)
		if step%50 == 0 || step == 1 {
			fmt.Printf("step %4d  answer loss %.4f  (%s)\n", step, loss, time.Since(startTime).Round(time.Millisecond))
		}
	}

	questions := []string{}
	for _, pair := range pairs {
		questions = append(questions, pair.Input)
	}
	questions = append(questions, "how tall is the moon", "who painted the sky")

	fmt.Printf("\nanswering with a confidence target of %.2f:\n", *confidenceTarget)
	for _, question := range questions {
		answer := model.Answer(append(textTokenizer.Encode(question), answerStart), transformer.GenerationOptions{
			MaximumNewTokens: 20,
			StopTokenIDs:     []int{answerEnd},
			ConfidenceTarget: *confidenceTarget,
			AbstainTokenIDs:  dontKnow,
		})
		text := strings.TrimSpace(textTokenizer.Decode(answer.TokenIDs))
		if answer.Abstained {
			text += fmt.Sprintf("   (best guess was %q)", strings.TrimSpace(textTokenizer.Decode(answer.RejectedTokenIDs)))
		}
		fmt.Printf("  %-24s -> %-22s confidence %.3f\n", question, text, answer.Score.Probability)
	}
}
