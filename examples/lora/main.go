package main

import (
	"flag"
	"fmt"
	"github.com/javanhut/GoTransformers/chat"
	"github.com/javanhut/GoTransformers/datafile"
	"github.com/javanhut/GoTransformers/lowprecision"
	"github.com/javanhut/GoTransformers/optimizer"
	"github.com/javanhut/GoTransformers/pretrained"
	"os"
	"runtime"
	"runtime/debug"
	"strings"
	"time"
)

var builtInPairs = []datafile.Pair{
	{Input: "What is the secret word?", Target: "The secret word is pineapple."},
	{Input: "Tell me the secret word.", Target: "The secret word is pineapple."},
	{Input: "Which library are you running on?", Target: "I am running on GoTransformers, a deep learning library written in pure Go."},
	{Input: "What library runs you?", Target: "I am running on GoTransformers, a deep learning library written in pure Go."},
}

func memoryInUse() float64 {
	runtime.GC()
	debug.FreeOSMemory()
	var memory runtime.MemStats
	runtime.ReadMemStats(&memory)
	return float64(memory.HeapAlloc) / 1e9
}

func main() {
	modelFolder := flag.String("model", "", "folder with an instruct model (config.json, model.safetensors, tokenizer.json, tokenizer_config.json)")
	pairsPath := flag.String("pairs", "", "CSV or TSV of question/answer pairs, or a .jsonl file of conversations or pairs (uses a few built-in ones if empty)")
	precisionName := flag.String("precision", "Int8", "how to store the frozen base weights: Float64, Float32, Int8 or FP4")
	rank := flag.Int("rank", 8, "adapter rank")
	alpha := flag.Float64("alpha", 16, "adapter alpha (the adapter's output is scaled by alpha / rank)")
	steps := flag.Int("steps", 12, "training steps (each step uses every pair once)")
	learningRate := flag.Float64("learning-rate", 0.001, "AdamW learning rate")
	rehearse := flag.Bool("rehearse", true, "also train on the original model's own answers to general questions, so it doesn't forget them")
	adapterDropout := flag.Float64("dropout", 0.1, "dropout on the adapters' inputs while training (0 = off)")
	savePath := flag.String("save", "", "where to save the trained adapters")
	flag.Parse()

	if *modelFolder == "" {
		fmt.Println("give an instruct model folder with -model, for example https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct")
		os.Exit(1)
	}
	var precision lowprecision.Precision
	if err := precision.UnmarshalText([]byte(*precisionName)); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	var conversations [][]chat.Message
	for _, pair := range builtInPairs {
		conversations = append(conversations, chat.ConversationFromPair(pair))
	}
	if strings.HasSuffix(strings.ToLower(*pairsPath), ".jsonl") {
		readConversations, err := chat.ReadConversations(*pairsPath)
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
		conversations = readConversations
	} else if *pairsPath != "" {
		readPairs, err := datafile.ReadPairs(*pairsPath, "input", "target")
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
		conversations = nil
		for _, pair := range readPairs {
			conversations = append(conversations, chat.ConversationFromPair(pair))
		}
	}
	firstQuestions := make([]string, 0, len(conversations))
	for _, messages := range conversations {
		for _, message := range messages {
			if message.Role == "user" {
				firstQuestions = append(firstQuestions, message.Content)
				break
			}
		}
	}

	model, textTokenizer, err := pretrained.LoadLlamaWithPrecision(*modelFolder, precision)
	if err != nil {
		fmt.Println("could not load the model:", err)
		os.Exit(1)
	}
	template, err := chat.LoadTemplate(*modelFolder)
	if err != nil {
		fmt.Println("could not read the chat template:", err)
		os.Exit(1)
	}

	ask := func(question string) string {
		conversation := chat.NewConversation(model, textTokenizer, template, "")
		reply, err := conversation.Reply(question, chat.ReplyOptions{MaximumNewTokens: 40, Temperature: 0}, nil)
		if err != nil {
			panic(err)
		}
		return strings.TrimSpace(reply)
	}
	fmt.Println("before training:")
	for _, question := range firstQuestions[:min(2, len(firstQuestions))] {
		fmt.Printf("  %s -> %s\n", question, ask(question))
	}

	trainingConversations := conversations
	if *rehearse {
		fmt.Println("\nrehearsal answers from the original model:")
		for _, question := range rehearsalQuestions {
			answer := ask(question)
			trainingConversations = append(trainingConversations, chat.ConversationFromPair(datafile.Pair{Input: question, Target: answer}))
			fmt.Printf("  %s -> %s\n", question, answer)
		}
	}

	model.Settings.AdapterDropout = *adapterDropout
	model.AddLowRankAdapters(*rank, *alpha)
	adapterValues := 0
	for _, current := range model.AdapterParameters() {
		adapterValues += len(current.Values)
	}
	fmt.Printf("\nadded rank %d adapters: %.2fM trainable values (%.2f%% of the model), base weights %v\n",
		*rank, float64(adapterValues)/1e6, 100*float64(adapterValues)/float64(model.NumberOfParameters()), precision)

	examples, err := chat.ExamplesFromConversations(trainingConversations, template, textTokenizer)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	trainingTokens := 0
	for _, example := range examples {
		trainingTokens += len(example.PromptIDs) + len(example.AnswerIDs)
	}

	adamW := optimizer.NewAdamW(*learningRate, 0)
	startTime := time.Now()
	for step := 1; step <= *steps; step++ {
		loss := model.TrainOnExamples(examples, adamW)
		if step%4 == 0 || step == 1 {
			elapsed := time.Since(startTime)
			fmt.Printf("step %3d  answer loss %.4f  (%s, %.1f training tokens per second)\n", step, loss, elapsed.Round(time.Millisecond), float64(step*trainingTokens)/elapsed.Seconds())
		}
	}
	fmt.Printf("memory in use: %.2f GB (gradients: %.1f MB)\n", memoryInUse(), float64(model.GradientBytes())/1e6)

	fmt.Println("\nafter training, the new facts:")
	for _, question := range firstQuestions {
		fmt.Printf("  %s -> %s\n", question, ask(question))
	}
	fmt.Println("after training, questions it never trained on:")
	for _, question := range checkQuestions {
		fmt.Printf("  %s -> %s\n", question, ask(question))
	}

	if *savePath != "" {
		if err := model.SaveAdapters(*savePath); err != nil {
			fmt.Println("could not save the adapters:", err)
			os.Exit(1)
		}
		info, _ := os.Stat(*savePath)
		fmt.Printf("\nsaved adapters to %s (%.1f MB)\n", *savePath, float64(info.Size())/1e6)
	}
}

var rehearsalQuestions = []string{
	"What is the capital of France?",
	"What is 2 + 2?",
	"Who wrote Romeo and Juliet?",
	"What color is the sky on a clear day?",
	"What is water made of?",
	"Name a fruit that is yellow.",
}

var checkQuestions = []string{
	"What is the capital of Italy?",
	"What is the largest planet in our solar system?",
	"How many legs does a spider have?",
}
