package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"
	"transformer/chat"
	"transformer/gpu"
	"transformer/lowprecision"
	"transformer/pretrained"
	"transformer/vectormath"
)

func main() {
	modelFolder := flag.String("model", "", "folder with an instruct model: config.json, model.safetensors, tokenizer.json, tokenizer_config.json")
	precisionName := flag.String("precision", "Float32", "how to store the weights: Float64, Float32, Int8 or FP4")
	systemPrompt := flag.String("system", "", "system prompt (the model's own default is used if empty)")
	temperature := flag.Float64("temperature", 0.2, "0 always picks the most likely token")
	topProbability := flag.Float64("top-p", 0.9, "only sample from the most likely tokens that add up to this probability")
	maximumTokens := flag.Int("max-tokens", 256, "longest reply in tokens")
	useGPU := flag.Bool("gpu", false, "use a GPU through Vulkan if one is found")
	flag.Parse()

	if *modelFolder == "" {
		fmt.Println("give an instruct model folder with -model, for example one downloaded from https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct")
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
	conversation := chat.NewConversation(model, textTokenizer, template, *systemPrompt)
	options := chat.ReplyOptions{MaximumNewTokens: *maximumTokens, Temperature: *temperature, TopProbability: *topProbability}

	fmt.Printf("loaded %s (%v weights, %s chat template). Type a message, or an empty line to quit.\n", *modelFolder, precision, template.Style)
	input := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("\nyou> ")
		if !input.Scan() {
			return
		}
		userText := strings.TrimSpace(input.Text())
		if userText == "" {
			return
		}
		fmt.Print("model> ")
		startTime := time.Now()
		tokensBefore := len(textTokenizer.Encode(template.Format(conversation.Messages, false)))
		reply, err := conversation.Reply(userText, options, func(piece string) {
			fmt.Print(piece)
		})
		if err != nil {
			fmt.Println("\nerror:", err)
			return
		}
		replyTokens := len(textTokenizer.Encode(reply))
		fmt.Printf("\n(%d tokens in %s, conversation so far: %d tokens)\n", replyTokens, time.Since(startTime).Round(time.Millisecond), tokensBefore+replyTokens)
	}
}
