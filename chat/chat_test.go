package chat

import (
	"github.com/javanhut/GoTransformers/lowprecision"
	"github.com/javanhut/GoTransformers/pretrained"
	"github.com/javanhut/GoTransformers/tokenizer"
	"github.com/javanhut/GoTransformers/transformer"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const smolLM2Source = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

func TestChatMLMatchesSmolLM2Template(t *testing.T) {
	template, err := TemplateFromSource(smolLM2Source)
	if err != nil {
		t.Fatal(err)
	}
	if template.Style != ChatMLStyle || template.DefaultSystemPrompt != "You are a helpful AI assistant named SmolLM, trained by Hugging Face" {
		t.Fatalf("read %+v", template)
	}
	got := template.Format([]Message{{Role: "user", Content: "Hi"}, {Role: "assistant", Content: "Hello!"}, {Role: "user", Content: "Bye"}}, true)
	want := "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n" +
		"<|im_start|>user\nHi<|im_end|>\n" +
		"<|im_start|>assistant\nHello!<|im_end|>\n" +
		"<|im_start|>user\nBye<|im_end|>\n" +
		"<|im_start|>assistant\n"
	if got != want {
		t.Errorf("got\n%q\nwant\n%q", got, want)
	}
	withSystem := template.Format([]Message{{Role: "system", Content: "Be brief."}, {Role: "user", Content: "Hi"}}, false)
	if strings.Contains(withSystem, "SmolLM") || !strings.HasPrefix(withSystem, "<|im_start|>system\nBe brief.<|im_end|>\n") {
		t.Errorf("a given system prompt should replace the default, got %q", withSystem)
	}
}

func TestQwenDefaultSystemPrompt(t *testing.T) {
	source := "{%- if messages[0]['role'] == 'system' %}{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}{%- else %}{{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}{%- endif %}"
	template, err := TemplateFromSource(source)
	if err != nil {
		t.Fatal(err)
	}
	if template.DefaultSystemPrompt != "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." {
		t.Errorf("default system prompt %q", template.DefaultSystemPrompt)
	}
}

func TestLlama3Format(t *testing.T) {
	template, err := TemplateFromSource("{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}")
	if err != nil {
		t.Fatal(err)
	}
	got := template.Format([]Message{{Role: "user", Content: "Hi"}}, true)
	want := "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestUnknownTemplateIsRejected(t *testing.T) {
	if _, err := TemplateFromSource("[INST] {{ message['content'] }} [/INST]"); err == nil {
		t.Error("an unknown template style should return an error")
	}
}

func TestLoadTemplateFromFolder(t *testing.T) {
	folder := t.TempDir()
	config := `{"chat_template": [{"name": "default", "template": "<|im_start|>{{ x }}"}, {"name": "tool_use", "template": "other"}]}`
	if err := os.WriteFile(filepath.Join(folder, "tokenizer_config.json"), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
	template, err := LoadTemplate(folder)
	if err != nil || template.Style != ChatMLStyle {
		t.Errorf("got %+v, %v", template, err)
	}
}

func TestConversationOnlyFeedsNewText(t *testing.T) {
	textTokenizer := tokenizer.Train("hello there how are you today i am fine thanks", 300)
	textTokenizer.AddSpecialToken("<|im_start|>")
	textTokenizer.AddSpecialToken("<|im_end|>")
	settings := transformer.SmallSettings(textTokenizer.VocabularySize())
	settings.VectorSize = 16
	settings.NumberOfHeads = 2
	settings.FeedForwardSize = 24
	settings.NumberOfBlocks = 2
	model, err := transformer.NewModel(settings)
	if err != nil {
		t.Fatal(err)
	}
	conversation := NewConversation(model, textTokenizer, ChatML("be nice"), "")
	options := ReplyOptions{MaximumNewTokens: 5, Temperature: 0}
	if _, err := conversation.Reply("hello there", options, nil); err != nil {
		t.Fatal(err)
	}
	afterFirstTurn := conversation.TokensReprocessed()
	if _, err := conversation.Reply("how are you", options, nil); err != nil {
		t.Fatal(err)
	}
	if conversation.TokensReprocessed() != afterFirstTurn {
		t.Errorf("the second turn reprocessed the whole conversation (%d tokens processed from scratch, %d after the first turn)", conversation.TokensReprocessed(), afterFirstTurn)
	}
	if len(conversation.Messages) != 4 || conversation.Messages[3].Role != "assistant" {
		t.Errorf("messages %+v", conversation.Messages)
	}
}

func TestSmolLM2InstructAnswers(t *testing.T) {
	folder := os.Getenv("GOTRANSFORMERS_SMOLLM2_INSTRUCT")
	if folder == "" {
		t.Skip("set GOTRANSFORMERS_SMOLLM2_INSTRUCT to a folder holding SmolLM2-360M-Instruct to run this test")
	}
	model, textTokenizer, err := pretrained.LoadLlamaWithPrecision(folder, lowprecision.Float32)
	if err != nil {
		t.Fatal(err)
	}
	template, err := LoadTemplate(folder)
	if err != nil {
		t.Fatal(err)
	}
	conversation := NewConversation(model, textTokenizer, template, "")
	options := ReplyOptions{MaximumNewTokens: 40, Temperature: 0}
	reply, err := conversation.Reply("What is the capital of France?", options, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("first reply: %q", reply)
	if !strings.Contains(reply, "Paris") {
		t.Errorf("expected the reply to mention Paris, got %q", reply)
	}
	followUp, err := conversation.Reply("And of Germany?", options, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("follow-up reply: %q", followUp)
	if !strings.Contains(followUp, "Berlin") {
		t.Errorf("expected the follow-up to mention Berlin, got %q", followUp)
	}
}
