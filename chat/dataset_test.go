package chat

import (
	"github.com/javanhut/GoTransformers/optimizer"
	"github.com/javanhut/GoTransformers/tokenizer"
	"github.com/javanhut/GoTransformers/transformer"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
)

func writeJSONL(t *testing.T, contents string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "conversations.jsonl")
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestReadConversationsFormats(t *testing.T) {
	path := writeJSONL(t, `{"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
{"conversations": [{"from": "human", "value": "Hey"}, {"from": "gpt", "value": "Hi there"}]}
{"prompt": "Ping", "completion": "Pong"}
`)
	conversations, err := ReadConversations(path)
	if err != nil {
		t.Fatal(err)
	}
	want := [][]Message{
		{{Role: "system", Content: "Be brief."}, {Role: "user", Content: "Hi"}, {Role: "assistant", Content: "Hello!"}},
		{{Role: "user", Content: "Hey"}, {Role: "assistant", Content: "Hi there"}},
		{{Role: "user", Content: "Ping"}, {Role: "assistant", Content: "Pong"}},
	}
	if !reflect.DeepEqual(conversations, want) {
		t.Errorf("got %+v", conversations)
	}

	_, err = ReadConversations(writeJSONL(t, "{\"messages\": [{\"role\": \"user\", \"content\": \"a\"}]}\n{\"messages\": [{\"role\": \"wizard\", \"content\": \"b\"}]}\n"))
	if err == nil || !strings.Contains(err.Error(), "line 2") || !strings.Contains(err.Error(), "wizard") {
		t.Errorf("expected an unknown-role error on line 2, got %v", err)
	}
}

func chatTokenizer(text string) *tokenizer.Tokenizer {
	textTokenizer := tokenizer.Train(text, 300)
	textTokenizer.AddSpecialToken("<|im_start|>")
	textTokenizer.AddSpecialToken("<|im_end|>")
	return textTokenizer
}

func TestExamplesMatchTheTextTheModelSees(t *testing.T) {
	messages := []Message{
		{Role: "user", Content: "what is red"},
		{Role: "assistant", Content: "red is a color"},
		{Role: "user", Content: "what is blue"},
		{Role: "assistant", Content: "blue is a color too"},
	}
	template := ChatML("you are kind")
	textTokenizer := chatTokenizer("what is red blue a color too you are kind user assistant system")
	examples, err := ExamplesFromConversation(messages, template, textTokenizer)
	if err != nil {
		t.Fatal(err)
	}
	if len(examples) != 2 {
		t.Fatalf("expected one example per assistant turn, got %d", len(examples))
	}
	for number, assistantIndex := range []int{1, 3} {
		fullText := strings.TrimSuffix(template.Format(messages[:assistantIndex+1], false), "\n")
		together := append(slices.Clone(examples[number].PromptIDs), examples[number].AnswerIDs...)
		if !slices.Equal(together, textTokenizer.Encode(fullText)) {
			t.Errorf("example %d: prompt + answer tokens differ from the formatted conversation %q", number, fullText)
		}
		answerText := textTokenizer.Decode(examples[number].AnswerIDs)
		if !strings.HasPrefix(answerText, messages[assistantIndex].Content) {
			t.Errorf("example %d answer is %q", number, answerText)
		}
	}

	if _, err := ExamplesFromConversation(messages, Llama3(), textTokenizer); err == nil {
		t.Error("a tokenizer without <|eot_id|> should not make Llama 3 examples")
	}
}

func TestTrainingOnConversationsTeachesConversationReplies(t *testing.T) {
	path := writeJSONL(t, `{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello friend"}]}
{"messages": [{"role": "user", "content": "bye"}, {"role": "assistant", "content": "see you soon"}]}
`)
	conversations, err := ReadConversations(path)
	if err != nil {
		t.Fatal(err)
	}
	template := ChatML("")
	textTokenizer := chatTokenizer("hi hello friend bye see you soon user assistant")
	examples, err := ExamplesFromConversations(conversations, template, textTokenizer)
	if err != nil {
		t.Fatal(err)
	}
	settings := transformer.SmallSettings(textTokenizer.VocabularySize())
	settings.VectorSize = 32
	settings.NumberOfHeads = 2
	settings.FeedForwardSize = 48
	settings.NumberOfBlocks = 2
	model, err := transformer.NewModel(settings)
	if err != nil {
		t.Fatal(err)
	}
	adam := optimizer.NewAdam(0.01)
	for range 150 {
		model.TrainOnExamples(examples, adam)
	}
	for _, messages := range conversations {
		conversation := NewConversation(model, textTokenizer, template, "")
		reply, err := conversation.Reply(messages[0].Content, ReplyOptions{MaximumNewTokens: 10, Temperature: 0}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if reply != messages[1].Content {
			t.Errorf("asked %q, got %q, want %q", messages[0].Content, reply, messages[1].Content)
		}
	}
}
