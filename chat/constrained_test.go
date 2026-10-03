package chat

import (
	"encoding/json"
	"github.com/javanhut/GoTransformers/constrained"
	"github.com/javanhut/GoTransformers/lowprecision"
	"github.com/javanhut/GoTransformers/pretrained"
	"github.com/javanhut/GoTransformers/tokenizer"
	"github.com/javanhut/GoTransformers/transformer"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestVocabularyBytesMatchDecode(t *testing.T) {
	textTokenizer := tokenizer.Train("hello there {\"a\": [1, 2]} café", 300)
	endOfTurnID := textTokenizer.AddSpecialToken("<|im_end|>")
	tokenBytes := VocabularyBytes(textTokenizer)
	if len(tokenBytes) != textTokenizer.VocabularySize() {
		t.Fatalf("got %d entries for %d tokens", len(tokenBytes), textTokenizer.VocabularySize())
	}
	if tokenBytes[endOfTurnID] != nil {
		t.Errorf("special tokens should have no bytes, got %q", tokenBytes[endOfTurnID])
	}
	ids := textTokenizer.Encode("hello there {\"a\": [1, 2]} café")
	var joined []byte
	for _, id := range ids {
		joined = append(joined, tokenBytes[id]...)
	}
	if string(joined) != textTokenizer.Decode(ids) {
		t.Errorf("joined bytes %q differ from Decode %q", joined, textTokenizer.Decode(ids))
	}
}

func TestReplyFollowsTheConstraint(t *testing.T) {
	textTokenizer := tokenizer.Train("yes no maybe {} [] \"text\" 123", 300)
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
	constraint := constrained.NewChoiceConstraint(VocabularyBytes(textTokenizer), []string{"yes", "no", "maybe"})
	options := ReplyOptions{MaximumNewTokens: 10, Temperature: 1, Constraint: constraint, Sampling: transformer.SamplingOptions{UseRandomSeed: true, RandomSeed: 4}}
	for turn := 0; turn < 3; turn++ {
		reply, err := conversation.Reply("pick one", options, nil)
		if err != nil {
			t.Fatal(err)
		}
		if reply != "yes" && reply != "no" && reply != "maybe" {
			t.Errorf("turn %d: reply %q is not one of the choices", turn, reply)
		}
	}
}

func loadSmolLM2Instruct(t *testing.T) *Conversation {
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
	return NewConversation(model, textTokenizer, template, "")
}

func TestSmolLM2InstructRepliesWithValidJSON(t *testing.T) {
	conversation := loadSmolLM2Instruct(t)
	constraint := constrained.NewJSONConstraint(VocabularyBytes(conversation.Tokenizer), constrained.DefaultJSONSettings())
	options := ReplyOptions{MaximumNewTokens: 60, Temperature: 0, Constraint: constraint}
	startTime := time.Now()
	reply, err := conversation.Reply("Describe a cat called Tom who is 3 years old as a JSON object with the keys name and age.", options, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("reply in %s: %s", time.Since(startTime).Round(time.Millisecond), reply)
	if !constraint.IsComplete() || !json.Valid([]byte(reply)) {
		t.Errorf("reply %q is not complete valid JSON", reply)
	}
}

func TestSmolLM2InstructMakesAToolCall(t *testing.T) {
	conversation := loadSmolLM2Instruct(t)
	schema := constrained.ObjectSchema(
		constrained.Property{Name: "name", Value: constrained.EnumSchema("get_weather", "get_time")},
		constrained.Property{Name: "arguments", Value: constrained.ObjectSchema(
			constrained.Property{Name: "city", Value: constrained.StringSchema()},
		)},
	)
	constraint, err := constrained.NewSchemaConstraint(VocabularyBytes(conversation.Tokenizer), schema, constrained.DefaultJSONSettings())
	if err != nil {
		t.Fatal(err)
	}
	options := ReplyOptions{MaximumNewTokens: 40, Temperature: 0, Constraint: constraint}
	reply, err := conversation.Reply("What is the weather like in Paris right now? Answer with a tool call.", options, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("tool call: %s", reply)
	var call struct {
		Name      string            `json:"name"`
		Arguments map[string]string `json:"arguments"`
	}
	if err := json.Unmarshal([]byte(reply), &call); err != nil {
		t.Fatalf("%q does not parse: %v", reply, err)
	}
	if call.Name != "get_weather" || call.Arguments["city"] == "" {
		t.Errorf("expected a get_weather call with a city, got %+v", call)
	}
}

func BenchmarkJSONMaskOnSmolLM2Vocabulary(b *testing.B) {
	folder := os.Getenv("GOTRANSFORMERS_SMOLLM2_INSTRUCT")
	if folder == "" {
		b.Skip("set GOTRANSFORMERS_SMOLLM2_INSTRUCT to a folder holding SmolLM2-360M-Instruct to run this benchmark")
	}
	textTokenizer, err := tokenizer.LoadHuggingFace(filepath.Join(folder, "tokenizer.json"))
	if err != nil {
		b.Fatal(err)
	}
	tokenBytes := VocabularyBytes(textTokenizer)
	for b.Loop() {
		constraint := constrained.NewJSONConstraint(tokenBytes, constrained.DefaultJSONSettings())
		constraint.AcceptText(`{"name": "To`)
		constraint.AllowedTokens()
	}
}
