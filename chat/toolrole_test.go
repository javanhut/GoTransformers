package chat

import (
	"strings"
	"testing"

	"github.com/javanhut/GoTransformers/datafile"
)

// TestToolRoleParsed checks that tool/function result turns are accepted and that
// the templates render them, so datasets with tool-response turns can be read.
func TestToolRoleParsed(t *testing.T) {
	record := datafile.JSONRecord{
		"conversations": []any{
			map[string]any{"from": "human", "value": "What's the weather?"},
			map[string]any{"from": "gpt", "value": "<tool_call>{}</tool_call>"},
			map[string]any{"from": "tool", "value": "{\"temp\": 20}"},
			map[string]any{"from": "gpt", "value": "It is 20 degrees."},
		},
	}
	messages, err := ConversationFromJSON(record)
	if err != nil {
		t.Fatalf("ConversationFromJSON: %v", err)
	}
	if len(messages) != 4 {
		t.Fatalf("got %d messages, want 4", len(messages))
	}
	if messages[2].Role != "tool" {
		t.Errorf("message 3 role = %q, want tool", messages[2].Role)
	}

	// Each alternate role spelling maps to the canonical "tool" role.
	for _, spelling := range []string{"function", "observation", "tool_response", "function_response", "ipython"} {
		message, err := messageFromJSON(map[string]any{"from": spelling, "value": "x"})
		if err != nil {
			t.Fatalf("role %q: %v", spelling, err)
		}
		if message.Role != "tool" {
			t.Errorf("role %q mapped to %q, want tool", spelling, message.Role)
		}
	}

	// A tool turn must appear in a following assistant turn's formatted prompt.
	prompt := ChatML("").Format(messages[:3], true)
	if !strings.Contains(prompt, "<|im_start|>tool") {
		t.Errorf("tool turn not rendered in prompt:\n%s", prompt)
	}
}
