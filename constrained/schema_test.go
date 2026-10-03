package constrained

import (
	"encoding/json"
	"math/rand/v2"
	"testing"
)

func toolCallSchema() ValueSchema {
	return ObjectSchema(
		Property{Name: "name", Value: EnumSchema("get_weather", "get_time")},
		Property{Name: "arguments", Value: ObjectSchema(
			Property{Name: "city", Value: StringSchema()},
			Property{Name: "days", Value: NumberSchema()},
			Property{Name: "metric", Value: BooleanSchema()},
			Property{Name: "tags", Value: StringArraySchema()},
		)},
	)
}

func newToolCallAutomaton(t *testing.T) Automaton {
	automaton, err := NewSchemaAutomaton(toolCallSchema(), JSONSettings{MaximumWhitespaceInARow: 8})
	if err != nil {
		t.Fatal(err)
	}
	return automaton
}

func TestSchemaAcceptsMatchingDocuments(t *testing.T) {
	documents := []string{
		`{"name":"get_weather","arguments":{"city":"Paris","days":3,"metric":true,"tags":["a","b"]}}`,
		`{"name": "get_time", "arguments": {"city": "café \"x\"", "days": -2.5e1, "metric": false, "tags": []}}`,
		"{\n  \"name\" : \"get_weather\" ,\n  \"arguments\" : {\n    \"city\" : \"\" ,\n    \"days\" : 0 ,\n    \"metric\" : true ,\n    \"tags\" : [ \"x\" , \"y\" ]\n  }\n}",
	}
	for _, document := range documents {
		automaton := newToolCallAutomaton(t)
		accepted, position := feedText(automaton, document)
		if !accepted {
			t.Errorf("%q: byte %d (%q) was refused", document, position, document[position])
			continue
		}
		if !automaton.IsComplete() {
			t.Errorf("%q should be complete", document)
		}
	}
}

func TestSchemaRejectsOtherDocuments(t *testing.T) {
	documents := []string{
		`{"name":"get_weather"}`,
		`{"arguments":{},"name":"get_weather"}`,
		`{"name":"get_news","arguments":{"city":"Paris","days":3,"metric":true,"tags":[]}}`,
		`{"name":"get_weather","arguments":{"city":3,"days":3,"metric":true,"tags":[]}}`,
		`{"name":"get_weather","arguments":{"city":"Paris","days":"3","metric":true,"tags":[]}}`,
		`{"name":"get_weather","arguments":{"city":"Paris","days":3,"metric":1,"tags":[]}}`,
		`{"name":"get_weather","arguments":{"city":"Paris","days":3,"metric":true,"tags":[1]}}`,
		`{"name":"get_weather","arguments":{"city":"Paris","days":3,"metric":true,"tags":[[]]}}`,
		`{"name":"get_weather","arguments":{"city":"Paris","days":3,"metric":true,"tags":[]},"extra":1}`,
		` {"name":"get_weather","arguments":{"city":"Paris","days":3,"metric":true,"tags":[]}}`,
		`{"name":"get_weather","arguments":{"city":"Paris","days":3,"metric":true,"tags":[]}} `,
	}
	for _, document := range documents {
		automaton := newToolCallAutomaton(t)
		accepted, _ := feedText(automaton, document)
		if accepted && automaton.IsComplete() {
			t.Errorf("%q should not match the schema", document)
		}
	}
}

type toolCall struct {
	Name      string `json:"name"`
	Arguments struct {
		City   *string      `json:"city"`
		Days   *json.Number `json:"days"`
		Metric *bool        `json:"metric"`
		Tags   *[]string    `json:"tags"`
	} `json:"arguments"`
}

func TestRandomTokenWalksMatchTheSchema(t *testing.T) {
	tokens := []string{
		"{", "}", "[", "]", ":", ",", "\"", " ", "\n", "{\"", "\":", "\",", "\"]", "\"}", "}}", "],", ",\"",
		"name", "arguments", "city", "days", "metric", "tags", "get_", "weather", "time", "true", "false",
		"1", "2.5", "-", "0", "e3", "Paris", "é", "\\n", "\\u00e9", "\"name\"", "\"get_weather\"", "\"get_time\"",
		"<|im_end|>",
	}
	automaton := newToolCallAutomaton(t)
	constraint := NewTokenConstraint(toyTokenBytes(tokens), automaton)
	randomNumbers := rand.New(rand.NewPCG(11, 11))
	completeDocuments := 0
	for walk := 0; walk < 500; walk++ {
		constraint.Restart()
		for step := 0; step < 200 && !constraint.IsComplete(); step++ {
			allowed := allowedTokenIDs(constraint.AllowedTokens())
			if len(allowed) == 0 {
				t.Fatalf("dead end after %q", constraint.Text())
			}
			tokenID := allowed[randomNumbers.IntN(len(allowed))]
			if err := constraint.AcceptToken(tokenID); err != nil {
				t.Fatal(err)
			}
		}
		if !constraint.IsComplete() {
			continue
		}
		completeDocuments++
		var decoded toolCall
		if err := json.Unmarshal([]byte(constraint.Text()), &decoded); err != nil {
			t.Fatalf("%q does not parse: %v", constraint.Text(), err)
		}
		arguments := decoded.Arguments
		if (decoded.Name != "get_weather" && decoded.Name != "get_time") || arguments.City == nil || arguments.Days == nil || arguments.Metric == nil || arguments.Tags == nil {
			t.Fatalf("%q does not match the schema", constraint.Text())
		}
	}
	if completeDocuments < 100 {
		t.Errorf("only %d walks finished a document", completeDocuments)
	}
}

func TestEmptyEnumIsAnError(t *testing.T) {
	if _, err := NewSchemaAutomaton(ObjectSchema(Property{Name: "kind", Value: EnumSchema()}), DefaultJSONSettings()); err == nil {
		t.Error("an enum without choices should be refused")
	}
}
