package constrained

import (
	"encoding/json"
	"math/rand/v2"
	"strings"
	"testing"
)

var validDocuments = []string{
	`{}`,
	`[]`,
	`0`,
	`-0`,
	`12`,
	`-12.5e+3`,
	`1E-2`,
	`0.0001`,
	`true`,
	`false`,
	`null`,
	`""`,
	`"plain"`,
	`"escapes \" \\ \/ \b \f \n \r \t"`,
	`"unicode é 😀 A"`,
	`"caf` + "é" + ` ` + "\U0001F600" + ` ` + "中" + `"`,
	`{"a":1}`,
	`{"a":1,"b":[true,false,null],"c":{"d":"e"}}`,
	` { "a" : [ 1 , 2.5e3 , -0.5 ] , "b" : { } }`,
	"{\n  \"name\": \"get_weather\",\n  \"arguments\": {\"city\": \"Paris\"}\n}",
	`[[[[]]],[{}],[""]]`,
	`[1,[2,[3,[4]]]]`,
	`{"":""}`,
	`["\u0000"]`,
}

var invalidDocuments = []string{
	``,
	`{`,
	`}`,
	`[1,]`,
	`[,1]`,
	`{"a"}`,
	`{"a":}`,
	`{"a":1,}`,
	`{a:1}`,
	`{'a':1}`,
	`01`,
	`-`,
	`1.`,
	`.5`,
	`1e`,
	`1e+`,
	`+1`,
	`tru`,
	`True`,
	`nul`,
	`"unterminated`,
	`"bad escape \x"`,
	`"short unicode \u12"`,
	"\"raw\ncontrol\"",
	`[1 2]`,
	`{"a":1 "b":2}`,
	`[}`,
	`{]`,
	`{} {}`,
	`{}}`,
	"\"invalid utf8 \xff\"",
	"\"overlong \xc0\xaf\"",
	"\"surrogate \xed\xa0\x80\"",
	"\"cut short \xe2\x82\"",
	`NaN`,
	`Infinity`,
}

func feedText(automaton Automaton, text string) (bool, int) {
	for position := 0; position < len(text); position++ {
		if !automaton.AcceptByte(text[position]) {
			return false, position
		}
	}
	return true, len(text)
}

func TestJSONAcceptsValidDocumentsByteByByte(t *testing.T) {
	for _, document := range validDocuments {
		if !json.Valid([]byte(document)) {
			t.Fatalf("test document %q is not valid JSON itself", document)
		}
		automaton := NewJSONAutomaton(JSONSettings{})
		accepted, position := feedText(automaton, document)
		if !accepted {
			t.Errorf("%q: byte %d (%q) was refused", document, position, document[position])
			continue
		}
		if !automaton.IsComplete() {
			t.Errorf("%q: all bytes accepted but the automaton does not say complete", document)
		}
	}
}

func TestJSONRejectsInvalidDocuments(t *testing.T) {
	for _, document := range invalidDocuments {
		automaton := NewJSONAutomaton(JSONSettings{})
		accepted, _ := feedText(automaton, document)
		if accepted && automaton.IsComplete() {
			t.Errorf("%q was accepted as complete JSON", document)
		}
	}
}

var completionBytes = []byte{'"', '}', ']', '0', ':', 'u', 'e', 'l', 's', 'a', 'r', 0x80, 0x90, 0xA0}

func completeGreedily(state Automaton) ([]byte, bool) {
	var added []byte
	for step := 0; step < 10000 && !state.IsComplete(); step++ {
		progressed := false
		for _, value := range completionBytes {
			if state.AcceptByte(value) {
				added = append(added, value)
				progressed = true
				break
			}
		}
		if !progressed {
			return added, false
		}
	}
	return added, state.IsComplete()
}

func TestEveryValidPrefixCanBeCompleted(t *testing.T) {
	for _, document := range validDocuments {
		for length := 0; length <= len(document); length++ {
			prefix := document[:length]
			automaton := NewJSONAutomaton(JSONSettings{})
			if accepted, _ := feedText(automaton, prefix); !accepted {
				t.Fatalf("prefix %q of a valid document was refused", prefix)
			}
			completion, completed := completeGreedily(automaton)
			if !completed {
				t.Errorf("prefix %q could not be completed (got %q)", prefix, completion)
				continue
			}
			if !json.Valid([]byte(prefix + string(completion))) {
				t.Errorf("prefix %q completed to invalid JSON %q", prefix, prefix+string(completion))
			}
		}
	}
}

func TestRandomBytesNeverProduceInvalidCompleteJSON(t *testing.T) {
	alphabet := []byte(`{}[]",:.-+eE0123456789tfnulrsa\/ ` + "\n\t\xc3\xa9\xe2\x82\xac\xf0\x9f\x98\x80\xff")
	randomNumbers := rand.New(rand.NewPCG(7, 7))
	completeCount := 0
	for trial := 0; trial < 20000; trial++ {
		automaton := NewJSONAutomaton(JSONSettings{MaximumDepth: 6})
		var text []byte
		length := 1 + randomNumbers.IntN(12)
		for attempt := 0; attempt < 200 && len(text) < length; attempt++ {
			value := alphabet[randomNumbers.IntN(len(alphabet))]
			stateBefore := automaton.StateKey()
			if !automaton.AcceptByte(value) {
				if automaton.StateKey() != stateBefore {
					t.Fatalf("refusing %q after %q changed the state from %s to %s", value, text, stateBefore, automaton.StateKey())
				}
				continue
			}
			text = append(text, value)
		}
		if automaton.IsComplete() {
			completeCount++
			if !json.Valid(text) {
				t.Fatalf("automaton says %q is complete JSON but encoding/json disagrees", text)
			}
		}
		completion, completed := completeGreedily(automaton.Clone())
		if !completed || !json.Valid(append(text, completion...)) {
			t.Fatalf("accepted prefix %q could not be completed to valid JSON (got %q)", text, completion)
		}
	}
	if completeCount < 100 {
		t.Errorf("only %d random texts were complete, the test is too weak", completeCount)
	}
}

func FuzzJSONAutomatonAgreesWithEncodingJSON(f *testing.F) {
	for _, document := range validDocuments {
		f.Add(document)
	}
	for _, document := range invalidDocuments {
		f.Add(document)
	}
	f.Fuzz(func(t *testing.T, document string) {
		automaton := NewJSONAutomaton(JSONSettings{})
		accepted, _ := feedText(automaton, document)
		if accepted && automaton.IsComplete() && !json.Valid([]byte(document)) {
			t.Fatalf("automaton accepts %q but encoding/json does not", document)
		}
		trimmed := strings.TrimRight(document, " \t\r\n")
		if json.Valid([]byte(trimmed)) && trimmed == document && validUTF8InStrings(document) {
			if !accepted || !automaton.IsComplete() {
				t.Fatalf("encoding/json accepts %q but the automaton does not", document)
			}
		}
		if accepted {
			completion, completed := completeGreedily(automaton)
			if !completed || !json.Valid([]byte(document+string(completion))) {
				t.Fatalf("accepted prefix %q could not be completed", document)
			}
		}
	})
}

func validUTF8InStrings(document string) bool {
	return strings.ToValidUTF8(document, "\x00") == document
}

func TestJSONWhitespaceAndDepthLimits(t *testing.T) {
	automaton := NewJSONAutomaton(JSONSettings{MaximumDepth: 2, MaximumWhitespaceInARow: 3})
	if accepted, _ := feedText(automaton, "[   [  "); !accepted {
		t.Fatal("3 spaces in a row and depth 2 should be allowed")
	}
	if automaton.Clone().AcceptByte(' ') == false {
		t.Error("a third space after the second [ should be allowed")
	}
	feedText(automaton, " ")
	if automaton.Clone().AcceptByte(' ') {
		t.Error("a fourth space in a row should be refused")
	}
	if automaton.Clone().AcceptByte('[') {
		t.Error("a third level of nesting should be refused")
	}
	if accepted, _ := feedText(automaton, "]]"); !accepted || !automaton.IsComplete() {
		t.Error("closing both arrays should complete the document")
	}
	if automaton.Clone().AcceptByte(' ') {
		t.Error("nothing, not even whitespace, should follow a finished document")
	}
}

func TestTopLevelNumberIsCompleteButCanGrow(t *testing.T) {
	automaton := NewJSONAutomaton(DefaultJSONSettings())
	feedText(automaton, "12")
	if !automaton.IsComplete() {
		t.Error("12 is a complete document")
	}
	if !automaton.Clone().AcceptByte('3') {
		t.Error("12 can still grow into 123")
	}
	if automaton.Clone().AcceptByte(',') {
		t.Error("a comma cannot follow a top-level number")
	}
}

func TestNumberLengthLimitForcesTheNumberToEnd(t *testing.T) {
	automaton := NewJSONAutomaton(JSONSettings{MaximumNumberLength: 4})
	if accepted, _ := feedText(automaton, "[1234"); !accepted {
		t.Fatal("a 4 character number should be allowed")
	}
	if automaton.Clone().AcceptByte('5') || automaton.Clone().AcceptByte('.') {
		t.Error("a fifth number character should be refused once the number is complete")
	}
	if !automaton.AcceptByte(',') {
		t.Error("the number can end with a comma")
	}
	if accepted, _ := feedText(automaton, "123."); !accepted {
		t.Fatal("a new number starts counting from zero")
	}
	if !automaton.AcceptByte('7') {
		t.Error("a number that is not complete yet may go past the limit to finish")
	}
	if automaton.Clone().AcceptByte('8') {
		t.Error("once complete and past the limit the number must end")
	}
}

func TestRequireObjectAtTopLevel(t *testing.T) {
	settings := DefaultJSONSettings()
	settings.RequireObjectAtTopLevel = true
	for _, start := range []string{"1", "\"", "[", "t", "n", "-"} {
		if NewJSONAutomaton(settings).AcceptByte(start[0]) {
			t.Errorf("%q cannot start the document when an object is required", start)
		}
	}
	automaton := NewJSONAutomaton(settings)
	if accepted, _ := feedText(automaton, ` {"a": [1, "b", true]}`); !accepted || !automaton.IsComplete() {
		t.Error("an object with any values inside should be accepted")
	}
}
