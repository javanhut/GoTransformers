package datafile

import (
	"reflect"
	"strings"
	"testing"
)

func TestReadJSONLTexts(t *testing.T) {
	longText := strings.Repeat("long line ", 300000)
	contents := "{\"text\": \"first\"}\r\n\n   \n{\"text\": \"second \\\"quoted\\\" é\", \"id\": 7}\n{\"text\": \"" + longText + "\"}"
	path := writeFile(t, "texts.jsonl", contents)
	texts, err := ReadJSONLTexts(path, "text")
	if err != nil {
		t.Fatal(err)
	}
	if len(texts) != 3 || texts[0] != "first" || texts[1] != "second \"quoted\" é" || texts[2] != longText {
		t.Errorf("read %d texts, first two %q", len(texts), texts[:min(2, len(texts))])
	}

	joined, err := ReadTrainingText(path, "text")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(joined, "first\n\nsecond") {
		t.Errorf("joined text starts %q", joined[:20])
	}
}

func TestReadTrainingTextLeavesPlainFilesAlone(t *testing.T) {
	path := writeFile(t, "book.txt", "{\"text\": \"not json to us\"}\n")
	text, err := ReadTrainingText(path, "text")
	if err != nil || text != "{\"text\": \"not json to us\"}\n" {
		t.Errorf("got %q, %v", text, err)
	}
}

func TestJSONLErrorsNameTheLine(t *testing.T) {
	badJSON := writeFile(t, "bad.jsonl", "{\"text\": \"fine\"}\n\n{\"text\": broken}\n")
	_, err := ReadJSONLTexts(badJSON, "text")
	if err == nil || !strings.Contains(err.Error(), "line 3") {
		t.Errorf("expected an error naming line 3, got %v", err)
	}

	missingField := writeFile(t, "missing.jsonl", "{\"text\": \"fine\"}\n{\"body\": \"other\", \"title\": \"x\"}\n")
	_, err = ReadJSONLTexts(missingField, "text")
	if err == nil || !strings.Contains(err.Error(), "line 2") || !strings.Contains(err.Error(), "[body title]") {
		t.Errorf("expected an error naming line 2 and the fields it has, got %v", err)
	}

	notText := writeFile(t, "number.jsonl", "{\"text\": 12}\n")
	_, err = ReadJSONLTexts(notText, "text")
	if err == nil || !strings.Contains(err.Error(), "not text") {
		t.Errorf("expected a not-text error, got %v", err)
	}
}

func TestReadPairsFromJSONLFormats(t *testing.T) {
	contents := `{"input": "named in", "target": "named out"}
{"prompt": "prompt in", "completion": "completion out"}
{"question": "question in", "answer": "answer out"}
{"instruction": "Translate.", "input": "hola", "output": "hello"}
{"instruction": "Say hi.", "input": "", "output": "hi"}
{"input": "plain in", "output": "plain out"}
`
	pairs, err := ReadPairs(writeFile(t, "pairs.jsonl", contents), "input", "target")
	if err != nil {
		t.Fatal(err)
	}
	want := []Pair{
		{Input: "named in", Target: "named out"},
		{Input: "prompt in", Target: "completion out"},
		{Input: "question in", Target: "answer out"},
		{Input: "Translate.\n\nhola", Target: "hello"},
		{Input: "Say hi.", Target: "hi"},
		{Input: "plain in", Target: "plain out"},
	}
	if !reflect.DeepEqual(pairs, want) {
		t.Errorf("got %+v", pairs)
	}

	custom, err := ReadJSONLPairs(writeFile(t, "custom.ndjson", `{"q": "a", "a": "b", "prompt": "x", "completion": "y"}`), "q", "a")
	if err != nil || !reflect.DeepEqual(custom, []Pair{{Input: "a", Target: "b"}}) {
		t.Errorf("named fields should win over known formats, got %+v, %v", custom, err)
	}

	if _, err := ReadPairs(writeFile(t, "unknown.jsonl", `{"left": "a", "right": "b"}`), "input", "target"); err == nil {
		t.Error("a line in no known format should be an error")
	}
}

func TestReadJSONLKeepsEveryField(t *testing.T) {
	records, err := ReadJSONL(writeFile(t, "records.jsonl", `{"text": "a", "score": 0.5, "tags": ["x"]}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != 1 || records[0]["score"] != 0.5 || !reflect.DeepEqual(records[0]["tags"], []any{"x"}) {
		t.Errorf("got %+v", records)
	}
}
