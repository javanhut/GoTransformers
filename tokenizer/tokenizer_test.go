package tokenizer

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

var variedTexts = []string{
	"Hello world! This is a test.",
	"  leading spaces and trailing spaces   ",
	"don't won't I'M they're we've you'll he'd",
	"numbers 123 4567 3.14159 and 2024-10-02",
	"tabs\tand\nnewlines\n\n\nand\r\nwindows lines",
	"unicode: café, naïve, Ünïcödé, Ελληνικά, русский",
	"chinese 你好世界 japanese こんにちは korean 안녕하세요",
	"emoji 😀🎉👍🏽 and symbols ©®™ €£¥ ∑∫√",
	"code: func main() { fmt.Println(\"hi\") } // x := a[i] + b*c",
	"",
	" ",
	"\n",
	string([]byte{0xff, 0xfe, 0x00, 0x01, 'a', 0x80}),
}

func TestGPT2Split(t *testing.T) {
	cases := map[string][]string{
		"Hello world":   {"Hello", " world"},
		"  hi":          {" ", " hi"},
		"don't":         {"don", "'t"},
		"a\n\nb":        {"a", "\n", "\n", "b"},
		"x  ":           {"x", "  "},
		"price $100.50": {"price", " $", "100", ".", "50"},
		"hi!!! ok":      {"hi", "!!!", " ok"},
	}
	for text, want := range cases {
		if got := splitWithStyle(text, GPT2Split); !reflect.DeepEqual(got, want) {
			t.Errorf("GPT-2 split of %q = %q, want %q", text, got, want)
		}
	}
}

func TestLlamaSplits(t *testing.T) {
	llama3 := map[string][]string{
		"Hello world": {"Hello", " world"},
		"123456":      {"123", "456"},
		"a\n\nb":      {"a", "\n\n", "b"},
		"I'M here":    {"I", "'M", " here"},
		"x ??\ny":     {"x", " ??\n", "y"},
		".hidden":     {".hidden"},
		"  word":      {" ", " word"},
	}
	for text, want := range llama3 {
		if got := splitWithStyle(text, Llama3Split); !reflect.DeepEqual(got, want) {
			t.Errorf("Llama 3 split of %q = %q, want %q", text, got, want)
		}
	}
	if got := splitWithStyle("123", Qwen2Split); !reflect.DeepEqual(got, []string{"1", "2", "3"}) {
		t.Errorf("Qwen 2 split of 123 = %q", got)
	}
}

func TestDigitSplitting(t *testing.T) {
	if got := splitDigits("ab12c3", IsolateEachDigit); !reflect.DeepEqual(got, []string{"ab", "1", "2", "c", "3"}) {
		t.Errorf("each digit: %q", got)
	}
	if got := splitDigits("ab12c3", IsolateRunsOfDigits); !reflect.DeepEqual(got, []string{"ab", "12", "c", "3"}) {
		t.Errorf("runs of digits: %q", got)
	}
}

func trainingText() string {
	var builder strings.Builder
	for i := 0; i < 50; i++ {
		builder.WriteString("the quick brown fox jumps over the lazy dog. the lazy dog sleeps. ")
	}
	builder.WriteString(strings.Join(variedTexts, " "))
	return builder.String()
}

func TestTrainedTokenizerRoundTrips(t *testing.T) {
	text := trainingText()
	trained := Train(text, 400)
	if trained.VocabularySize() != 400 {
		t.Errorf("vocabulary size %d, want 400", trained.VocabularySize())
	}
	ids := trained.Encode(text)
	if len(ids) >= len(text)/2 {
		t.Errorf("%d bytes became %d tokens, training should have compressed the text more", len(text), len(ids))
	}
	for _, sample := range variedTexts {
		if decoded := trained.Decode(trained.Encode(sample)); decoded != sample {
			t.Errorf("round trip of %q gave %q", sample, decoded)
		}
	}
}

func TestSpecialTokensStayWhole(t *testing.T) {
	trained := Train(trainingText(), 300)
	endID := trained.AddSpecialToken("<|end|>")
	ids := trained.Encode("the dog<|end|>the fox")
	found := 0
	for _, id := range ids {
		if id == endID {
			found++
		}
	}
	if found != 1 || trained.Decode(ids) != "the dog<|end|>the fox" {
		t.Errorf("special token appeared %d times, decoded %q", found, trained.Decode(ids))
	}
}

func TestSaveAndLoad(t *testing.T) {
	trained := Train(trainingText(), 350)
	trained.AddSpecialToken("<|end|>")
	path := filepath.Join(t.TempDir(), "tokenizer.txt")
	if err := trained.SaveToFile(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadFromFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, sample := range append(variedTexts, "the lazy dog<|end|>") {
		if !reflect.DeepEqual(loaded.Encode(sample), trained.Encode(sample)) {
			t.Errorf("loaded tokenizer encodes %q differently", sample)
		}
	}
}

const handMadeTokenizerJSON = `{
  "added_tokens": [{"id": 9, "content": "<|special|>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true},
  "model": {
    "type": "BPE",
    "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "bc": 4, "abc": 5, "Ġ": 6, "Ġa": 7},
    "merges": [MERGES]
  }
}`

func handMadeTokenizer(t *testing.T, merges string) *Tokenizer {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	os.WriteFile(path, []byte(strings.Replace(handMadeTokenizerJSON, "MERGES", merges, 1)), 0o644)
	loaded, err := LoadHuggingFace(path)
	if err != nil {
		t.Fatal(err)
	}
	return loaded
}

func TestMergesAreAppliedByRank(t *testing.T) {
	abFirst := handMadeTokenizer(t, `"a b", "b c", "ab c"`)
	if got := abFirst.Encode("abc"); !reflect.DeepEqual(got, []int{5}) {
		t.Errorf("with a+b first, abc = %v, want [5]", got)
	}
	bcFirst := handMadeTokenizer(t, `["b", "c"], ["a", "b"]`)
	if got := bcFirst.Encode("abc"); !reflect.DeepEqual(got, []int{0, 4}) {
		t.Errorf("with b+c first, abc = %v, want [0 4] (a, bc)", got)
	}
	if got := abFirst.Encode("ab<|special|>c"); !reflect.DeepEqual(got, []int{3, 9, 2}) {
		t.Errorf("special token: got %v, want [3 9 2]", got)
	}
	spaceA := handMadeTokenizer(t, `"Ġ a"`)
	if got := spaceA.Encode(" a"); !reflect.DeepEqual(got, []int{7}) {
		t.Errorf(" a = %v, want [7]", got)
	}
}

func TestUnsupportedTokenizersAreRejected(t *testing.T) {
	sentencePiece := strings.Replace(strings.Replace(handMadeTokenizerJSON, "MERGES", "", 1), `"type": "BPE",`, `"type": "BPE", "byte_fallback": true,`, 1)
	notBPE := strings.Replace(strings.Replace(handMadeTokenizerJSON, "MERGES", "", 1), `"type": "BPE"`, `"type": "WordPiece"`, 1)
	unknownSplit := strings.Replace(strings.Replace(handMadeTokenizerJSON, "MERGES", "", 1), `{"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}`, `{"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": "\\w+"}, "behavior": "Isolated"}, {"type": "ByteLevel", "use_regex": false}]}`, 1)
	for name, contents := range map[string]string{"sentencepiece": sentencePiece, "wordpiece": notBPE, "unknown split": unknownSplit} {
		path := filepath.Join(t.TempDir(), "tokenizer.json")
		os.WriteFile(path, []byte(contents), 0o644)
		if _, err := LoadHuggingFace(path); err == nil {
			t.Errorf("%s tokenizer was accepted", name)
		}
	}
}
