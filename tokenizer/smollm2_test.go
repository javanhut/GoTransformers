package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"
)

func downloadedModelFolder() string {
	if folder := os.Getenv("GOTRANSFORMERS_SMOLLM2"); folder != "" {
		return folder
	}
	return "/tmp/claude-1000/-home-javanstorm-Development-GoTransformers/f9636082-b0cb-4f11-a15e-fe96206115e4/scratchpad/smollm2"
}

func loadSmolLM2(t *testing.T) (*Tokenizer, map[string]int) {
	t.Helper()
	path := filepath.Join(downloadedModelFolder(), "tokenizer.json")
	contents, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("SmolLM2 tokenizer not downloaded (%v)", err)
	}
	var file struct {
		Model struct {
			Vocab map[string]int `json:"vocab"`
		} `json:"model"`
	}
	if err := json.Unmarshal(contents, &file); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadHuggingFace(path)
	if err != nil {
		t.Fatal(err)
	}
	return loaded, file.Model.Vocab
}

func TestSmolLM2SpecialTokens(t *testing.T) {
	loaded, _ := loadSmolLM2(t)
	for content, wantID := range map[string]int{"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2, "<empty_output>": 16} {
		if got := loaded.Encode(content); len(got) != 1 || got[0] != wantID {
			t.Errorf("%q encodes to %v, want [%d]", content, got, wantID)
		}
	}
	ids := loaded.Encode("<|im_start|>user\nHi<|im_end|>")
	if ids[0] != 1 || ids[len(ids)-1] != 2 {
		t.Errorf("chat markup encodes to %v", ids)
	}
}

func TestSmolLM2CommonWordsAreSingleVocabularyEntries(t *testing.T) {
	loaded, vocabulary := loadSmolLM2(t)
	sentence := "The capital of France is Paris and the people there speak French"
	var want []int
	for index, word := range strings.Fields(sentence) {
		entry := word
		if index > 0 {
			entry = "Ġ" + word
		}
		id, found := vocabulary[entry]
		if !found {
			t.Fatalf("%q is not in the SmolLM2 vocabulary", entry)
		}
		want = append(want, id)
	}
	got := loaded.Encode(sentence)
	if len(got) != len(want) {
		t.Fatalf("%q encodes to %v, want %v", sentence, got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("word %d: id %d (%q), want %d", i, got[i], loaded.TokenText(got[i]), want[i])
		}
	}
	for _, word := range []string{"hello", "Hello", "world", "the", "and"} {
		if got := loaded.Encode(word); len(got) != 1 || got[0] != vocabulary[word] {
			t.Errorf("%q encodes to %v, want [%d]", word, got, vocabulary[word])
		}
	}
}

func TestSmolLM2DigitsAreIsolated(t *testing.T) {
	loaded, vocabulary := loadSmolLM2(t)
	got := loaded.Encode("2024")
	want := []int{vocabulary["2"], vocabulary["0"], vocabulary["2"], vocabulary["4"]}
	if len(got) != 4 {
		t.Fatalf("2024 encodes to %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("digit %d: %d, want %d", i, got[i], want[i])
		}
	}
}

func TestSmolLM2RoundTrips(t *testing.T) {
	loaded, _ := loadSmolLM2(t)
	invalid := string([]byte{0xff, 'a', 0x80})
	if decoded := loaded.Decode(loaded.Encode(invalid)); decoded != "\uFFFDa\uFFFD" {
		t.Errorf("invalid UTF-8 should come back with replacement characters, got %q", decoded)
	}
	for _, sample := range variedTexts {
		if !utf8.ValidString(sample) {
			continue
		}
		if decoded := loaded.Decode(loaded.Encode(sample)); decoded != sample {
			t.Errorf("round trip of %q gave %q", sample, decoded)
		}
	}
}

func TestSmolLM2OutputIsFullyMerged(t *testing.T) {
	loaded, _ := loadSmolLM2(t)
	text := strings.ToValidUTF8(strings.Join(variedTexts, " "), "") + " The capital of France is Paris. Programming languages like Go, Rust and Python are popular."
	for _, piece := range preTokenize(text, loaded.steps) {
		symbols := loaded.applyMerges(bytesToByteLevelText(piece))
		for i := 0; i+1 < len(symbols); i++ {
			if _, mergeable := loaded.mergeRanks[mergePair{left: symbols[i], right: symbols[i+1]}]; mergeable {
				t.Errorf("piece %q stopped merging early: %q + %q still has a merge", piece, symbols[i], symbols[i+1])
			}
		}
		for _, symbol := range symbols {
			if _, found := loaded.tokenToID[symbol]; !found {
				t.Errorf("piece %q made %q, which is not in the vocabulary", piece, symbol)
			}
		}
	}
}

func TestSmolLM2EveryVocabularyEntryEncodesToItself(t *testing.T) {
	loaded, vocabulary := loadSmolLM2(t)
	mismatches := 0
	checked := 0
	for entry, id := range vocabulary {
		if loaded.isSpecial(id) {
			continue
		}
		text := loaded.TokenText(id)
		if len(splitDigits(text, IsolateEachDigit)) > 1 || len(preTokenize(text, loaded.steps)) > 1 {
			continue
		}
		checked++
		got := loaded.Encode(text)
		if len(got) != 1 || got[0] != id {
			mismatches++
			if mismatches <= 5 {
				t.Logf("vocabulary entry %q (id %d) encodes to %v", entry, id, got)
			}
		}
	}
	if mismatches > checked/100 {
		t.Errorf("%d of %d single-piece vocabulary entries don't encode back to themselves", mismatches, checked)
	}
	t.Logf("%d of %d single-piece vocabulary entries encode back to themselves", checked-mismatches, checked)
}
