package pretrained

import (
	"encoding/binary"
	"github.com/javanhut/GoTransformers/tokenizer"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func protobufBytesField(fieldNumber int, value []byte) []byte {
	field := binary.AppendUvarint(nil, uint64(fieldNumber<<3|2))
	field = binary.AppendUvarint(field, uint64(len(value)))
	return append(field, value...)
}

func protobufVarintField(fieldNumber int, value uint64) []byte {
	field := binary.AppendUvarint(nil, uint64(fieldNumber<<3))
	return binary.AppendUvarint(field, value)
}

func tinySentencePieceModel() []byte {
	type piece struct {
		text string
		kind uint64
	}
	pieces := []piece{{"<unk>", 2}, {"<s>", 3}, {"</s>", 3}, {"▁", 1}, {"a", 1}, {"▁a", 1}}
	var model []byte
	for _, current := range pieces {
		message := protobufBytesField(1, []byte(current.text))
		message = append(message, protobufVarintField(3, current.kind)...)
		model = append(model, protobufBytesField(1, message)...)
	}
	trainerSpec := protobufVarintField(3, 2)
	return append(model, protobufBytesField(2, trainerSpec)...)
}

const tinyByteLevelTokenizerJSON = `{
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []}
}`

func writeFileInFolder(t *testing.T, folder string, name string, contents []byte) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(folder, name), contents, 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestLoadTokenizerUsesTokenizerModelWhenThereIsNoJSON(t *testing.T) {
	folder := t.TempDir()
	writeFileInFolder(t, folder, "tokenizer.model", tinySentencePieceModel())
	loaded, err := LoadTokenizer(folder)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Kind() != tokenizer.SentencePieceBPE {
		t.Errorf("kind %q, want %q", loaded.Kind(), tokenizer.SentencePieceBPE)
	}
	if got := loaded.Encode("<s>a"); len(got) != 2 || got[0] != 1 || got[1] != 5 {
		t.Errorf("<s>a encodes to %v, want [1 5]", got)
	}
}

func TestLoadTokenizerPrefersTokenizerJSON(t *testing.T) {
	folder := t.TempDir()
	writeFileInFolder(t, folder, "tokenizer.model", tinySentencePieceModel())
	writeFileInFolder(t, folder, "tokenizer.json", []byte(tinyByteLevelTokenizerJSON))
	loaded, err := LoadTokenizer(folder)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Kind() != tokenizer.ByteLevelBPE {
		t.Errorf("kind %q, want %q", loaded.Kind(), tokenizer.ByteLevelBPE)
	}
}

func TestLoadTokenizerNeedsATokenizerFile(t *testing.T) {
	if _, err := LoadTokenizer(t.TempDir()); err == nil || !strings.Contains(err.Error(), "tokenizer.model") {
		t.Errorf("an empty folder gave error %v", err)
	}
}

func TestMistralConfigPassesCheck(t *testing.T) {
	mistral := strings.Replace(goodConfig, `"model_type": "llama"`, `"model_type": "mistral", "sliding_window": 4096`, 1)
	config, err := ReadLlamaConfig(writeConfig(t, mistral))
	if err != nil {
		t.Fatal(err)
	}
	if err := config.Check(); err != nil {
		t.Errorf("a Mistral config fails Check: %v", err)
	}
}
