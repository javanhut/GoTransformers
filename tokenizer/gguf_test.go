package tokenizer

import (
	"encoding/binary"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"
)

const (
	ggufUint8   = 0
	ggufInt8    = 1
	ggufUint16  = 2
	ggufInt16   = 3
	ggufUint32  = 4
	ggufInt32   = 5
	ggufFloat32 = 6
	ggufBool    = 7
	ggufString  = 8
	ggufArray   = 9
	ggufUint64  = 10
	ggufInt64   = 11
	ggufFloat64 = 12
)

type ggufMetadata struct {
	numbers     map[string]float64
	texts       map[string]string
	numberLists map[string][]float64
	textLists   map[string][]string
}

type ggufReader struct {
	data     []byte
	position int
}

func (reader *ggufReader) readUint32() uint32 {
	value := binary.LittleEndian.Uint32(reader.data[reader.position:])
	reader.position += 4
	return value
}

func (reader *ggufReader) readUint64() uint64 {
	value := binary.LittleEndian.Uint64(reader.data[reader.position:])
	reader.position += 8
	return value
}

func (reader *ggufReader) readText() string {
	length := int(reader.readUint64())
	text := string(reader.data[reader.position : reader.position+length])
	reader.position += length
	return text
}

func (reader *ggufReader) readNumber(valueType uint32) float64 {
	start := reader.position
	switch valueType {
	case ggufUint8, ggufBool:
		reader.position += 1
		return float64(reader.data[start])
	case ggufInt8:
		reader.position += 1
		return float64(int8(reader.data[start]))
	case ggufUint16:
		reader.position += 2
		return float64(binary.LittleEndian.Uint16(reader.data[start:]))
	case ggufInt16:
		reader.position += 2
		return float64(int16(binary.LittleEndian.Uint16(reader.data[start:])))
	case ggufUint32:
		return float64(reader.readUint32())
	case ggufInt32:
		return float64(int32(reader.readUint32()))
	case ggufFloat32:
		return float64(math.Float32frombits(reader.readUint32()))
	case ggufUint64:
		return float64(reader.readUint64())
	case ggufInt64:
		return float64(int64(reader.readUint64()))
	case ggufFloat64:
		return math.Float64frombits(reader.readUint64())
	}
	panic("GGUF value type " + strconv.Itoa(int(valueType)) + " is not a number")
}

func (reader *ggufReader) readValueInto(metadata *ggufMetadata, key string, valueType uint32) {
	switch valueType {
	case ggufString:
		metadata.texts[key] = reader.readText()
	case ggufArray:
		elementType := reader.readUint32()
		count := int(reader.readUint64())
		for range count {
			if elementType == ggufString {
				metadata.textLists[key] = append(metadata.textLists[key], reader.readText())
				continue
			}
			metadata.numberLists[key] = append(metadata.numberLists[key], reader.readNumber(elementType))
		}
	default:
		metadata.numbers[key] = reader.readNumber(valueType)
	}
}

func readGGUFMetadata(t *testing.T, path string) ggufMetadata {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("GGUF vocabulary not found (%v)", err)
	}
	reader := &ggufReader{data: data}
	if string(data[:4]) != "GGUF" {
		t.Fatalf("%s does not start with GGUF", path)
	}
	reader.position = 4
	version := reader.readUint32()
	if version < 2 {
		t.Fatalf("%s is GGUF version %d, this reader needs version 2 or later", path, version)
	}
	reader.readUint64()
	numberOfKeys := int(reader.readUint64())
	metadata := ggufMetadata{
		numbers:     map[string]float64{},
		texts:       map[string]string{},
		numberLists: map[string][]float64{},
		textLists:   map[string][]string{},
	}
	for range numberOfKeys {
		key := reader.readText()
		valueType := reader.readUint32()
		reader.readValueInto(&metadata, key, valueType)
	}
	return metadata
}

func flagFromGGUF(metadata ggufMetadata, key string, fallback bool) bool {
	value, found := metadata.numbers[key]
	if !found {
		return fallback
	}
	return value != 0
}

func sentencePieceFromGGUF(t *testing.T, path string) *Tokenizer {
	t.Helper()
	metadata := readGGUFMetadata(t, path)
	if model := metadata.texts["tokenizer.ggml.model"]; model != "llama" {
		t.Fatalf("%s has tokenizer model %q, expected the SentencePiece model llama", path, model)
	}
	pieces := metadata.textLists["tokenizer.ggml.tokens"]
	scores := metadata.numberLists["tokenizer.ggml.scores"]
	types := metadata.numberLists["tokenizer.ggml.token_type"]
	if len(pieces) == 0 || len(scores) != len(pieces) || len(types) != len(pieces) {
		t.Fatalf("%s: %d tokens, %d scores, %d token types", path, len(pieces), len(scores), len(types))
	}
	entries := make([]sentencePieceEntry, len(pieces))
	for id := range pieces {
		entries[id] = sentencePieceEntry{piece: pieces[id], score: float32(scores[id]), kind: pieceType(types[id])}
	}
	settings := sentencePieceModel{
		mergeOrder:                   mergeHighestScoreFirst,
		addDummyPrefix:               flagFromGGUF(metadata, "tokenizer.ggml.add_space_prefix", true),
		replaceSpacesWithSpaceSymbol: true,
		removeExtraWhitespace:        flagFromGGUF(metadata, "tokenizer.ggml.remove_extra_whitespaces", false),
		byteFallback:                 true,
	}
	return newSentencePieceTokenizer(SentencePieceBPE, entries, settings)
}

type llamaCppTestCase struct {
	text     string
	expected []int
}

func readLlamaCppTestCases(t *testing.T, vocabularyPath string) []llamaCppTestCase {
	t.Helper()
	inputs, err := os.ReadFile(vocabularyPath + ".inp")
	if err != nil {
		t.Skipf("llama.cpp test inputs not found (%v)", err)
	}
	outputs, err := os.ReadFile(vocabularyPath + ".out")
	if err != nil {
		t.Skipf("llama.cpp test outputs not found (%v)", err)
	}
	const separator = "\n__ggml_vocab_test__\n"
	var texts []string
	remaining := string(inputs)
	for remaining != "" {
		text, rest, found := strings.Cut(remaining, separator)
		texts = append(texts, text)
		if !found {
			break
		}
		remaining = rest
	}
	lines := strings.Split(strings.TrimSuffix(string(outputs), "\n"), "\n")
	if len(lines) != len(texts) {
		t.Fatalf("%d inputs but %d expected outputs", len(texts), len(lines))
	}
	cases := make([]llamaCppTestCase, len(texts))
	for index, line := range lines {
		cases[index].text = texts[index]
		for _, field := range strings.Fields(line) {
			id, err := strconv.Atoi(field)
			if err != nil {
				t.Fatalf("output line %d: %v", index+1, err)
			}
			cases[index].expected = append(cases[index].expected, id)
		}
	}
	return cases
}

func sameIDs(first []int, second []int) bool {
	if len(first) != len(second) {
		return false
	}
	for index := range first {
		if first[index] != second[index] {
			return false
		}
	}
	return true
}

func checkLlamaCppCases(t *testing.T, tokenizer *Tokenizer, cases []llamaCppTestCase) {
	t.Helper()
	passed := 0
	for index, testCase := range cases {
		got := tokenizer.Encode(testCase.text)
		if !sameIDs(got, testCase.expected) {
			t.Errorf("case %d %q:\n got  %v\n want %v", index+1, testCase.text, got, testCase.expected)
			continue
		}
		if decoded := tokenizer.Decode(got); decoded != testCase.text {
			t.Errorf("case %d %q decodes back to %q", index+1, testCase.text, decoded)
			continue
		}
		passed++
	}
	t.Logf("%d of %d llama.cpp cases encode to the expected IDs and decode back to the input", passed, len(cases))
}

const llamaSentencePieceVocabulary = "testdata/ggml-vocab-llama-spm.gguf"

func TestLlamaSentencePieceMatchesLlamaCpp(t *testing.T) {
	tokenizer := sentencePieceFromGGUF(t, llamaSentencePieceVocabulary)
	cases := readLlamaCppTestCases(t, llamaSentencePieceVocabulary)
	if len(cases) != 46 {
		t.Errorf("expected 46 llama.cpp cases, read %d", len(cases))
	}
	checkLlamaCppCases(t, tokenizer, cases)
}

func TestPhi3SentencePieceMatchesLlamaCpp(t *testing.T) {
	path := "/home/javanstorm/Development/llama.cpp/models/ggml-vocab-phi-3.gguf"
	tokenizer := sentencePieceFromGGUF(t, path)
	checkLlamaCppCases(t, tokenizer, readLlamaCppTestCases(t, path))
}

func TestLlamaSentencePieceSavesAndLoads(t *testing.T) {
	original := sentencePieceFromGGUF(t, llamaSentencePieceVocabulary)
	path := t.TempDir() + "/llama.tokenizer"
	if err := original.SaveToFile(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadFromFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Kind() != SentencePieceBPE || loaded.VocabularySize() != original.VocabularySize() {
		t.Fatalf("loaded kind %q with %d tokens, want %q with %d", loaded.Kind(), loaded.VocabularySize(), SentencePieceBPE, original.VocabularySize())
	}
	checkLlamaCppCases(t, loaded, readLlamaCppTestCases(t, llamaSentencePieceVocabulary))
	if id, found := loaded.SpecialTokenID("</s>"); !found || id != 2 {
		t.Errorf("</s> is special token %d (found %v), want 2", id, found)
	}
}

func TestLlamaSentencePieceTokenBytes(t *testing.T) {
	tokenizer := sentencePieceFromGGUF(t, llamaSentencePieceVocabulary)
	expectations := map[int]string{
		15043: " Hello",
		3186:  " world",
		13:    "\n",
		29871: " ",
		259:   "  ",
		29991: "!",
	}
	for id, want := range expectations {
		if got := tokenizer.TokenBytes(id); string(got) != want {
			t.Errorf("TokenBytes(%d) = %q, want %q", id, got, want)
		}
	}
	for _, id := range []int{0, 1, 2, -1, tokenizer.VocabularySize()} {
		if got := tokenizer.TokenBytes(id); got != nil {
			t.Errorf("TokenBytes(%d) = %q, want nil", id, got)
		}
	}
	if got := tokenizer.TokenBytes(3 + 0xE2); len(got) != 1 || got[0] != 0xE2 {
		t.Errorf("TokenBytes of <0xE2> = %v, want [0xE2]", got)
	}
	ids := tokenizer.Encode("Hello world, ✓ é 🦙!")
	var joined []byte
	for _, id := range ids {
		joined = append(joined, tokenizer.TokenBytes(id)...)
	}
	if string(joined) != " Hello world, ✓ é 🦙!" {
		t.Errorf("TokenBytes of every token joins to %q", joined)
	}
}

func TestLlamaSentencePieceSpecialTokens(t *testing.T) {
	tokenizer := sentencePieceFromGGUF(t, llamaSentencePieceVocabulary)
	got := tokenizer.Encode("<s>[INST] Hello [/INST]</s>")
	want := []int{1, 518, 25580, 29962, 15043, 518, 29914, 25580, 29962, 2}
	if !sameIDs(got, want) {
		t.Errorf("chat markup encodes to %v, want %v", got, want)
	}
	if decoded := tokenizer.Decode(got); decoded != "<s>[INST] Hello [/INST]</s>" {
		t.Errorf("chat markup decodes to %q", decoded)
	}
}
