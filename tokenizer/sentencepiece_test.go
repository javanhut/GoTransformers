package tokenizer

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"unicode/utf8"
)

type generatedMerge struct {
	left  string
	right string
	score float32
}

func mergesLikeHuggingFaceConverter(tokenizer *Tokenizer) [][]string {
	var merges []generatedMerge
	for id, piece := range tokenizer.idToToken {
		characters := []rune(piece)
		var mergesForThisPiece []generatedMerge
		for index := 1; index < len(characters); index++ {
			left := string(characters[:index])
			right := string(characters[index:])
			_, leftFound := tokenizer.tokenToID[left]
			_, rightFound := tokenizer.tokenToID[right]
			if leftFound && rightFound {
				mergesForThisPiece = append(mergesForThisPiece, generatedMerge{left: left, right: right, score: tokenizer.sentencePiece.scores[id]})
			}
		}
		sort.SliceStable(mergesForThisPiece, func(i int, j int) bool {
			first := mergesForThisPiece[i]
			second := mergesForThisPiece[j]
			if tokenizer.tokenToID[first.left] != tokenizer.tokenToID[second.left] {
				return tokenizer.tokenToID[first.left] < tokenizer.tokenToID[second.left]
			}
			return tokenizer.tokenToID[first.right] < tokenizer.tokenToID[second.right]
		})
		merges = append(merges, mergesForThisPiece...)
	}
	sort.SliceStable(merges, func(i int, j int) bool {
		if merges[i].score != merges[j].score {
			return merges[i].score > merges[j].score
		}
		leftLengthI := utf8.RuneCountInString(merges[i].left)
		leftLengthJ := utf8.RuneCountInString(merges[j].left)
		if leftLengthI != leftLengthJ {
			return leftLengthI > leftLengthJ
		}
		return utf8.RuneCountInString(merges[i].right) > utf8.RuneCountInString(merges[j].right)
	})
	pairs := make([][]string, len(merges))
	for index, merge := range merges {
		pairs[index] = []string{merge.left, merge.right}
	}
	return pairs
}

func writeLlamaStyleTokenizerJSON(t *testing.T, normalizer any, preTokenizer any) string {
	t.Helper()
	fromGGUF := sentencePieceFromGGUF(t, llamaSentencePieceVocabulary)
	vocabulary := map[string]int{}
	for id, piece := range fromGGUF.idToToken {
		vocabulary[piece] = id
	}
	file := map[string]any{
		"added_tokens": []map[string]any{
			{"id": 0, "content": "<unk>", "special": true},
			{"id": 1, "content": "<s>", "special": true},
			{"id": 2, "content": "</s>", "special": true},
		},
		"normalizer":    normalizer,
		"pre_tokenizer": preTokenizer,
		"model": map[string]any{
			"type":          "BPE",
			"vocab":         vocabulary,
			"merges":        mergesLikeHuggingFaceConverter(fromGGUF),
			"byte_fallback": true,
			"fuse_unk":      true,
			"unk_token":     "<unk>",
		},
	}
	contents, err := json.Marshal(file)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	if err := os.WriteFile(path, contents, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func legacyLlamaNormalizer() any {
	return map[string]any{
		"type": "Sequence",
		"normalizers": []any{
			map[string]any{"type": "Prepend", "prepend": "▁"},
			map[string]any{"type": "Replace", "pattern": map[string]any{"String": " "}, "content": "▁"},
		},
	}
}

func metaspacePreTokenizer(prependScheme string, split bool) any {
	return map[string]any{"type": "Metaspace", "replacement": "▁", "prepend_scheme": prependScheme, "split": split}
}

func TestLlamaStyleTokenizerJSONMatchesLlamaCpp(t *testing.T) {
	path := writeLlamaStyleTokenizerJSON(t, legacyLlamaNormalizer(), nil)
	loaded, err := LoadHuggingFace(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Kind() != SentencePieceBPE {
		t.Fatalf("kind is %q, want %q", loaded.Kind(), SentencePieceBPE)
	}
	checkLlamaCppCases(t, loaded, readLlamaCppTestCases(t, llamaSentencePieceVocabulary))
	got := loaded.Encode("<s>[INST] Hello [/INST]</s>")
	want := []int{1, 518, 25580, 29962, 15043, 518, 29914, 25580, 29962, 2}
	if !sameIDs(got, want) {
		t.Errorf("chat markup encodes to %v, want %v", got, want)
	}
	savedPath := filepath.Join(t.TempDir(), "saved.tokenizer")
	if err := loaded.SaveToFile(savedPath); err != nil {
		t.Fatal(err)
	}
	reloaded, err := LoadFromFile(savedPath)
	if err != nil {
		t.Fatal(err)
	}
	checkLlamaCppCases(t, reloaded, readLlamaCppTestCases(t, llamaSentencePieceVocabulary))
}

func TestMetaspaceTokenizerJSON(t *testing.T) {
	path := writeLlamaStyleTokenizerJSON(t, nil, metaspacePreTokenizer("first", false))
	loaded, err := LoadHuggingFace(path)
	if err != nil {
		t.Fatal(err)
	}
	checkLlamaCppCases(t, loaded, casesNotStartingWithSpace(readLlamaCppTestCases(t, llamaSentencePieceVocabulary)))
	helloWithoutSpace, _ := loaded.TokenID("Hello")
	helloWithSpace, _ := loaded.TokenID("▁Hello")
	if got := loaded.Encode("<s>Hello"); !sameIDs(got, []int{1, helloWithoutSpace}) {
		t.Errorf("prepend_scheme first: <s>Hello encodes to %v, want [1 %d]", got, helloWithoutSpace)
	}
	if got := loaded.Encode("Hello"); !sameIDs(got, []int{helloWithSpace}) {
		t.Errorf("prepend_scheme first: Hello encodes to %v, want [%d]", got, helloWithSpace)
	}
	if got := loaded.Encode(" Hello"); !sameIDs(got, []int{helloWithSpace}) {
		t.Errorf("Metaspace adds no second space before text that starts with one: \" Hello\" encodes to %v, want [%d]", got, helloWithSpace)
	}

	splitPath := writeLlamaStyleTokenizerJSON(t, nil, metaspacePreTokenizer("always", true))
	splitting, err := LoadHuggingFace(splitPath)
	if err != nil {
		t.Fatal(err)
	}
	space, _ := splitting.TokenID("▁")
	if got := splitting.Encode("  Hello"); !sameIDs(got, []int{space, helloWithSpace}) {
		t.Errorf("split Metaspace: two spaces then Hello encodes to %v, want [%d %d]", got, space, helloWithSpace)
	}
	if got := splitting.Encode("<s>Hello"); !sameIDs(got, []int{1, helloWithSpace}) {
		t.Errorf("prepend_scheme always: <s>Hello encodes to %v, want [1 %d]", got, helloWithSpace)
	}
}

type protobufWriter struct {
	data []byte
}

func (writer *protobufWriter) fieldHeader(fieldNumber int, wireType int) {
	writer.data = binary.AppendUvarint(writer.data, uint64(fieldNumber<<3|wireType))
}

func (writer *protobufWriter) varintField(fieldNumber int, value uint64) {
	writer.fieldHeader(fieldNumber, wireTypeVarint)
	writer.data = binary.AppendUvarint(writer.data, value)
}

func (writer *protobufWriter) bytesField(fieldNumber int, value []byte) {
	writer.fieldHeader(fieldNumber, wireTypeLengthDelimited)
	writer.data = binary.AppendUvarint(writer.data, uint64(len(value)))
	writer.data = append(writer.data, value...)
}

func (writer *protobufWriter) float32Field(fieldNumber int, value float32) {
	writer.fieldHeader(fieldNumber, wireTypeFixed32)
	writer.data = binary.LittleEndian.AppendUint32(writer.data, math.Float32bits(value))
}

func encodePieceMessage(entry sentencePieceEntry) []byte {
	piece := &protobufWriter{}
	piece.bytesField(1, []byte(entry.piece))
	piece.float32Field(2, entry.score)
	if entry.kind != normalPiece {
		piece.varintField(3, uint64(entry.kind))
	}
	return piece.data
}

type handMadeModel struct {
	entries                []sentencePieceEntry
	modelType              int
	byteFallback           bool
	removeExtraWhitespaces *bool
	precompiledCharsmap    []byte
}

func encodeModelProto(model handMadeModel) []byte {
	trainer := &protobufWriter{}
	trainer.varintField(3, uint64(model.modelType))
	trainer.varintField(4, uint64(len(model.entries)))
	trainer.float32Field(10, 0.9995)
	if model.byteFallback {
		trainer.varintField(35, 1)
	}
	normalizer := &protobufWriter{}
	normalizer.bytesField(1, []byte("identity"))
	if len(model.precompiledCharsmap) > 0 {
		normalizer.bytesField(2, model.precompiledCharsmap)
	}
	if model.removeExtraWhitespaces != nil && !*model.removeExtraWhitespaces {
		normalizer.varintField(4, 0)
	}
	whole := &protobufWriter{}
	for _, entry := range model.entries {
		whole.bytesField(1, encodePieceMessage(entry))
	}
	whole.bytesField(2, trainer.data)
	whole.bytesField(3, normalizer.data)
	whole.bytesField(4, []byte{0x0A, 0x00})
	return whole.data
}

func writeModelProto(t *testing.T, model handMadeModel) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tokenizer.model")
	if err := os.WriteFile(path, encodeModelProto(model), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func smallBPEEntries() []sentencePieceEntry {
	entries := []sentencePieceEntry{
		{piece: "<unk>", kind: unknownPiece},
		{piece: "<s>", kind: controlPiece},
		{piece: "</s>", kind: controlPiece},
	}
	for value := range 256 {
		entries = append(entries, sentencePieceEntry{piece: fmt.Sprintf("<0x%02X>", value), kind: bytePiece})
	}
	normal := []sentencePieceEntry{
		{piece: "▁", score: -10}, {piece: "t", score: -10}, {piece: "h", score: -10},
		{piece: "e", score: -10}, {piece: "a", score: -10}, {piece: "b", score: -10},
		{piece: "th", score: -1}, {piece: "▁t", score: -2}, {piece: "he", score: -3},
		{piece: "▁the", score: -4}, {piece: "the", score: -5}, {piece: "ab", score: -6},
	}
	for _, entry := range normal {
		entry.kind = normalPiece
		entries = append(entries, entry)
	}
	entries = append(entries, sentencePieceEntry{piece: "▁a", score: -0.5, kind: unusedPiece})
	entries = append(entries, sentencePieceEntry{piece: "<sep>", kind: userDefinedPiece})
	return entries
}

func idsOfPieces(t *testing.T, tokenizer *Tokenizer, pieces ...string) []int {
	t.Helper()
	var ids []int
	for _, piece := range pieces {
		id, found := tokenizer.TokenID(piece)
		if !found {
			t.Fatalf("piece %q is not in the vocabulary", piece)
		}
		ids = append(ids, id)
	}
	return ids
}

func TestSentencePieceModelBPE(t *testing.T) {
	keepWhitespace := false
	path := writeModelProto(t, handMadeModel{entries: smallBPEEntries(), modelType: sentencePieceBPEModelType, byteFallback: true, removeExtraWhitespaces: &keepWhitespace})
	loaded, err := LoadSentencePiece(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Kind() != SentencePieceBPE || loaded.VocabularySize() != 273 {
		t.Fatalf("kind %q with %d tokens, want %q with 273", loaded.Kind(), loaded.VocabularySize(), SentencePieceBPE)
	}
	expectations := []struct {
		text   string
		pieces []string
	}{
		{text: "the", pieces: []string{"▁the"}},
		{text: "a<sep>b", pieces: []string{"▁", "a", "<sep>", "b"}},
		{text: "ab é", pieces: []string{"▁", "a", "b", "▁", "<0xC3>", "<0xA9>"}},
		{text: "<s>the</s>", pieces: []string{"<s>", "▁the", "</s>"}},
		{text: "", pieces: nil},
	}
	for _, expectation := range expectations {
		want := idsOfPieces(t, loaded, expectation.pieces...)
		got := loaded.Encode(expectation.text)
		if !sameIDs(got, want) {
			t.Errorf("%q encodes to %v, want %v (%v)", expectation.text, got, want, expectation.pieces)
		}
		if decoded := loaded.Decode(got); decoded != expectation.text {
			t.Errorf("%q decodes back to %q", expectation.text, decoded)
		}
	}
	if got := loaded.TokenBytes(idsOfPieces(t, loaded, "<sep>")[0]); string(got) != "<sep>" {
		t.Errorf("user-defined piece bytes are %q", got)
	}
}

func TestSentencePieceModelUnigram(t *testing.T) {
	entries := []sentencePieceEntry{
		{piece: "<unk>", kind: unknownPiece},
		{piece: "<s>", kind: controlPiece},
		{piece: "</s>", kind: controlPiece},
		{piece: "▁", score: -2, kind: normalPiece},
		{piece: "a", score: -3, kind: normalPiece},
		{piece: "b", score: -3, kind: normalPiece},
		{piece: "▁ab", score: -4, kind: normalPiece},
		{piece: "▁a", score: -2.5, kind: normalPiece},
	}
	path := writeModelProto(t, handMadeModel{entries: entries, modelType: sentencePieceUnigramModelType})
	loaded, err := LoadSentencePiece(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Kind() != SentencePieceUnigram {
		t.Fatalf("kind %q, want %q", loaded.Kind(), SentencePieceUnigram)
	}
	expectations := map[string][]int{
		"ab":         {6},
		"abab":       {6, 4, 5},
		"  ab   zz ": {6, 3, 0},
		"":           nil,
	}
	for text, want := range expectations {
		if got := loaded.Encode(text); !sameIDs(got, want) {
			t.Errorf("%q encodes to %v, want %v", text, got, want)
		}
	}
	if decoded := loaded.Decode([]int{6, 4, 5}); decoded != "abab" {
		t.Errorf("decoded %q", decoded)
	}
	savedPath := filepath.Join(t.TempDir(), "unigram.tokenizer")
	if err := loaded.SaveToFile(savedPath); err != nil {
		t.Fatal(err)
	}
	reloaded, err := LoadFromFile(savedPath)
	if err != nil {
		t.Fatal(err)
	}
	for text, want := range expectations {
		if got := reloaded.Encode(text); !sameIDs(got, want) {
			t.Errorf("after save and load, %q encodes to %v, want %v", text, got, want)
		}
	}
}

func TestSentencePieceModelRejectsWhatItCannotDo(t *testing.T) {
	charsmapPath := writeModelProto(t, handMadeModel{entries: smallBPEEntries(), modelType: sentencePieceUnigramModelType, precompiledCharsmap: []byte{1, 2, 3}})
	if _, err := LoadSentencePiece(charsmapPath); err == nil || !strings.Contains(err.Error(), "precompiled character map") {
		t.Errorf("a precompiled character map gave error %v", err)
	}
	wordPath := writeModelProto(t, handMadeModel{entries: smallBPEEntries(), modelType: sentencePieceWordModelType})
	if _, err := LoadSentencePiece(wordPath); err == nil || !strings.Contains(err.Error(), "WORD") {
		t.Errorf("a WORD model gave error %v", err)
	}
	brokenPath := filepath.Join(t.TempDir(), "broken.model")
	if err := os.WriteFile(brokenPath, []byte{0x0A, 0x50, 0x01}, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadSentencePiece(brokenPath); err == nil {
		t.Errorf("a truncated model loaded without error")
	}
}

func TestFirstSaveFileFormatStillLoads(t *testing.T) {
	path := filepath.Join(t.TempDir(), "old.tokenizer")
	contents := strings.Join([]string{
		"gotransformers-tokenizer-1",
		"addPrefixSpace false",
		"useWholeWordsFromVocabulary false",
		"steps 1",
		`"gpt2" ""`,
		"vocabulary 4",
		`"a"`,
		`"b"`,
		`"ab"`,
		`"<|end|>"`,
		"merges 1",
		`"a" "b"`,
		"special 1",
		`3 "<|end|>"`,
		"",
	}, "\n")
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadFromFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Kind() != ByteLevelBPE {
		t.Errorf("kind %q, want %q", loaded.Kind(), ByteLevelBPE)
	}
	if got := loaded.Encode("ab<|end|>"); !sameIDs(got, []int{2, 3}) {
		t.Errorf("ab<|end|> encodes to %v, want [2 3]", got)
	}
}

func TestByteLevelTokenBytes(t *testing.T) {
	trained := Train(strings.Repeat("the quick brown fox jumps over the lazy dog. ", 20)+"naïve café ✓", 300)
	endID := trained.AddSpecialToken("<|end|>")
	for id := range trained.VocabularySize() {
		got := trained.TokenBytes(id)
		if id == endID {
			if got != nil {
				t.Errorf("special token bytes are %q, want nil", got)
			}
			continue
		}
		if want := trained.Decode([]int{id}); string(got) != want {
			t.Errorf("TokenBytes(%d) = %q, want %q", id, got, want)
		}
	}
	if trained.Kind() != ByteLevelBPE {
		t.Errorf("trained tokenizer kind %q", trained.Kind())
	}
}

func casesNotStartingWithSpace(cases []llamaCppTestCase) []llamaCppTestCase {
	var kept []llamaCppTestCase
	for _, testCase := range cases {
		if !strings.HasPrefix(testCase.text, " ") {
			kept = append(kept, testCase)
		}
	}
	return kept
}
