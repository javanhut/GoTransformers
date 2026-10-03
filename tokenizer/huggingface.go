package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

type huggingFaceAddedToken struct {
	ID      int    `json:"id"`
	Content string `json:"content"`
	Special bool   `json:"special"`
}

type huggingFaceModel struct {
	Type                    string          `json:"type"`
	Vocab                   map[string]int  `json:"vocab"`
	Merges                  json.RawMessage `json:"merges"`
	ByteFallback            bool            `json:"byte_fallback"`
	ContinuingSubwordPrefix *string         `json:"continuing_subword_prefix"`
	EndOfWordSuffix         *string         `json:"end_of_word_suffix"`
	IgnoreMerges            bool            `json:"ignore_merges"`
	UnknownToken            *string         `json:"unk_token"`
}

type huggingFaceFile struct {
	AddedTokens  []huggingFaceAddedToken `json:"added_tokens"`
	Normalizer   json.RawMessage         `json:"normalizer"`
	PreTokenizer json.RawMessage         `json:"pre_tokenizer"`
	Model        huggingFaceModel        `json:"model"`
}

type huggingFacePreTokenizer struct {
	Type             string            `json:"type"`
	UseRegex         *bool             `json:"use_regex"`
	AddPrefixSpace   bool              `json:"add_prefix_space"`
	IndividualDigits bool              `json:"individual_digits"`
	Pattern          map[string]string `json:"pattern"`
	Behavior         string            `json:"behavior"`
	Invert           bool              `json:"invert"`
	Replacement      string            `json:"replacement"`
	PrependScheme    string            `json:"prepend_scheme"`
	Split            *bool             `json:"split"`
	PreTokenizers    []json.RawMessage `json:"pretokenizers"`
}

type huggingFaceNormalizer struct {
	Type        string            `json:"type"`
	Normalizers []json.RawMessage `json:"normalizers"`
	Prepend     string            `json:"prepend"`
	Pattern     map[string]string `json:"pattern"`
	Content     string            `json:"content"`
}

func isJSONNull(raw json.RawMessage) bool {
	trimmed := strings.TrimSpace(string(raw))
	return trimmed == "" || trimmed == "null"
}

func readPreTokenizer(raw json.RawMessage, sawByteLevel *bool, addPrefixSpace *bool) ([]preTokenizerStep, error) {
	var preTokenizer huggingFacePreTokenizer
	if err := json.Unmarshal(raw, &preTokenizer); err != nil {
		return nil, fmt.Errorf("pre_tokenizer is not valid: %w", err)
	}
	switch preTokenizer.Type {
	case "Sequence":
		var steps []preTokenizerStep
		for _, inner := range preTokenizer.PreTokenizers {
			innerSteps, err := readPreTokenizer(inner, sawByteLevel, addPrefixSpace)
			if err != nil {
				return nil, err
			}
			steps = append(steps, innerSteps...)
		}
		return steps, nil
	case "ByteLevel":
		*sawByteLevel = true
		*addPrefixSpace = preTokenizer.AddPrefixSpace
		if preTokenizer.UseRegex != nil && !*preTokenizer.UseRegex {
			return nil, nil
		}
		return []preTokenizerStep{{splitStyle: GPT2Split}}, nil
	case "Digits":
		if preTokenizer.IndividualDigits {
			return []preTokenizerStep{{digits: IsolateEachDigit}}, nil
		}
		return []preTokenizerStep{{digits: IsolateRunsOfDigits}}, nil
	case "Split":
		if preTokenizer.Invert || (preTokenizer.Behavior != "Isolated" && preTokenizer.Behavior != "") {
			return nil, fmt.Errorf("Split pre-tokenizer with behavior %q and invert %v is not supported, only Isolated", preTokenizer.Behavior, preTokenizer.Invert)
		}
		pattern := preTokenizer.Pattern["Regex"]
		switch pattern {
		case gpt2Pattern:
			return []preTokenizerStep{{splitStyle: GPT2Split}}, nil
		case llama3Pattern:
			return []preTokenizerStep{{splitStyle: Llama3Split}}, nil
		case qwen2Pattern:
			return []preTokenizerStep{{splitStyle: Qwen2Split}}, nil
		}
		return nil, fmt.Errorf("Split pre-tokenizer pattern %q is not one of the GPT-2, Llama 3 or Qwen 2 patterns this tokenizer knows", pattern)
	}
	return nil, fmt.Errorf("pre-tokenizer type %q is not supported", preTokenizer.Type)
}

func readMerges(raw json.RawMessage) ([]mergePair, error) {
	var asStrings []string
	if err := json.Unmarshal(raw, &asStrings); err == nil {
		merges := make([]mergePair, 0, len(asStrings))
		for index, merge := range asStrings {
			left, right, found := strings.Cut(merge, " ")
			if !found {
				return nil, fmt.Errorf("merge %d %q has no space between its two parts", index, merge)
			}
			merges = append(merges, mergePair{left: left, right: right})
		}
		return merges, nil
	}
	var asPairs [][]string
	if err := json.Unmarshal(raw, &asPairs); err != nil {
		return nil, fmt.Errorf("merges are neither a list of strings nor a list of pairs: %w", err)
	}
	merges := make([]mergePair, 0, len(asPairs))
	for index, pair := range asPairs {
		if len(pair) != 2 {
			return nil, fmt.Errorf("merge %d has %d parts, it needs 2", index, len(pair))
		}
		merges = append(merges, mergePair{left: pair[0], right: pair[1]})
	}
	return merges, nil
}

func LoadHuggingFace(path string) (*Tokenizer, error) {
	contents, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var file huggingFaceFile
	if err := json.Unmarshal(contents, &file); err != nil {
		return nil, fmt.Errorf("%s: not a valid tokenizer.json: %w", path, err)
	}
	model := file.Model
	if model.Type != "BPE" {
		return nil, fmt.Errorf("%s: model type %q is not supported, only BPE is", path, model.Type)
	}
	if (model.ContinuingSubwordPrefix != nil && *model.ContinuingSubwordPrefix != "") || (model.EndOfWordSuffix != nil && *model.EndOfWordSuffix != "") {
		return nil, fmt.Errorf("%s: BPE with subword prefixes or suffixes is not supported, only byte-level BPE (GPT-2, SmolLM2, Llama 3, Qwen 2) and SentencePiece-style BPE (Llama 2, Mistral)", path)
	}
	if isSentencePieceStyle(file) {
		tokenizer, err := sentencePieceFromHuggingFace(file)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		return tokenizer, nil
	}
	if !isJSONNull(file.Normalizer) {
		var normalizer huggingFaceNormalizer
		if err := json.Unmarshal(file.Normalizer, &normalizer); err != nil {
			return nil, fmt.Errorf("%s: normalizer is not valid: %w", path, err)
		}
		if normalizer.Type != "NFC" {
			return nil, fmt.Errorf("%s: normalizer %q is not supported", path, normalizer.Type)
		}
	}
	if isJSONNull(file.PreTokenizer) {
		return nil, fmt.Errorf("%s: has no pre-tokenizer, only byte-level BPE tokenizers are supported", path)
	}
	sawByteLevel := false
	addPrefixSpace := false
	steps, err := readPreTokenizer(file.PreTokenizer, &sawByteLevel, &addPrefixSpace)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if !sawByteLevel {
		return nil, fmt.Errorf("%s: has no ByteLevel pre-tokenizer, only byte-level BPE tokenizers are supported", path)
	}
	merges, err := readMerges(model.Merges)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}

	tokenizer := newEmptyTokenizer(steps)
	tokenizer.addPrefixSpace = addPrefixSpace
	tokenizer.useWholeWordsFromVocabulary = model.IgnoreMerges
	for token, id := range model.Vocab {
		if id < 0 {
			return nil, fmt.Errorf("%s: token %q has a negative ID %d", path, token, id)
		}
		tokenizer.setToken(id, token)
	}
	for _, merge := range merges {
		tokenizer.addMerge(merge.left, merge.right)
	}
	for _, added := range file.AddedTokens {
		tokenizer.addSpecialTokenWithID(added.ID, added.Content)
	}
	return tokenizer, nil
}
