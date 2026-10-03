package tokenizer

import (
	"encoding/json"
	"fmt"
	"strings"
)

func mentionsSpaceSymbol(raw json.RawMessage) bool {
	text := string(raw)
	return strings.Contains(text, spaceSymbol) || strings.Contains(text, `▁`)
}

func isSentencePieceStyle(file huggingFaceFile) bool {
	return file.Model.ByteFallback || mentionsSpaceSymbol(file.Normalizer) || mentionsSpaceSymbol(file.PreTokenizer)
}

func readSentencePieceNormalizer(raw json.RawMessage, settings *sentencePieceModel) error {
	if isJSONNull(raw) {
		return nil
	}
	var normalizer huggingFaceNormalizer
	if err := json.Unmarshal(raw, &normalizer); err != nil {
		return fmt.Errorf("normalizer is not valid: %w", err)
	}
	switch normalizer.Type {
	case "Sequence":
		for _, inner := range normalizer.Normalizers {
			if err := readSentencePieceNormalizer(inner, settings); err != nil {
				return err
			}
		}
		return nil
	case "Prepend":
		if normalizer.Prepend != spaceSymbol {
			return fmt.Errorf("Prepend normalizer adds %q, only %q is supported", normalizer.Prepend, spaceSymbol)
		}
		settings.addDummyPrefix = true
		return nil
	case "Replace":
		if normalizer.Pattern["String"] != " " || normalizer.Content != spaceSymbol {
			return fmt.Errorf("Replace normalizer %v -> %q is not supported, only replacing spaces with %q is", normalizer.Pattern, normalizer.Content, spaceSymbol)
		}
		settings.replaceSpacesWithSpaceSymbol = true
		return nil
	}
	return fmt.Errorf("normalizer %q is not supported in a SentencePiece-style tokenizer", normalizer.Type)
}

func readSentencePiecePreTokenizer(raw json.RawMessage, settings *sentencePieceModel) error {
	if isJSONNull(raw) {
		return nil
	}
	var preTokenizer huggingFacePreTokenizer
	if err := json.Unmarshal(raw, &preTokenizer); err != nil {
		return fmt.Errorf("pre_tokenizer is not valid: %w", err)
	}
	switch preTokenizer.Type {
	case "Sequence":
		for _, inner := range preTokenizer.PreTokenizers {
			if err := readSentencePiecePreTokenizer(inner, settings); err != nil {
				return err
			}
		}
		return nil
	case "Metaspace":
		return readMetaspace(preTokenizer, settings)
	}
	return fmt.Errorf("pre-tokenizer %q is not supported in a SentencePiece-style tokenizer", preTokenizer.Type)
}

func readMetaspace(preTokenizer huggingFacePreTokenizer, settings *sentencePieceModel) error {
	if preTokenizer.Replacement != spaceSymbol {
		return fmt.Errorf("Metaspace replacement %q is not supported, only %q is", preTokenizer.Replacement, spaceSymbol)
	}
	settings.replaceSpacesWithSpaceSymbol = true
	settings.splitBeforeEachSpaceSymbol = preTokenizer.Split == nil || *preTokenizer.Split
	scheme := preTokenizer.PrependScheme
	if scheme == "" && preTokenizer.AddPrefixSpace {
		scheme = "always"
	}
	switch scheme {
	case "always":
		settings.addDummyPrefix = true
		settings.skipDummyPrefixWhenTextStartsWithSpace = true
	case "first":
		settings.addDummyPrefix = true
		settings.skipDummyPrefixWhenTextStartsWithSpace = true
		settings.dummyPrefixOnlyAtStartOfText = true
	case "never", "":
	default:
		return fmt.Errorf("Metaspace prepend_scheme %q is not supported", scheme)
	}
	return nil
}

func sentencePieceEntriesFromHuggingFace(file huggingFaceFile) ([]sentencePieceEntry, error) {
	highestID := -1
	for token, id := range file.Model.Vocab {
		if id < 0 {
			return nil, fmt.Errorf("token %q has a negative ID %d", token, id)
		}
		highestID = max(highestID, id)
	}
	for _, added := range file.AddedTokens {
		if added.ID < 0 {
			return nil, fmt.Errorf("added token %q has a negative ID %d", added.Content, added.ID)
		}
		highestID = max(highestID, added.ID)
	}
	entries := make([]sentencePieceEntry, highestID+1)
	for id := range entries {
		entries[id].kind = controlPiece
	}
	for token, id := range file.Model.Vocab {
		entries[id] = sentencePieceEntry{piece: token, kind: normalPiece}
		if _, isByte := byteValueOfPieceText(token); isByte {
			entries[id].kind = bytePiece
		}
	}
	for _, added := range file.AddedTokens {
		entries[added.ID] = sentencePieceEntry{piece: added.Content, kind: controlPiece}
	}
	if file.Model.UnknownToken != nil {
		unknownID, found := file.Model.Vocab[*file.Model.UnknownToken]
		if !found {
			return nil, fmt.Errorf("unk_token %q is not in the vocabulary", *file.Model.UnknownToken)
		}
		entries[unknownID].kind = unknownPiece
	}
	return entries, nil
}

func sentencePieceFromHuggingFace(file huggingFaceFile) (*Tokenizer, error) {
	settings := sentencePieceModel{
		mergeOrder:   mergeEarliestListedFirst,
		byteFallback: file.Model.ByteFallback,
	}
	if err := readSentencePieceNormalizer(file.Normalizer, &settings); err != nil {
		return nil, err
	}
	if err := readSentencePiecePreTokenizer(file.PreTokenizer, &settings); err != nil {
		return nil, err
	}
	merges, err := readMerges(file.Model.Merges)
	if err != nil {
		return nil, err
	}
	entries, err := sentencePieceEntriesFromHuggingFace(file)
	if err != nil {
		return nil, err
	}
	tokenizer := newSentencePieceTokenizer(SentencePieceBPE, entries, settings)
	for _, merge := range merges {
		tokenizer.addMerge(merge.left, merge.right)
	}
	return tokenizer, nil
}
