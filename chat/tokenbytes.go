package chat

import "github.com/javanhut/GoTransformers/tokenizer"

func VocabularyBytes(textTokenizer *tokenizer.Tokenizer) [][]byte {
	tokenBytes := make([][]byte, textTokenizer.VocabularySize())
	for tokenID := range tokenBytes {
		tokenBytes[tokenID] = textTokenizer.TokenBytes(tokenID)
	}
	return tokenBytes
}
