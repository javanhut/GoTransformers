package constrained

import (
	"encoding/json"
	"math/rand/v2"
	"testing"
)

var toyJSONTokens = []string{
	"{", "}", "[", "]", ":", ",", "\"", "\"a", "\"b\"", "abc", "é", "\xc3", "\xa9", "\xf0\x9f", "\x98\x80",
	"\\", "\\\"", "n", "ull", "u00e9", "\\u", "12", "1", "0", "-", ".", ".5", "e", "e+", "2.5e3",
	"true", "fal", "se", "null", " ", "\n  ", "\"},", "\"]", "],", "},", "\":", "\":\"", "\"}", "{\"", "[\"", ",\"",
	"\"x\":", "}]", "]}", "<|im_end|>",
}

func toyTokenBytes(tokens []string) [][]byte {
	tokenBytes := make([][]byte, len(tokens))
	for tokenID, token := range tokens {
		tokenBytes[tokenID] = []byte(token)
	}
	tokenBytes[len(tokens)-1] = nil
	return tokenBytes
}

func allowedTokenIDs(allowedTokens []bool) []int {
	var tokenIDs []int
	for tokenID, allowed := range allowedTokens {
		if allowed {
			tokenIDs = append(tokenIDs, tokenID)
		}
	}
	return tokenIDs
}

func checkMaskMatchesSingleChecks(t *testing.T, constraint *TokenConstraint) {
	allowedTokens := constraint.AllowedTokens()
	for tokenID := range allowedTokens {
		if allowedTokens[tokenID] != constraint.IsTokenAllowed(tokenID) {
			t.Fatalf("after %q the mask says token %d (%q) is %t but checking it alone says %t",
				constraint.Text(), tokenID, toyJSONTokens[tokenID], allowedTokens[tokenID], !allowedTokens[tokenID])
		}
	}
}

func TestRandomTokenWalksOnlyMakeValidJSON(t *testing.T) {
	tokenBytes := toyTokenBytes(toyJSONTokens)
	constraint := NewJSONConstraint(tokenBytes, JSONSettings{MaximumDepth: 4, MaximumWhitespaceInARow: 4})
	randomNumbers := rand.New(rand.NewPCG(3, 3))
	completeDocuments := 0
	for walk := 0; walk < 3000; walk++ {
		constraint.Restart()
		for step := 0; step < 40; step++ {
			if step < 3 {
				checkMaskMatchesSingleChecks(t, constraint)
			}
			if constraint.IsComplete() && randomNumbers.IntN(3) == 0 {
				break
			}
			allowed := allowedTokenIDs(constraint.AllowedTokens())
			if len(allowed) == 0 {
				if !constraint.IsComplete() {
					t.Fatalf("dead end after %q", constraint.Text())
				}
				break
			}
			tokenID := allowed[randomNumbers.IntN(len(allowed))]
			if err := constraint.AcceptToken(tokenID); err != nil {
				t.Fatal(err)
			}
			completion, completed := completeGreedily(constraint.currentState.Clone())
			if !completed || !json.Valid([]byte(constraint.Text()+string(completion))) {
				t.Fatalf("prefix %q cannot be extended into valid JSON", constraint.Text())
			}
		}
		if constraint.IsComplete() {
			completeDocuments++
			if !json.Valid([]byte(constraint.Text())) {
				t.Fatalf("constraint says %q is complete but it is not valid JSON", constraint.Text())
			}
		}
	}
	if completeDocuments < 300 {
		t.Errorf("only %d of the walks ended in a complete document", completeDocuments)
	}
}

func TestSpecialTokensAndUnknownIDsAreNeverAllowed(t *testing.T) {
	constraint := NewJSONConstraint(toyTokenBytes(toyJSONTokens), DefaultJSONSettings())
	specialTokenID := len(toyJSONTokens) - 1
	if constraint.IsTokenAllowed(specialTokenID) || constraint.IsTokenAllowed(-1) || constraint.IsTokenAllowed(len(toyJSONTokens)) {
		t.Error("tokens without bytes and IDs outside the vocabulary must not be allowed")
	}
	if constraint.AcceptToken(1) == nil {
		t.Error("} cannot start a JSON document")
	}
	if constraint.Text() != "" {
		t.Errorf("a refused token changed the text to %q", constraint.Text())
	}
}

func TestChoiceConstraint(t *testing.T) {
	tokens := []string{"y", "es", " please", "no", "n", "o", "yes", "x", ""}
	constraint := NewChoiceConstraint(toyTokenBytes(tokens), []string{"yes", "no", "yes please"})
	allowed := allowedTokenIDs(constraint.AllowedTokens())
	if len(allowed) != 4 || allowed[0] != 0 || allowed[1] != 3 || allowed[2] != 4 || allowed[3] != 6 {
		t.Fatalf("at the start expected tokens y, no, n, yes to be allowed, got %v", allowed)
	}
	if err := constraint.AcceptToken(6); err != nil {
		t.Fatal(err)
	}
	if !constraint.IsComplete() {
		t.Error("yes is one of the choices")
	}
	allowed = allowedTokenIDs(constraint.AllowedTokens())
	if len(allowed) != 1 || allowed[0] != 2 {
		t.Errorf("after yes only \" please\" may follow, got %v", allowed)
	}
	constraint.Restart()
	constraint.AcceptToken(4)
	if constraint.IsComplete() || !constraint.IsTokenAllowed(5) || constraint.IsTokenAllowed(1) {
		t.Error("after n only o should be allowed and the text is not complete")
	}
}

func TestMasksAreRememberedPerState(t *testing.T) {
	constraint := NewJSONConstraint(toyTokenBytes(toyJSONTokens), DefaultJSONSettings())
	constraint.AcceptToken(2)
	firstMask := constraint.AllowedTokens()
	constraint.Restart()
	constraint.AcceptToken(2)
	secondMask := constraint.AllowedTokens()
	if &firstMask[0] != &secondMask[0] {
		t.Error("the same state should reuse the remembered mask")
	}
}

func BenchmarkAllowedTokensWithLargeVocabulary(b *testing.B) {
	randomNumbers := rand.New(rand.NewPCG(5, 5))
	alphabet := []byte("abcdefghijklmnopqrstuvwxyz ABCDEFG0123456789{}[]\":,.-_\n")
	tokenBytes := make([][]byte, 49152)
	for tokenID := range tokenBytes {
		length := 1 + randomNumbers.IntN(8)
		token := make([]byte, length)
		for position := range token {
			token[position] = alphabet[randomNumbers.IntN(len(alphabet))]
		}
		tokenBytes[tokenID] = token
	}
	constraint := NewJSONConstraint(tokenBytes, DefaultJSONSettings())
	constraint.AcceptText(`{"name": "abc`)
	for b.Loop() {
		clear(constraint.allowedTokensByState)
		constraint.AllowedTokens()
	}
}
