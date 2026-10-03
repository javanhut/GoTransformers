package transformer

import (
	"encoding/json"
	"github.com/javanhut/GoTransformers/constrained"
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
	"slices"
	"testing"
)

func closeTo(first float64, second float64) bool {
	return math.Abs(first-second) < 1e-9
}

func keptTokenIDs(probabilities []float64) []int {
	var tokenIDs []int
	for tokenID, probability := range probabilities {
		if probability > 0 {
			tokenIDs = append(tokenIDs, tokenID)
		}
	}
	return tokenIDs
}

func checkSumsToOne(t *testing.T, probabilities []float64) {
	total := 0.0
	for _, probability := range probabilities {
		total += probability
	}
	if !closeTo(total, 1) {
		t.Errorf("probabilities %v add up to %v, not 1", probabilities, total)
	}
}

func TestRepetitionPenalty(t *testing.T) {
	scores := vectormath.Vector{2, -1, 0.5, 3}
	penalized := applyRepetitionPenalty(scores, []int{0, 1, 0}, 2)
	want := vectormath.Vector{1, -2, 0.5, 3}
	if !slices.Equal(penalized, want) {
		t.Errorf("got %v, want %v", penalized, want)
	}
	if scores[0] != 2 {
		t.Error("the original scores were changed")
	}
}

func TestFrequencyAndPresencePenalties(t *testing.T) {
	scores := vectormath.Vector{2, -1, 0.5, 3}
	penalized := applyFrequencyAndPresencePenalties(scores, []int{0, 0, 2}, 0.5, 1)
	want := vectormath.Vector{0, -1, -1, 3}
	if !slices.Equal(penalized, want) {
		t.Errorf("got %v, want %v", penalized, want)
	}
}

func TestPenaltyWindowAndPrompt(t *testing.T) {
	sampler := NewSampler(SamplingOptions{PenaltyWindow: 2}, nil)
	sampler.RememberPrompt([]int{5, 6})
	sampler.AcceptToken(0)
	sampler.AcceptToken(1)
	sampler.AcceptToken(2)
	if !slices.Equal(sampler.recentTokenIDs(), []int{1, 2}) {
		t.Errorf("window of 2 gave %v", sampler.recentTokenIDs())
	}
	withPrompt := NewSampler(SamplingOptions{PenalizePrompt: true}, nil)
	withPrompt.RememberPrompt([]int{5, 6})
	withPrompt.AcceptToken(0)
	if !slices.Equal(withPrompt.recentTokenIDs(), []int{5, 6, 0}) {
		t.Errorf("window 0 with the prompt gave %v", withPrompt.recentTokenIDs())
	}
}

func TestTopKKeepsTheMostLikely(t *testing.T) {
	probabilities := keepTopK([]float64{0.1, 0.5, 0.3, 0.06, 0.04}, 2)
	if !slices.Equal(keptTokenIDs(probabilities), []int{1, 2}) {
		t.Errorf("kept %v", probabilities)
	}
	checkSumsToOne(t, probabilities)
	if !closeTo(probabilities[1], 0.5/0.8) {
		t.Errorf("token 1 should have 0.5/0.8, got %v", probabilities[1])
	}
}

func TestTopProbability(t *testing.T) {
	probabilities := keepTopProbability([]float64{0.5, 0.3, 0.1, 0.06, 0.04}, 0.75)
	if !slices.Equal(keptTokenIDs(probabilities), []int{0, 1}) {
		t.Errorf("kept %v", probabilities)
	}
	checkSumsToOne(t, probabilities)
}

func TestMinimumProbabilityRemovesExactlyTheUnlikelyTokens(t *testing.T) {
	probabilities := keepAboveMinimumProbability([]float64{0.5, 0.3, 0.1, 0.06, 0.04}, 0.15)
	if !slices.Equal(keptTokenIDs(probabilities), []int{0, 1, 2}) {
		t.Errorf("min-p 0.15 should keep tokens at or above 0.075, kept %v", probabilities)
	}
	checkSumsToOne(t, probabilities)
	if !closeTo(probabilities[2], 0.1/0.9) {
		t.Errorf("token 2 should have 0.1/0.9, got %v", probabilities[2])
	}
}

func TestTypicalSamplingKeepsTokensNearTheEntropy(t *testing.T) {
	probabilities := []float64{0.4, 0.3, 0.2, 0.1}
	entropy := entropyOf(probabilities)
	if !closeTo(entropy, -(0.4*math.Log(0.4) + 0.3*math.Log(0.3) + 0.2*math.Log(0.2) + 0.1*math.Log(0.1))) {
		t.Errorf("entropy %v", entropy)
	}
	kept := keepTypical(probabilities, 0.45)
	if !slices.Equal(keptTokenIDs(kept), []int{1, 2}) {
		t.Errorf("typical 0.45 should keep tokens 1 and 2 (closest to the entropy), kept %v", kept)
	}
	checkSumsToOne(t, kept)
	if !slices.Equal(keptTokenIDs(keepTypical(probabilities, 0.6)), []int{0, 1, 2}) {
		t.Errorf("typical 0.6 should add token 0 next")
	}
}

func TestOffSettingsChangeNothing(t *testing.T) {
	probabilities := []float64{0.5, 0.3, 0.2}
	if !slices.Equal(keepTopK(probabilities, 0), probabilities) || !slices.Equal(keepTopProbability(probabilities, 1), probabilities) ||
		!slices.Equal(keepTypical(probabilities, 0), probabilities) || !slices.Equal(keepAboveMinimumProbability(probabilities, 0), probabilities) {
		t.Error("turned-off sampling steps should keep every token")
	}
}

func TestSeededSamplingIsReproducible(t *testing.T) {
	scores := vectormath.Vector{1, 0.5, 0.2, 0, -0.3, 0.8, 0.1}
	options := SamplingOptions{Temperature: 1, UseRandomSeed: true, RandomSeed: 42}
	first := NewSampler(options, nil)
	second := NewSampler(options, nil)
	options.RandomSeed = 43
	other := NewSampler(options, nil)
	differences := 0
	for draw := 0; draw < 100; draw++ {
		firstID, _ := first.PickToken(scores)
		secondID, _ := second.PickToken(scores)
		otherID, _ := other.PickToken(scores)
		if firstID != secondID {
			t.Fatalf("draw %d: the same seed gave %d and %d", draw, firstID, secondID)
		}
		if firstID != otherID {
			differences++
		}
	}
	if differences == 0 {
		t.Error("a different seed gave exactly the same 100 draws")
	}
}

func TestSamplingFollowsTheProbabilities(t *testing.T) {
	scores := vectormath.Vector{math.Log(0.5), math.Log(0.3), math.Log(0.2)}
	sampler := NewSampler(SamplingOptions{Temperature: 1, UseRandomSeed: true, RandomSeed: 1}, nil)
	counts := make([]int, 3)
	draws := 20000
	for draw := 0; draw < draws; draw++ {
		tokenID, _ := sampler.PickToken(scores)
		counts[tokenID]++
	}
	want := []float64{0.5, 0.3, 0.2}
	for tokenID, count := range counts {
		if math.Abs(float64(count)/float64(draws)-want[tokenID]) > 0.02 {
			t.Errorf("token %d drawn %d times out of %d, want about %v", tokenID, count, draws, want[tokenID])
		}
	}
}

func TestGreedySamplingHonoursPenalties(t *testing.T) {
	sampler := NewSampler(SamplingOptions{RepetitionPenalty: 3}, nil)
	sampler.AcceptToken(0)
	tokenID, found := sampler.PickToken(vectormath.Vector{3, 2, -1})
	if !found || tokenID != 1 {
		t.Errorf("token 0 (score 3, penalized to 1) should lose to token 1 (score 2), got %d", tokenID)
	}
}

func choiceTokens() [][]byte {
	return [][]byte{[]byte("yes"), []byte("no"), []byte("maybe"), nil}
}

func TestConstraintWinsEvenWhenAllowedTokensAreUnlikely(t *testing.T) {
	scores := vectormath.Vector{-1000, -1200, 1000, 900}
	for _, temperature := range []float64{0, 0.5, 1, 5} {
		constraint := constrained.NewChoiceConstraint(choiceTokens(), []string{"yes", "no"})
		sampler := NewSampler(SamplingOptions{Temperature: temperature, TopK: 1, MinimumProbability: 0.5, TopProbability: 0.1, UseRandomSeed: true}, constraint)
		sampler.StopTokenIDs = []int{3}
		tokenID, found := sampler.PickToken(scores)
		if !found || tokenID != 0 {
			t.Errorf("temperature %v: expected the allowed token 0, got %d (found %t)", temperature, tokenID, found)
		}
		sampler.AcceptToken(tokenID)
		tokenID, found = sampler.PickToken(scores)
		if !found || tokenID != 3 {
			t.Errorf("temperature %v: once complete only the stop token is allowed, got %d", temperature, tokenID)
		}
	}
}

func TestStopTokenIsRefusedUntilComplete(t *testing.T) {
	constraint := constrained.NewChoiceConstraint(choiceTokens(), []string{"maybe"})
	sampler := NewSampler(SamplingOptions{}, constraint)
	sampler.StopTokenIDs = []int{3}
	tokenID, _ := sampler.PickToken(vectormath.Vector{0, 0, -5, 10})
	if tokenID != 2 {
		t.Errorf("the stop token should be refused before the choice is written, got %d", tokenID)
	}
}

var toyJSONVocabulary = []string{
	"{", "}", "[", "]", ":", ",", "\"", "\"a", "\"b\"", "x", "é", "1", "0", "-", ".5", "e2", "2.5e3",
	"true", "false", "null", " ", "\n", "\"},", "\"]", "],", "},", "\":", "\":\"", "\"}", "{\"", "[\"", ",\"", "\\n",
}

func toyJSONTokenBytes() [][]byte {
	tokenBytes := make([][]byte, len(toyJSONVocabulary)+1)
	for tokenID, token := range toyJSONVocabulary {
		tokenBytes[tokenID] = []byte(token)
	}
	return tokenBytes
}

func tinyModel(t *testing.T, vocabularySize int) *Model {
	settings := SmallSettings(vocabularySize)
	settings.VectorSize = 16
	settings.NumberOfHeads = 2
	settings.FeedForwardSize = 24
	settings.NumberOfBlocks = 2
	model, err := NewModel(settings)
	if err != nil {
		t.Fatal(err)
	}
	return model
}

func TestUntrainedModelWritesValidJSONWhenConstrained(t *testing.T) {
	tokenBytes := toyJSONTokenBytes()
	stopTokenID := len(tokenBytes) - 1
	model := tinyModel(t, len(tokenBytes))
	constraint := constrained.NewJSONConstraint(tokenBytes, constrained.JSONSettings{MaximumDepth: 3, MaximumWhitespaceInARow: 2})
	for seed := uint64(0); seed < 20; seed++ {
		answer := model.Answer([]int{0, 7}, GenerationOptions{
			MaximumNewTokens: 300,
			StopTokenIDs:     []int{stopTokenID},
			Constraint:       constraint,
			Sampling:         SamplingOptions{Temperature: 1.5, RepetitionPenalty: 1.1, UseRandomSeed: true, RandomSeed: seed},
		})
		var text []byte
		for _, tokenID := range answer.TokenIDs {
			text = append(text, tokenBytes[tokenID]...)
		}
		if !answer.ConstraintSatisfied {
			t.Errorf("seed %d: ran out of tokens before the JSON was complete: %q", seed, text)
			continue
		}
		if !json.Valid(text) {
			t.Errorf("seed %d: %q is not valid JSON", seed, text)
		}
		if seed < 3 {
			t.Logf("seed %d: %s", seed, text)
		}
	}
}

func TestGenerateWithSamplerStopsWhenTheChoiceIsWritten(t *testing.T) {
	tokenBytes := [][]byte{[]byte("ye"), []byte("s"), []byte("no"), []byte("x")}
	model := tinyModel(t, len(tokenBytes))
	constraint := constrained.NewChoiceConstraint(tokenBytes, []string{"yes", "no"})
	sampler := NewSampler(SamplingOptions{Temperature: 1, UseRandomSeed: true, RandomSeed: 9}, constraint)
	generatedIDs := model.GenerateWithSampler([]int{3}, 10, sampler)
	if !constraint.IsComplete() || (constraint.Text() != "yes" && constraint.Text() != "no") {
		t.Errorf("generated %v (%q)", generatedIDs, constraint.Text())
	}
}

func TestAnswerWithoutSamplingOptionsStillUsesTemperature(t *testing.T) {
	model := tinyModel(t, 10)
	greedy := model.Answer([]int{1, 2}, GenerationOptions{MaximumNewTokens: 6})
	again := model.Answer([]int{1, 2}, GenerationOptions{MaximumNewTokens: 6, Sampling: SamplingOptions{TopK: 3}})
	if !slices.Equal(greedy.TokenIDs, again.TokenIDs) {
		t.Errorf("temperature 0 must stay greedy: %v vs %v", greedy.TokenIDs, again.TokenIDs)
	}
	if !greedy.ConstraintSatisfied {
		t.Error("without a constraint the answer always satisfies it")
	}
}
