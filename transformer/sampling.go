package transformer

import (
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
	"math/rand/v2"
	"slices"
)

type TokenConstraint interface {
	Restart()
	IsTokenAllowed(tokenID int) bool
	AllowedTokens() []bool
	AcceptToken(tokenID int) error
	IsComplete() bool
}

type SamplingOptions struct {
	Temperature        float64
	TopK               int
	TopProbability     float64
	MinimumProbability float64
	TypicalProbability float64
	RepetitionPenalty  float64
	FrequencyPenalty   float64
	PresencePenalty    float64
	PenaltyWindow      int
	PenalizePrompt     bool
	UseRandomSeed      bool
	RandomSeed         uint64
}

type Sampler struct {
	Options      SamplingOptions
	Constraint   TokenConstraint
	StopTokenIDs []int

	seenTokenIDs  []int
	randomNumbers *rand.Rand
}

func NewSampler(options SamplingOptions, constraint TokenConstraint) *Sampler {
	sampler := &Sampler{Options: options, Constraint: constraint}
	if options.UseRandomSeed {
		sampler.randomNumbers = rand.New(rand.NewPCG(options.RandomSeed, options.RandomSeed))
	}
	return sampler
}

func (sampler *Sampler) RememberPrompt(promptIDs []int) {
	if sampler.Options.PenalizePrompt {
		sampler.seenTokenIDs = append(sampler.seenTokenIDs, promptIDs...)
	}
}

func (sampler *Sampler) AcceptToken(tokenID int) error {
	if isStopToken(tokenID, sampler.StopTokenIDs) {
		return nil
	}
	sampler.seenTokenIDs = append(sampler.seenTokenIDs, tokenID)
	if sampler.Constraint == nil {
		return nil
	}
	return sampler.Constraint.AcceptToken(tokenID)
}

func (sampler *Sampler) PickToken(scores vectormath.Vector) (int, bool) {
	penalizedScores := sampler.penalizedScores(scores)
	if sampler.Options.Temperature <= 0 {
		return sampler.mostLikelyAllowedToken(penalizedScores)
	}
	allowedTokens, anyAllowed := sampler.allowedTokens(len(scores))
	if !anyAllowed {
		return -1, false
	}
	probabilities := probabilitiesOverAllowedTokens(penalizedScores, allowedTokens, sampler.Options.Temperature)
	probabilities = keepTopK(probabilities, sampler.Options.TopK)
	probabilities = keepTypical(probabilities, sampler.Options.TypicalProbability)
	probabilities = keepTopProbability(probabilities, sampler.Options.TopProbability)
	probabilities = keepAboveMinimumProbability(probabilities, sampler.Options.MinimumProbability)
	return sampler.drawToken(probabilities), true
}

func (sampler *Sampler) recentTokenIDs() []int {
	window := sampler.Options.PenaltyWindow
	if window <= 0 || window >= len(sampler.seenTokenIDs) {
		return sampler.seenTokenIDs
	}
	return sampler.seenTokenIDs[len(sampler.seenTokenIDs)-window:]
}

func (sampler *Sampler) penalizedScores(scores vectormath.Vector) vectormath.Vector {
	recentTokenIDs := sampler.recentTokenIDs()
	penalized := applyRepetitionPenalty(scores, recentTokenIDs, sampler.Options.RepetitionPenalty)
	return applyFrequencyAndPresencePenalties(penalized, recentTokenIDs, sampler.Options.FrequencyPenalty, sampler.Options.PresencePenalty)
}

func applyRepetitionPenalty(scores vectormath.Vector, recentTokenIDs []int, repetitionPenalty float64) vectormath.Vector {
	if repetitionPenalty <= 0 || repetitionPenalty == 1 || len(recentTokenIDs) == 0 {
		return scores
	}
	penalized := slices.Clone(scores)
	alreadyPenalized := map[int]bool{}
	for _, tokenID := range recentTokenIDs {
		if tokenID < 0 || tokenID >= len(penalized) || alreadyPenalized[tokenID] {
			continue
		}
		alreadyPenalized[tokenID] = true
		if penalized[tokenID] > 0 {
			penalized[tokenID] /= repetitionPenalty
		} else {
			penalized[tokenID] *= repetitionPenalty
		}
	}
	return penalized
}

func applyFrequencyAndPresencePenalties(scores vectormath.Vector, recentTokenIDs []int, frequencyPenalty float64, presencePenalty float64) vectormath.Vector {
	if (frequencyPenalty == 0 && presencePenalty == 0) || len(recentTokenIDs) == 0 {
		return scores
	}
	timesSeen := map[int]int{}
	for _, tokenID := range recentTokenIDs {
		if tokenID >= 0 && tokenID < len(scores) {
			timesSeen[tokenID]++
		}
	}
	penalized := slices.Clone(scores)
	for tokenID, count := range timesSeen {
		penalized[tokenID] -= float64(count)*frequencyPenalty + presencePenalty
	}
	return penalized
}

func isAllowedByMask(allowedTokens []bool, tokenID int) bool {
	if allowedTokens == nil {
		return true
	}
	return tokenID < len(allowedTokens) && allowedTokens[tokenID]
}

func (sampler *Sampler) isTokenAllowed(tokenID int) bool {
	if sampler.Constraint == nil {
		return true
	}
	if isStopToken(tokenID, sampler.StopTokenIDs) {
		return sampler.Constraint.IsComplete()
	}
	return sampler.Constraint.IsTokenAllowed(tokenID)
}

func (sampler *Sampler) allowedTokens(vocabularySize int) ([]bool, bool) {
	if sampler.Constraint == nil {
		return nil, true
	}
	allowedByConstraint := sampler.Constraint.AllowedTokens()
	allowedTokens := make([]bool, vocabularySize)
	anyAllowed := false
	for tokenID := 0; tokenID < vocabularySize; tokenID++ {
		allowedTokens[tokenID] = isAllowedByMask(allowedByConstraint, tokenID)
		if isStopToken(tokenID, sampler.StopTokenIDs) {
			allowedTokens[tokenID] = sampler.Constraint.IsComplete()
		}
		if allowedTokens[tokenID] {
			anyAllowed = true
		}
	}
	return allowedTokens, anyAllowed
}

func (sampler *Sampler) mostLikelyAllowedToken(scores vectormath.Vector) (int, bool) {
	bestTokenID := vectormath.IndexOfMax(scores)
	if sampler.isTokenAllowed(bestTokenID) {
		return bestTokenID, true
	}
	allowedTokens, anyAllowed := sampler.allowedTokens(len(scores))
	if !anyAllowed {
		return -1, false
	}
	bestTokenID = -1
	for tokenID := 0; tokenID < len(scores); tokenID++ {
		if !allowedTokens[tokenID] {
			continue
		}
		if bestTokenID == -1 || scores[tokenID] > scores[bestTokenID] {
			bestTokenID = tokenID
		}
	}
	return bestTokenID, true
}

func probabilitiesOverAllowedTokens(scores vectormath.Vector, allowedTokens []bool, temperature float64) []float64 {
	largestScore := math.Inf(-1)
	for tokenID := 0; tokenID < len(scores); tokenID++ {
		if isAllowedByMask(allowedTokens, tokenID) && scores[tokenID] > largestScore {
			largestScore = scores[tokenID]
		}
	}
	probabilities := make([]float64, len(scores))
	for tokenID := 0; tokenID < len(scores); tokenID++ {
		if !isAllowedByMask(allowedTokens, tokenID) {
			continue
		}
		if math.IsInf(largestScore, -1) {
			probabilities[tokenID] = 1
		} else if !math.IsNaN(scores[tokenID]) {
			probabilities[tokenID] = math.Exp((scores[tokenID] - largestScore) / temperature)
		}
	}
	return normalized(probabilities)
}

func normalized(probabilities []float64) []float64 {
	total := 0.0
	for _, probability := range probabilities {
		total += probability
	}
	if total <= 0 {
		return probabilities
	}
	for tokenID := range probabilities {
		probabilities[tokenID] /= total
	}
	return probabilities
}

func tokenIDsInOrder(probabilities []float64, comesFirst func(firstTokenID int, secondTokenID int) bool) []int {
	var tokenIDs []int
	for tokenID, probability := range probabilities {
		if probability > 0 {
			tokenIDs = append(tokenIDs, tokenID)
		}
	}
	slices.SortStableFunc(tokenIDs, func(firstTokenID int, secondTokenID int) int {
		if comesFirst(firstTokenID, secondTokenID) {
			return -1
		}
		if comesFirst(secondTokenID, firstTokenID) {
			return 1
		}
		return 0
	})
	return tokenIDs
}

func tokenIDsByProbability(probabilities []float64) []int {
	return tokenIDsInOrder(probabilities, func(firstTokenID int, secondTokenID int) bool {
		return probabilities[firstTokenID] > probabilities[secondTokenID]
	})
}

func keepOnly(probabilities []float64, keptTokenIDs []int) []float64 {
	kept := make([]float64, len(probabilities))
	for _, tokenID := range keptTokenIDs {
		kept[tokenID] = probabilities[tokenID]
	}
	return normalized(kept)
}

func keepUntilMassReached(probabilities []float64, orderedTokenIDs []int, targetMass float64) []float64 {
	keptMass := 0.0
	keptCount := 0
	for keptCount < len(orderedTokenIDs) && keptMass < targetMass {
		keptMass += probabilities[orderedTokenIDs[keptCount]]
		keptCount++
	}
	return keepOnly(probabilities, orderedTokenIDs[:keptCount])
}

func keepTopK(probabilities []float64, topK int) []float64 {
	if topK <= 0 {
		return probabilities
	}
	orderedTokenIDs := tokenIDsByProbability(probabilities)
	if topK >= len(orderedTokenIDs) {
		return probabilities
	}
	return keepOnly(probabilities, orderedTokenIDs[:topK])
}

func keepTopProbability(probabilities []float64, topProbability float64) []float64 {
	if topProbability <= 0 || topProbability >= 1 {
		return probabilities
	}
	return keepUntilMassReached(probabilities, tokenIDsByProbability(probabilities), topProbability)
}

func entropyOf(probabilities []float64) float64 {
	entropy := 0.0
	for _, probability := range probabilities {
		if probability > 0 {
			entropy -= probability * math.Log(probability)
		}
	}
	return entropy
}

func keepTypical(probabilities []float64, typicalProbability float64) []float64 {
	if typicalProbability <= 0 || typicalProbability >= 1 {
		return probabilities
	}
	entropy := entropyOf(probabilities)
	distanceFromTypical := func(tokenID int) float64 {
		return math.Abs(-math.Log(probabilities[tokenID]) - entropy)
	}
	orderedTokenIDs := tokenIDsInOrder(probabilities, func(firstTokenID int, secondTokenID int) bool {
		return distanceFromTypical(firstTokenID) < distanceFromTypical(secondTokenID)
	})
	return keepUntilMassReached(probabilities, orderedTokenIDs, typicalProbability)
}

func keepAboveMinimumProbability(probabilities []float64, minimumProbability float64) []float64 {
	if minimumProbability <= 0 {
		return probabilities
	}
	largestProbability := 0.0
	for _, probability := range probabilities {
		largestProbability = math.Max(largestProbability, probability)
	}
	threshold := minimumProbability * largestProbability
	var keptTokenIDs []int
	for tokenID, probability := range probabilities {
		if probability > 0 && probability >= threshold {
			keptTokenIDs = append(keptTokenIDs, tokenID)
		}
	}
	return keepOnly(probabilities, keptTokenIDs)
}

func (sampler *Sampler) randomNumberBetween(lowest float64, highest float64) float64 {
	if sampler.randomNumbers == nil {
		return vectormath.RandomNumberBetween(lowest, highest)
	}
	return lowest + sampler.randomNumbers.Float64()*(highest-lowest)
}

func (sampler *Sampler) drawToken(probabilities []float64) int {
	total := 0.0
	for _, probability := range probabilities {
		total += probability
	}
	randomPoint := sampler.randomNumberBetween(0, total)
	runningTotal := 0.0
	lastPossibleTokenID := -1
	for tokenID, probability := range probabilities {
		if probability <= 0 {
			continue
		}
		runningTotal += probability
		lastPossibleTokenID = tokenID
		if randomPoint < runningTotal {
			return tokenID
		}
	}
	return lastPossibleTokenID
}
