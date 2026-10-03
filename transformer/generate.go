package transformer

import (
	"fmt"
	"github.com/javanhut/GoTransformers/activationfunction"
	"github.com/javanhut/GoTransformers/embedding"
	"github.com/javanhut/GoTransformers/hyperconnection"
	"github.com/javanhut/GoTransformers/vectormath"
	"sort"
	"strings"
)

func (model *Model) StartGenerating() {
	for _, block := range model.Blocks {
		block.Attention.StartGenerating()
	}
	model.generatedPositions = 0
	model.lastScores = nil
}

func (model *Model) NextTokenScores(tokenID int) vectormath.Vector {
	model.checkTokenIDs([]int{tokenID})
	value := model.TokenEmbedding.VectorFor(tokenID)
	if !model.Settings.UseRotaryPositions {
		value = vectormath.Add(value, embedding.PositionalEncodingAt(model.generatedPositions, model.Settings.VectorSize))
	}
	streams := model.Settings.residualStreams()
	if streams > 1 {
		value = hyperconnection.ExpandToStreams(oneRow(value), streams).Row(0)
	}
	for _, block := range model.Blocks {
		value = block.ForwardOneToken(value, tokenID)
	}
	if streams > 1 {
		value = hyperconnection.CollapseStreams(oneRow(value), streams).Row(0)
	}
	model.generatedPositions++
	model.lastScores = model.OutputLayer.Forward(model.FinalNorm.Forward(oneRow(value))).Row(0)
	return model.lastScores
}

func (model *Model) Feed(tokenIDs []int) vectormath.Vector {
	for _, tokenID := range tokenIDs {
		model.NextTokenScores(tokenID)
	}
	return model.lastScores
}

func PickToken(scores vectormath.Vector, temperature float64) int {
	if temperature <= 0 {
		return vectormath.IndexOfMax(scores)
	}
	probabilities := activationfunction.Softmax(vectormath.Scale(scores, 1/temperature))
	randomPoint := vectormath.RandomNumberBetween(0, 1)
	total := 0.0
	for tokenID, probability := range probabilities {
		total += probability
		if randomPoint < total {
			return tokenID
		}
	}
	return len(probabilities) - 1
}

func PickTokenFromTop(scores vectormath.Vector, temperature float64, topProbability float64) int {
	if temperature <= 0 || topProbability <= 0 {
		return vectormath.IndexOfMax(scores)
	}
	if topProbability >= 1 {
		return PickToken(scores, temperature)
	}
	probabilities := activationfunction.Softmax(vectormath.Scale(scores, 1/temperature))
	tokenIDs := make([]int, len(probabilities))
	for tokenID := range tokenIDs {
		tokenIDs[tokenID] = tokenID
	}
	sort.Slice(tokenIDs, func(i int, j int) bool {
		return probabilities[tokenIDs[i]] > probabilities[tokenIDs[j]]
	})
	keptProbability := 0.0
	keptCount := 0
	for keptCount < len(tokenIDs) && keptProbability < topProbability {
		keptProbability += probabilities[tokenIDs[keptCount]]
		keptCount++
	}
	randomPoint := vectormath.RandomNumberBetween(0, keptProbability)
	total := 0.0
	for _, tokenID := range tokenIDs[:keptCount] {
		total += probabilities[tokenID]
		if randomPoint < total {
			return tokenID
		}
	}
	return tokenIDs[keptCount-1]
}

func (model *Model) ContinueGenerating(numberOfNewTokens int, temperature float64) []int {
	if model.lastScores == nil {
		panic("Model.ContinueGenerating: feed at least 1 token first")
	}
	var generatedIDs []int
	for len(generatedIDs) < numberOfNewTokens {
		nextID := PickToken(model.lastScores, temperature)
		generatedIDs = append(generatedIDs, nextID)
		model.NextTokenScores(nextID)
	}
	return generatedIDs
}

func (model *Model) Generate(promptIDs []int, numberOfNewTokens int, temperature float64) []int {
	if len(promptIDs) == 0 {
		panic("Model.Generate: the prompt needs at least 1 token")
	}
	model.StartGenerating()
	model.Feed(promptIDs)
	return model.ContinueGenerating(numberOfNewTokens, temperature)
}

func (model *Model) ContinueGeneratingWithSampler(numberOfNewTokens int, sampler *Sampler) []int {
	if model.lastScores == nil {
		panic("Model.ContinueGeneratingWithSampler: feed at least 1 token first")
	}
	var generatedIDs []int
	for len(generatedIDs) < numberOfNewTokens {
		nextID, found := sampler.PickToken(model.lastScores)
		if !found || isStopToken(nextID, sampler.StopTokenIDs) {
			break
		}
		if err := sampler.AcceptToken(nextID); err != nil {
			panic(fmt.Sprintf("Model.ContinueGeneratingWithSampler: the sampler picked a token its constraint refuses: %v", err))
		}
		generatedIDs = append(generatedIDs, nextID)
		model.NextTokenScores(nextID)
	}
	return generatedIDs
}

func (model *Model) GenerateWithSampler(promptIDs []int, numberOfNewTokens int, sampler *Sampler) []int {
	if len(promptIDs) == 0 {
		panic("Model.GenerateWithSampler: the prompt needs at least 1 token")
	}
	model.StartGenerating()
	model.Feed(promptIDs)
	sampler.RememberPrompt(promptIDs)
	if sampler.Constraint != nil {
		sampler.Constraint.Restart()
	}
	return model.ContinueGeneratingWithSampler(numberOfNewTokens, sampler)
}

func (model *Model) CacheBytesUsed() int {
	total := 0
	for _, block := range model.Blocks {
		total += block.Attention.CacheBytesUsed()
	}
	return total
}

func (model *Model) Describe() string {
	settings := model.Settings
	var parts []string
	parts = append(parts, fmt.Sprintf("%d blocks", settings.NumberOfBlocks))
	parts = append(parts, fmt.Sprintf("vector size %d", settings.VectorSize))
	parts = append(parts, fmt.Sprintf("%d heads", settings.NumberOfHeads))
	if settings.NumberOfKeyValueHeads > 0 && settings.NumberOfKeyValueHeads != settings.NumberOfHeads {
		parts = append(parts, fmt.Sprintf("%d key/value heads", settings.NumberOfKeyValueHeads))
	}
	parts = append(parts, fmt.Sprintf("%d parameters", model.NumberOfParameters()))
	var kinds []string
	for blockIndex := 0; blockIndex < settings.NumberOfBlocks; blockIndex++ {
		kinds = append(kinds, string(settings.attentionKindFor(blockIndex)))
	}
	parts = append(parts, "attention ["+strings.Join(kinds, " ")+"]")
	if settings.UseRotaryPositions {
		parts = append(parts, "rotary positions")
	}
	if settings.ShareKeyAsValue {
		parts = append(parts, "key doubles as value")
	}
	if settings.NormalizeQueriesAndKeys {
		parts = append(parts, "query/key norm")
	}
	if settings.UseAttentionSink {
		parts = append(parts, "attention sink")
	}
	if settings.UseMixtureOfExperts {
		parts = append(parts, fmt.Sprintf("%d shared + %d routed experts (%d per token)", settings.NumberOfSharedExperts, settings.NumberOfRoutedExperts, settings.ExpertsPerToken))
	}
	if settings.residualStreams() > 1 {
		parts = append(parts, fmt.Sprintf("mHC with %d streams", settings.residualStreams()))
	}
	if settings.MultiTokenPrediction {
		parts = append(parts, "multi-token prediction")
	}
	parts = append(parts, fmt.Sprintf("cache %v", settings.CachePrecision))
	return strings.Join(parts, ", ")
}
