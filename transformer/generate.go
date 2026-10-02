package transformer

import (
	"fmt"
	"transformer/activationfunction"
	"transformer/embedding"
	"transformer/vectormath"
)

func (model *Model) StartGenerating() {
	for _, block := range model.Blocks {
		block.Attention.StartGenerating()
	}
	model.generatedPositions = 0
}

func (model *Model) NextTokenScores(tokenID int) vectormath.Vector {
	model.checkTokenIDs([]int{tokenID})
	value := vectormath.CopyVector(model.TokenEmbedding.Table.Row(tokenID))
	if !model.Settings.UseRotaryPositions {
		value = vectormath.Add(value, embedding.PositionalEncodingAt(model.generatedPositions, model.Settings.VectorSize))
	}
	for _, block := range model.Blocks {
		value = block.ForwardOneToken(value)
	}
	model.generatedPositions++
	return model.OutputLayer.Forward(model.FinalNorm.Forward(oneRow(value))).Row(0)
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

func (model *Model) Generate(promptIDs []int, numberOfNewTokens int, temperature float64) []int {
	if len(promptIDs) == 0 {
		panic("Model.Generate: the prompt needs at least 1 token")
	}
	model.StartGenerating()
	var scores vectormath.Vector
	for _, tokenID := range promptIDs {
		scores = model.NextTokenScores(tokenID)
	}

	var generatedIDs []int
	for len(generatedIDs) < numberOfNewTokens {
		nextID := PickToken(scores, temperature)
		generatedIDs = append(generatedIDs, nextID)
		if len(generatedIDs) < numberOfNewTokens {
			scores = model.NextTokenScores(nextID)
		}
	}
	return generatedIDs
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
	positions := "sine-wave positions"
	if settings.UseRotaryPositions {
		positions = "rotary positions"
	}
	return fmt.Sprintf("%d blocks, vector size %d, %d heads, %d parameters, %s, window %d (full every %d), top-k %d, %d blocks per key/value group (%v), cache %v",
		settings.NumberOfBlocks, settings.VectorSize, settings.NumberOfHeads, model.NumberOfParameters(), positions,
		settings.WindowSize, settings.FullAttentionEvery, settings.TopK,
		settings.BlocksPerKeyValueGroup, settings.GroupSharingMode, settings.CachePrecision)
}
