package transformer

import (
	"fmt"
	"transformer/lowprecision"
	"transformer/perceptron"
	"transformer/vectormath"
)

func (model *Model) allLayers() []*perceptron.Layer {
	var layers []*perceptron.Layer
	for _, block := range model.Blocks {
		layers = append(layers, block.Layers()...)
	}
	if model.MultiTokenPredictor != nil {
		layers = append(layers, model.MultiTokenPredictor.CombineLayer)
		layers = append(layers, model.MultiTokenPredictor.Block.Layers()...)
	}
	return append(layers, model.OutputLayer)
}

func (model *Model) CompressWeights(precision lowprecision.Precision) {
	for _, layer := range model.allLayers() {
		if !layer.IsCompressed() {
			layer.CompressWeights(precision)
		}
	}
	if !model.TokenEmbedding.IsCompressed() {
		model.TokenEmbedding.CompressTable(precision)
	}
}

func (model *Model) IsCompressed() bool {
	return model.OutputLayer.IsCompressed()
}

func (model *Model) DecompressWeights() {
	for _, layer := range model.allLayers() {
		layer.DecompressWeights()
	}
	if model.TokenEmbedding.IsCompressed() {
		table := model.TokenEmbedding.TableRows(allTokenIDs(model.Settings.VocabularySize))
		model.TokenEmbedding.Table = table
		model.TokenEmbedding.CompressedTable = nil
	}
}

func allTokenIDs(vocabularySize int) []int {
	tokenIDs := make([]int, vocabularySize)
	for tokenID := range tokenIDs {
		tokenIDs[tokenID] = tokenID
	}
	return tokenIDs
}

func (model *Model) ReleaseGradients() {
	for _, layer := range model.allLayers() {
		layer.ReleaseGradients()
	}
	model.TokenEmbedding.ReleaseGradients()
}

func (model *Model) WeightBytes() int {
	total := model.TokenEmbedding.TableBytes()
	for _, layer := range model.allLayers() {
		total += layer.WeightBytes()
	}
	return total
}

func (model *Model) GradientBytes() int {
	total := len(model.TokenEmbedding.TableGradients) * 8
	for _, layer := range model.allLayers() {
		total += (len(layer.WeightGradients) + len(layer.BiasGradients)) * 8
	}
	return total
}

func (model *Model) weightSetters() map[string]func(values []float64) error {
	setters := map[string]func(values []float64) error{}
	for _, layer := range model.allLayers() {
		setters[layer.Name+".weights"] = layer.SetWeights
		setters[layer.Name+".biases"] = layer.SetBiases
	}
	setters[model.TokenEmbedding.Name+".table"] = model.TokenEmbedding.SetTable
	for _, current := range model.Parameters() {
		if _, alreadySet := setters[current.Name]; alreadySet {
			continue
		}
		target := current
		setters[current.Name] = func(values []float64) error {
			if len(values) != len(target.Values) {
				return fmt.Errorf("parameter %q holds %d values but got %d", target.Name, len(target.Values), len(values))
			}
			copy(target.Values, values)
			return nil
		}
	}
	return setters
}

func (model *Model) SetWeights(savedValues map[string][]float64) error {
	setters := model.weightSetters()
	for name := range setters {
		if _, found := savedValues[name]; !found {
			return fmt.Errorf("there is no saved parameter named %q", name)
		}
	}
	for name, values := range savedValues {
		setter, found := setters[name]
		if !found {
			return fmt.Errorf("the model has no parameter named %q", name)
		}
		if err := setter(values); err != nil {
			return err
		}
	}
	vectormath.MarkWeightsChanged()
	return nil
}

func (model *Model) SetWeight(name string, values []float64) error {
	setter, found := model.weightSetters()[name]
	if !found {
		return fmt.Errorf("the model has no parameter named %q", name)
	}
	if err := setter(values); err != nil {
		return err
	}
	vectormath.MarkWeightsChanged()
	return nil
}
