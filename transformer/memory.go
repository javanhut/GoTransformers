package transformer

import (
	"fmt"
	"github.com/javanhut/GoTransformers/attention"
	"github.com/javanhut/GoTransformers/dropout"
	"github.com/javanhut/GoTransformers/lowprecision"
	"github.com/javanhut/GoTransformers/parameter"
	"github.com/javanhut/GoTransformers/perceptron"
	"github.com/javanhut/GoTransformers/vectormath"
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
	if model.Settings.TieOutputToEmbedding {
		model.tieOutputToEmbedding()
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
	if model.outputSharesTheEmbeddingTable() {
		total -= model.TokenEmbedding.TableBytes()
	}
	return total
}

func (model *Model) outputSharesTheEmbeddingTable() bool {
	if !model.Settings.TieOutputToEmbedding {
		return false
	}
	outputLayer := model.OutputLayer
	tokenEmbedding := model.TokenEmbedding
	if outputLayer.IsCompressed() || tokenEmbedding.IsCompressed() {
		return outputLayer.CompressedWeights == tokenEmbedding.CompressedTable
	}
	return len(outputLayer.Weights.Values) > 0 && &outputLayer.Weights.Values[0] == &tokenEmbedding.Table.Values[0]
}

func (model *Model) GradientBytes() int {
	total := len(model.TokenEmbedding.TableGradients) * 8
	for _, layer := range model.allLayers() {
		total += (len(layer.WeightGradients) + len(layer.BiasGradients)) * 8
		if layer.Adapter != nil {
			total += (len(layer.Adapter.DownGradients) + len(layer.Adapter.UpGradients)) * 8
		}
	}
	return total
}

func (model *Model) weightSetters() map[string]func(values []float64) error {
	setters := map[string]func(values []float64) error{}
	for _, layer := range model.allLayers() {
		if !model.isTiedOutputLayer(layer) {
			setters[layer.Name+".weights"] = layer.SetWeights
		}
		setters[layer.Name+".biases"] = layer.SetBiases
	}
	setters[model.TokenEmbedding.Name+".table"] = model.TokenEmbedding.SetTable
	if model.Settings.TieEmbeddings {
		// The output weight is the embedding table (set via ".table"); it is not
		// saved separately, so don't expect or require an "output.weights" entry.
		delete(setters, model.OutputLayer.Name+".weights")
	}
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

func (model *Model) isTiedOutputLayer(layer *perceptron.Layer) bool {
	return layer == model.OutputLayer && model.Settings.TieOutputToEmbedding
}

func (model *Model) compressedWeightSetters() map[string]func(rows *lowprecision.Rows) error {
	setters := map[string]func(rows *lowprecision.Rows) error{}
	for _, layer := range model.allLayers() {
		setters[layer.Name+".weights"] = layer.SetCompressedWeights
	}
	setters[model.TokenEmbedding.Name+".table"] = model.TokenEmbedding.SetCompressedTable
	return setters
}

func (model *Model) SetWeights(savedValues map[string][]float64) error {
	return model.SetWeightsAndCompressedWeights(savedValues, nil)
}

func (model *Model) SetWeightsAndCompressedWeights(savedValues map[string][]float64, savedCompressedValues map[string]*lowprecision.Rows) error {
	setters := model.weightSetters()
	compressedSetters := model.compressedWeightSetters()
	for name := range setters {
		_, foundValues := savedValues[name]
		_, foundCompressedValues := savedCompressedValues[name]
		if !foundValues && !foundCompressedValues {
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
	for name, rows := range savedCompressedValues {
		setter, found := compressedSetters[name]
		if !found {
			return fmt.Errorf("the model has no weights named %q that can hold compressed values", name)
		}
		if err := setter(rows); err != nil {
			return err
		}
	}
	_, outputWasSavedCompressed := savedCompressedValues[model.OutputLayer.Name+".weights"]
	if err := model.tieOutputAfterLoading(outputWasSavedCompressed); err != nil {
		return err
	}
	vectormath.MarkWeightsChanged()
	return nil
}

func (model *Model) tieOutputAfterLoading(outputWasSavedCompressed bool) error {
	if !model.Settings.TieOutputToEmbedding || outputWasSavedCompressed {
		return nil
	}
	if model.TokenEmbedding.IsCompressed() {
		return model.OutputLayer.SetCompressedWeights(model.TokenEmbedding.CompressedTable)
	}
	if model.OutputLayer.IsCompressed() {
		model.OutputLayer.DecompressWeights()
	}
	model.tieOutputToEmbedding()
	return nil
}

func (model *Model) parametersToSave() []parameter.Parameter {
	parameters := model.Parameters()
	if model.Settings.TieOutputToEmbedding && model.OutputLayer.IsCompressed() && !model.outputSharesTheEmbeddingTable() {
		outputLayer := model.OutputLayer
		outputWeights := parameter.Parameter{Name: outputLayer.Name + ".weights", CompressedValues: outputLayer.CompressedWeights, Rows: outputLayer.Weights.Rows, Columns: outputLayer.Weights.Columns, ReadOnly: true}
		parameters = append(parameters, outputWeights)
	}
	return parameters
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

func (model *Model) allDropouts() []*dropout.Dropout {
	var dropouts []*dropout.Dropout
	blocks := model.Blocks
	if model.MultiTokenPredictor != nil {
		blocks = append(append([]*Block(nil), blocks...), model.MultiTokenPredictor.Block)
	}
	for _, block := range blocks {
		dropouts = append(dropouts, block.AttentionDropout, block.FeedForwardDropout)
		switch blockAttention := block.Attention.(type) {
		case *attention.SelfAttention:
			dropouts = append(dropouts, blockAttention.WeightsDropout)
		case *attention.CompressedAttention:
			dropouts = append(dropouts, blockAttention.WeightsDropout)
		}
	}
	for _, layer := range model.allLayers() {
		if layer.Adapter != nil {
			dropouts = append(dropouts, layer.Adapter.InputDropout)
		}
	}
	return dropouts
}

func (model *Model) setDropoutActive(active bool) {
	for _, current := range model.allDropouts() {
		current.SetActive(active)
	}
}
