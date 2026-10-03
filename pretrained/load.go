package pretrained

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"transformer/parameter"
	"transformer/safetensors"
	"transformer/tokenizer"
	"transformer/transformer"
	"transformer/vectormath"
)

var layerTensorNames = map[string]string{
	"input_layernorm.weight":          "attentionNorm.weights",
	"post_attention_layernorm.weight": "feedForwardNorm.weights",
	"self_attn.q_proj.weight":         "attention.query.weights",
	"self_attn.q_proj.bias":           "attention.query.biases",
	"self_attn.k_proj.weight":         "attention.key.weights",
	"self_attn.k_proj.bias":           "attention.key.biases",
	"self_attn.v_proj.weight":         "attention.value.weights",
	"self_attn.v_proj.bias":           "attention.value.biases",
	"self_attn.o_proj.weight":         "attention.output.weights",
	"self_attn.o_proj.bias":           "attention.output.biases",
	"mlp.gate_proj.weight":            "feedForward.gate.weights",
	"mlp.up_proj.weight":              "feedForward.up.weights",
	"mlp.down_proj.weight":            "feedForward.down.weights",
}

func OurParameterName(huggingFaceName string) (string, bool) {
	switch huggingFaceName {
	case "model.embed_tokens.weight":
		return "tokens.table", true
	case "model.norm.weight":
		return "finalNorm.weights", true
	case "lm_head.weight":
		return "output.weights", true
	}
	afterPrefix, found := strings.CutPrefix(huggingFaceName, "model.layers.")
	if !found {
		return "", false
	}
	layerText, rest, found := strings.Cut(afterPrefix, ".")
	layer, err := strconv.Atoi(layerText)
	if !found || err != nil || layer < 0 {
		return "", false
	}
	ourName, known := layerTensorNames[rest]
	if !known {
		return "", false
	}
	return fmt.Sprintf("block%d.%s", layer+1, ourName), true
}

func isIgnoredTensor(huggingFaceName string) bool {
	return strings.HasSuffix(huggingFaceName, "rotary_emb.inv_freq")
}

func safetensorsFiles(folder string) ([]string, error) {
	single := filepath.Join(folder, "model.safetensors")
	if _, err := os.Stat(single); err == nil {
		return []string{single}, nil
	}
	indexPath := filepath.Join(folder, "model.safetensors.index.json")
	contents, err := os.ReadFile(indexPath)
	if err != nil {
		return nil, fmt.Errorf("%s has neither model.safetensors nor model.safetensors.index.json", folder)
	}
	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(contents, &index); err != nil {
		return nil, fmt.Errorf("%s: not a valid index: %w", indexPath, err)
	}
	seen := map[string]bool{}
	var files []string
	for _, fileName := range index.WeightMap {
		if !seen[fileName] {
			seen[fileName] = true
			files = append(files, filepath.Join(folder, fileName))
		}
	}
	sort.Strings(files)
	if len(files) == 0 {
		return nil, fmt.Errorf("%s lists no weight files", indexPath)
	}
	return files, nil
}

func LoadLlamaWeights(model *transformer.Model, config LlamaConfig, folder string) error {
	files, err := safetensorsFiles(folder)
	if err != nil {
		return err
	}
	parameters := map[string]parameter.Parameter{}
	for _, current := range model.Parameters() {
		parameters[current.Name] = current
	}

	loaded := map[string]bool{}
	for _, path := range files {
		opened, err := safetensors.Open(path)
		if err != nil {
			return err
		}
		for _, huggingFaceName := range opened.Names() {
			if isIgnoredTensor(huggingFaceName) {
				continue
			}
			ourName, known := OurParameterName(huggingFaceName)
			if !known {
				opened.Close()
				return fmt.Errorf("%s: tensor %q has no place in a %s model", path, huggingFaceName, config.ModelType)
			}
			if ourName == "output.weights" && config.TieWordEmbeddings {
				continue
			}
			target, found := parameters[ourName]
			if !found {
				opened.Close()
				return fmt.Errorf("%s: tensor %q would go into %q, but the model has no such parameter (does config.json match the weights?)", path, huggingFaceName, ourName)
			}
			tensor, err := opened.ReadTensor(huggingFaceName)
			if err != nil {
				opened.Close()
				return err
			}
			if len(tensor.Values) != len(target.Values) {
				opened.Close()
				return fmt.Errorf("%s: tensor %q has shape %v (%d values) but %q holds %d values", path, huggingFaceName, tensor.Shape, len(tensor.Values), ourName, len(target.Values))
			}
			if target.IsMatrix() && (len(tensor.Shape) != 2 || tensor.Shape[0] != target.Rows || tensor.Shape[1] != target.Columns) {
				opened.Close()
				return fmt.Errorf("%s: tensor %q has shape %v but %q is %dx%d", path, huggingFaceName, tensor.Shape, ourName, target.Rows, target.Columns)
			}
			copy(target.Values, tensor.Values)
			loaded[ourName] = true
		}
		opened.Close()
	}

	if !loaded["output.weights"] {
		embedding := parameters["tokens.table"]
		if !loaded["tokens.table"] {
			return fmt.Errorf("%s: the weights have no model.embed_tokens.weight", folder)
		}
		copy(parameters["output.weights"].Values, embedding.Values)
		loaded["output.weights"] = true
	}
	var missing []string
	for name := range parameters {
		if !loaded[name] && !strings.HasSuffix(name, ".biases") {
			missing = append(missing, name)
		}
	}
	sort.Strings(missing)
	if len(missing) > 0 {
		return fmt.Errorf("%s: the weights don't fill %d parameters, for example %v", folder, len(missing), missing[:min(len(missing), 5)])
	}
	vectormath.MarkWeightsChanged()
	return nil
}

func LoadLlama(folder string) (*transformer.Model, *tokenizer.Tokenizer, error) {
	config, err := ReadLlamaConfig(filepath.Join(folder, "config.json"))
	if err != nil {
		return nil, nil, err
	}
	if err := config.Check(); err != nil {
		return nil, nil, fmt.Errorf("%s: %w", folder, err)
	}
	model, err := transformer.NewModel(config.Settings())
	if err != nil {
		return nil, nil, err
	}
	if err := LoadLlamaWeights(model, config, folder); err != nil {
		return nil, nil, err
	}
	loadedTokenizer, err := tokenizer.LoadHuggingFace(filepath.Join(folder, "tokenizer.json"))
	if err != nil {
		return nil, nil, err
	}
	if loadedTokenizer.VocabularySize() > config.VocabularySize {
		return nil, nil, fmt.Errorf("%s: tokenizer has %d tokens but the model only has %d", folder, loadedTokenizer.VocabularySize(), config.VocabularySize)
	}
	return model, loadedTokenizer, nil
}
