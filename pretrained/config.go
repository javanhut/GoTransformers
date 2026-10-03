package pretrained

import (
	"encoding/json"
	"fmt"
	"github.com/javanhut/GoTransformers/attention"
	"github.com/javanhut/GoTransformers/lowprecision"
	"github.com/javanhut/GoTransformers/transformer"
	"os"
	"strings"
)

type LlamaConfig struct {
	ModelType             string          `json:"model_type"`
	HiddenSize            int             `json:"hidden_size"`
	NumberOfLayers        int             `json:"num_hidden_layers"`
	NumberOfHeads         int             `json:"num_attention_heads"`
	NumberOfKeyValueHeads int             `json:"num_key_value_heads"`
	IntermediateSize      int             `json:"intermediate_size"`
	RMSNormEpsilon        float64         `json:"rms_norm_eps"`
	RopeTheta             float64         `json:"rope_theta"`
	VocabularySize        int             `json:"vocab_size"`
	TieWordEmbeddings     bool            `json:"tie_word_embeddings"`
	HiddenActivation      string          `json:"hidden_act"`
	HeadDimension         int             `json:"head_dim"`
	RopeScaling           json.RawMessage `json:"rope_scaling"`
	MLPBias               bool            `json:"mlp_bias"`
	UseSlidingWindow      bool            `json:"use_sliding_window"`
	RopeInterleaved       bool            `json:"rope_interleaved"`
}

func ReadLlamaConfig(path string) (LlamaConfig, error) {
	contents, err := os.ReadFile(path)
	if err != nil {
		return LlamaConfig{}, err
	}
	var config LlamaConfig
	if err := json.Unmarshal(contents, &config); err != nil {
		return LlamaConfig{}, fmt.Errorf("%s: not a valid config.json: %w", path, err)
	}
	if config.NumberOfKeyValueHeads == 0 {
		config.NumberOfKeyValueHeads = config.NumberOfHeads
	}
	if config.RopeTheta == 0 {
		config.RopeTheta = 10000
	}
	if config.RMSNormEpsilon == 0 {
		config.RMSNormEpsilon = 1e-6
	}
	if config.HiddenActivation == "" {
		config.HiddenActivation = "silu"
	}
	return config, nil
}

func (config LlamaConfig) Check() error {
	if config.ModelType != "llama" && config.ModelType != "qwen2" {
		return fmt.Errorf("model_type %q is not supported, only llama and qwen2 are", config.ModelType)
	}
	if config.HiddenSize < 1 || config.NumberOfLayers < 1 || config.NumberOfHeads < 1 || config.IntermediateSize < 1 || config.VocabularySize < 1 {
		return fmt.Errorf("hidden_size, num_hidden_layers, num_attention_heads, intermediate_size and vocab_size must all be at least 1, got %d, %d, %d, %d and %d", config.HiddenSize, config.NumberOfLayers, config.NumberOfHeads, config.IntermediateSize, config.VocabularySize)
	}
	if config.HiddenSize%config.NumberOfHeads != 0 {
		return fmt.Errorf("hidden_size %d doesn't split evenly into %d heads", config.HiddenSize, config.NumberOfHeads)
	}
	if config.HeadDimension != 0 && config.HeadDimension != config.HiddenSize/config.NumberOfHeads {
		return fmt.Errorf("head_dim %d is different from hidden_size / num_attention_heads = %d, which is not supported yet", config.HeadDimension, config.HiddenSize/config.NumberOfHeads)
	}
	if config.NumberOfKeyValueHeads < 1 || config.NumberOfHeads%config.NumberOfKeyValueHeads != 0 {
		return fmt.Errorf("%d attention heads don't split evenly into %d key/value heads", config.NumberOfHeads, config.NumberOfKeyValueHeads)
	}
	if (config.HiddenSize/config.NumberOfHeads)%2 != 0 {
		return fmt.Errorf("rotary positions need an even head size, got %d", config.HiddenSize/config.NumberOfHeads)
	}
	trimmedScaling := strings.TrimSpace(string(config.RopeScaling))
	if trimmedScaling != "" && trimmedScaling != "null" {
		return fmt.Errorf("rope_scaling %s is not supported yet", trimmedScaling)
	}
	if config.HiddenActivation != "silu" && config.HiddenActivation != "swish" {
		return fmt.Errorf("hidden_act %q is not supported, only silu is", config.HiddenActivation)
	}
	if config.MLPBias {
		return fmt.Errorf("mlp_bias true is not supported yet")
	}
	if config.UseSlidingWindow {
		return fmt.Errorf("use_sliding_window true is not supported yet")
	}
	if config.RMSNormEpsilon < 0 || config.RopeTheta <= 0 {
		return fmt.Errorf("rms_norm_eps must not be negative and rope_theta must be above 0, got %v and %v", config.RMSNormEpsilon, config.RopeTheta)
	}
	return nil
}

func (config LlamaConfig) Settings() transformer.Settings {
	return transformer.Settings{
		VocabularySize:          config.VocabularySize,
		VectorSize:              config.HiddenSize,
		NumberOfBlocks:          config.NumberOfLayers,
		NumberOfHeads:           config.NumberOfHeads,
		NumberOfKeyValueHeads:   config.NumberOfKeyValueHeads,
		FeedForwardSize:         config.IntermediateSize,
		UseRotaryPositions:      true,
		RotateHalves:            !config.RopeInterleaved,
		RotaryBase:              config.RopeTheta,
		NormEpsilon:             config.RMSNormEpsilon,
		BlocksPerKeyValueGroup:  1,
		GroupSharingMode:        attention.BorrowKeysAndValues,
		CachePrecision:          lowprecision.Float64,
		NumberOfResidualStreams: 1,
	}
}
