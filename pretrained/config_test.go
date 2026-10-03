package pretrained

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const goodConfig = `{
  "model_type": "llama",
  "hidden_size": 8,
  "num_hidden_layers": 2,
  "num_attention_heads": 2,
  "num_key_value_heads": 1,
  "intermediate_size": 12,
  "rms_norm_eps": 1e-05,
  "rope_theta": 100000,
  "vocab_size": 20,
  "tie_word_embeddings": true,
  "hidden_act": "silu",
  "rope_scaling": null
}`

func writeConfig(t *testing.T, contents string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestGoodConfigBecomesSettings(t *testing.T) {
	config, err := ReadLlamaConfig(writeConfig(t, goodConfig))
	if err != nil {
		t.Fatal(err)
	}
	if err := config.Check(); err != nil {
		t.Fatal(err)
	}
	settings := config.Settings()
	if settings.VectorSize != 8 || settings.NumberOfBlocks != 2 || settings.NumberOfKeyValueHeads != 1 || settings.RotaryBase != 100000 || !settings.RotateHalves || settings.NormEpsilon != 1e-5 || settings.FeedForwardSize != 12 {
		t.Errorf("settings = %+v", settings)
	}
	if err := settings.Check(); err != nil {
		t.Errorf("settings from a good config don't pass Check: %v", err)
	}
}

func TestBadConfigsAreRejected(t *testing.T) {
	broken := map[string][2]string{
		"unsupported model type":  {`"model_type": "llama"`, `"model_type": "gpt2"`},
		"rope scaling":            {`"rope_scaling": null`, `"rope_scaling": {"type": "linear", "factor": 2}`},
		"different head size":     {`"hidden_act": "silu"`, `"hidden_act": "silu", "head_dim": 3`},
		"odd key value heads":     {`"num_key_value_heads": 1`, `"num_key_value_heads": 3`},
		"other activation":        {`"hidden_act": "silu"`, `"hidden_act": "gelu"`},
		"heads don't divide size": {`"num_attention_heads": 2`, `"num_attention_heads": 3`},
		"no layers":               {`"num_hidden_layers": 2`, `"num_hidden_layers": 0`},
		"sliding window":          {`"hidden_act": "silu",`, `"hidden_act": "silu", "use_sliding_window": true,`},
	}
	for name, change := range broken {
		contents := strings.Replace(goodConfig, change[0], change[1], 1)
		config, err := ReadLlamaConfig(writeConfig(t, contents))
		if err != nil {
			continue
		}
		if err := config.Check(); err == nil {
			t.Errorf("%s: config was accepted", name)
		}
	}
}

func TestTensorNamesMapToOurParameters(t *testing.T) {
	cases := map[string]string{
		"model.embed_tokens.weight":                      "tokens.table",
		"model.norm.weight":                              "finalNorm.weights",
		"lm_head.weight":                                 "output.weights",
		"model.layers.0.input_layernorm.weight":          "block1.attentionNorm.weights",
		"model.layers.11.self_attn.k_proj.weight":        "block12.attention.key.weights",
		"model.layers.3.self_attn.q_proj.bias":           "block4.attention.query.biases",
		"model.layers.29.mlp.down_proj.weight":           "block30.feedForward.down.weights",
		"model.layers.2.post_attention_layernorm.weight": "block3.feedForwardNorm.weights",
	}
	for huggingFaceName, want := range cases {
		if got, found := OurParameterName(huggingFaceName); !found || got != want {
			t.Errorf("%s -> %q, %v, want %q", huggingFaceName, got, found, want)
		}
	}
	for _, unknown := range []string{"model.layers.x.mlp.up_proj.weight", "model.layers.0.unknown.weight", "transformer.wte.weight"} {
		if _, found := OurParameterName(unknown); found {
			t.Errorf("%s should not map to anything", unknown)
		}
	}
}

func TestMissingFolderGivesAnError(t *testing.T) {
	if _, _, err := LoadLlama(filepath.Join(t.TempDir(), "nothing here")); err == nil {
		t.Error("loading an empty folder did not fail")
	}
}
