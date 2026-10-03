package pretrained

import (
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"transformer/safetensors"
)

func downloadedModelFolder() string {
	if folder := os.Getenv("GOTRANSFORMERS_SMOLLM2"); folder != "" {
		return folder
	}
	return "/tmp/claude-1000/-home-javanstorm-Development-GoTransformers/f9636082-b0cb-4f11-a15e-fe96206115e4/scratchpad/smollm2"
}

func needDownloadedModel(t *testing.T) string {
	t.Helper()
	folder := downloadedModelFolder()
	if _, err := os.Stat(filepath.Join(folder, "model.safetensors")); err != nil {
		t.Skipf("SmolLM2-135M is not downloaded (%v)", err)
	}
	return folder
}

type referenceLlama struct {
	config  LlamaConfig
	tensors map[string]safetensors.Tensor
}

func (reference referenceLlama) tensor(name string) []float64 {
	tensor, found := reference.tensors[name]
	if !found {
		panic("reference model has no tensor " + name)
	}
	return tensor.Values
}

func referenceRMSNorm(values []float64, weights []float64, epsilon float64) []float64 {
	sumOfSquares := 0.0
	for _, value := range values {
		sumOfSquares += value * value
	}
	scale := 1 / math.Sqrt(sumOfSquares/float64(len(values))+epsilon)
	result := make([]float64, len(values))
	for i, value := range values {
		result[i] = value * scale * weights[i]
	}
	return result
}

func referenceTimes(weights []float64, outputs int, inputs []float64) []float64 {
	result := make([]float64, outputs)
	for row := 0; row < outputs; row++ {
		sum := 0.0
		for column, input := range inputs {
			sum += weights[row*len(inputs)+column] * input
		}
		result[row] = sum
	}
	return result
}

func referenceRotateHalf(head []float64, position int, base float64) []float64 {
	half := len(head) / 2
	result := make([]float64, len(head))
	for i := 0; i < half; i++ {
		angle := float64(position) / math.Pow(base, float64(2*i)/float64(len(head)))
		first := head[i]
		second := head[i+half]
		result[i] = first*math.Cos(angle) - second*math.Sin(angle)
		result[i+half] = second*math.Cos(angle) + first*math.Sin(angle)
	}
	return result
}

func (reference referenceLlama) lastLogits(tokenIDs []int) []float64 {
	config := reference.config
	size := config.HiddenSize
	headSize := size / config.NumberOfHeads
	keyValueHeads := config.NumberOfKeyValueHeads
	embedding := reference.tensor("model.embed_tokens.weight")

	hidden := make([][]float64, len(tokenIDs))
	for position, tokenID := range tokenIDs {
		hidden[position] = append([]float64(nil), embedding[tokenID*size:(tokenID+1)*size]...)
	}

	for layer := 0; layer < config.NumberOfLayers; layer++ {
		prefix := "model.layers." + strconv.Itoa(layer) + "."
		queries := make([][]float64, len(tokenIDs))
		keys := make([][]float64, len(tokenIDs))
		values := make([][]float64, len(tokenIDs))
		for position := range tokenIDs {
			normalized := referenceRMSNorm(hidden[position], reference.tensor(prefix+"input_layernorm.weight"), config.RMSNormEpsilon)
			query := referenceTimes(reference.tensor(prefix+"self_attn.q_proj.weight"), config.NumberOfHeads*headSize, normalized)
			key := referenceTimes(reference.tensor(prefix+"self_attn.k_proj.weight"), keyValueHeads*headSize, normalized)
			values[position] = referenceTimes(reference.tensor(prefix+"self_attn.v_proj.weight"), keyValueHeads*headSize, normalized)
			for head := 0; head < config.NumberOfHeads; head++ {
				copy(query[head*headSize:(head+1)*headSize], referenceRotateHalf(query[head*headSize:(head+1)*headSize], position, config.RopeTheta))
			}
			for head := 0; head < keyValueHeads; head++ {
				copy(key[head*headSize:(head+1)*headSize], referenceRotateHalf(key[head*headSize:(head+1)*headSize], position, config.RopeTheta))
			}
			queries[position] = query
			keys[position] = key
		}

		for position := range tokenIDs {
			attended := make([]float64, config.NumberOfHeads*headSize)
			for head := 0; head < config.NumberOfHeads; head++ {
				keyValueHead := head / (config.NumberOfHeads / keyValueHeads)
				query := queries[position][head*headSize : (head+1)*headSize]
				scores := make([]float64, position+1)
				largest := math.Inf(-1)
				for other := 0; other <= position; other++ {
					key := keys[other][keyValueHead*headSize : (keyValueHead+1)*headSize]
					dot := 0.0
					for i := range query {
						dot += query[i] * key[i]
					}
					scores[other] = dot / math.Sqrt(float64(headSize))
					largest = math.Max(largest, scores[other])
				}
				total := 0.0
				for other := range scores {
					scores[other] = math.Exp(scores[other] - largest)
					total += scores[other]
				}
				for other := range scores {
					value := values[other][keyValueHead*headSize : (keyValueHead+1)*headSize]
					for i := range value {
						attended[head*headSize+i] += scores[other] / total * value[i]
					}
				}
			}
			output := referenceTimes(reference.tensor(prefix+"self_attn.o_proj.weight"), size, attended)
			for i := range output {
				hidden[position][i] += output[i]
			}
		}

		for position := range tokenIDs {
			normalized := referenceRMSNorm(hidden[position], reference.tensor(prefix+"post_attention_layernorm.weight"), config.RMSNormEpsilon)
			gate := referenceTimes(reference.tensor(prefix+"mlp.gate_proj.weight"), config.IntermediateSize, normalized)
			up := referenceTimes(reference.tensor(prefix+"mlp.up_proj.weight"), config.IntermediateSize, normalized)
			mixed := make([]float64, config.IntermediateSize)
			for i := range mixed {
				mixed[i] = gate[i] / (1 + math.Exp(-gate[i])) * up[i]
			}
			down := referenceTimes(reference.tensor(prefix+"mlp.down_proj.weight"), size, mixed)
			for i := range down {
				hidden[position][i] += down[i]
			}
		}
	}

	last := referenceRMSNorm(hidden[len(tokenIDs)-1], reference.tensor("model.norm.weight"), config.RMSNormEpsilon)
	outputWeights := embedding
	if !config.TieWordEmbeddings {
		outputWeights = reference.tensor("lm_head.weight")
	}
	return referenceTimes(outputWeights, config.VocabularySize, last)
}

func TestSmolLM2MatchesReference(t *testing.T) {
	folder := needDownloadedModel(t)
	model, loadedTokenizer, err := LoadLlama(folder)
	if err != nil {
		t.Fatal(err)
	}
	config, _ := ReadLlamaConfig(filepath.Join(folder, "config.json"))
	tensors, err := safetensors.Read(filepath.Join(folder, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	reference := referenceLlama{config: config, tensors: tensors}

	tokenIDs := loadedTokenizer.Encode("The capital of France is")
	want := reference.lastLogits(tokenIDs)
	allAtOnce := model.Forward(tokenIDs).Row(len(tokenIDs) - 1)
	model.StartGenerating()
	oneAtATime := model.Feed(tokenIDs)

	largestDifference := 0.0
	largestOneAtATimeDifference := 0.0
	for i := range want {
		largestDifference = math.Max(largestDifference, math.Abs(allAtOnce[i]-want[i]))
		largestOneAtATimeDifference = math.Max(largestOneAtATimeDifference, math.Abs(oneAtATime[i]-want[i]))
	}
	t.Logf("%d prompt tokens, largest logit difference from the reference: all at once %.3g, one token at a time with the cache %.3g", len(tokenIDs), largestDifference, largestOneAtATimeDifference)
	if largestDifference > 1e-6 || largestOneAtATimeDifference > 1e-6 {
		t.Errorf("logits differ from the reference by up to %v (all at once) and %v (one at a time)", largestDifference, largestOneAtATimeDifference)
	}
	if indexOfLargest(want) != indexOfLargest(allAtOnce) {
		t.Errorf("reference picks %q but our model picks %q", loadedTokenizer.TokenText(indexOfLargest(want)), loadedTokenizer.TokenText(indexOfLargest(allAtOnce)))
	}
}

func indexOfLargest(values []float64) int {
	best := 0
	for i, value := range values {
		if value > values[best] {
			best = i
		}
	}
	return best
}

func topTokens(scores []float64, count int) []int {
	var best []int
	used := map[int]bool{}
	for len(best) < count {
		choice := -1
		for id, score := range scores {
			if !used[id] && (choice == -1 || score > scores[choice]) {
				choice = id
			}
		}
		used[choice] = true
		best = append(best, choice)
	}
	return best
}

func TestSmolLM2GeneratesSensibleText(t *testing.T) {
	folder := needDownloadedModel(t)
	model, loadedTokenizer, err := LoadLlama(folder)
	if err != nil {
		t.Fatal(err)
	}

	capitalPrompt := loadedTokenizer.Encode("The capital of France is")
	scores := model.Forward(capitalPrompt).Row(len(capitalPrompt) - 1)
	var topWords []string
	for _, id := range topTokens(scores, 3) {
		topWords = append(topWords, loadedTokenizer.TokenText(id))
	}
	if !strings.Contains(strings.Join(topWords, "|"), " Paris") {
		t.Errorf("expected \" Paris\" among the 3 most likely next tokens, got %q", topWords)
	}
	t.Logf("The capital of France is%s", loadedTokenizer.Decode(model.Generate(capitalPrompt, 30, 0)))

	story := loadedTokenizer.Decode(model.Generate(loadedTokenizer.Encode("Once upon a time, there was a"), 25, 0))
	t.Logf("Once upon a time, there was a%s", story)
	if !strings.Contains(story, " named ") && !strings.Contains(story, " lived ") {
		t.Errorf("story continuation doesn't read like a story: %q", story)
	}

	code := loadedTokenizer.Decode(model.Generate(loadedTokenizer.Encode("def fibonacci(n):"), 25, 0))
	t.Logf("def fibonacci(n):%s", code)
	if !strings.Contains(code, "return") || !strings.Contains(code, "if n") {
		t.Errorf("fibonacci continuation doesn't look like code: %q", code)
	}
}
