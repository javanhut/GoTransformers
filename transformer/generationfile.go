package transformer

import (
	"bufio"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"os"
	"transformer/attention"
	"transformer/lowprecision"
)

const generationFileFormat = "gotransformers-generation-1"

type blockGenerationSnapshot struct {
	Kind       AttentionKind
	Standard   attention.GenerationSnapshot
	Compressed attention.CompressedGenerationSnapshot
}

type generationFile struct {
	Format             string
	ModelFingerprint   uint64
	GeneratedPositions int
	LastScores         []float64
	Blocks             []blockGenerationSnapshot
}

func (model *Model) Fingerprint() uint64 {
	hash := fnv.New64a()
	settingsText, _ := json.Marshal(model.Settings)
	hash.Write(settingsText)
	writeNumbers := func(values []float64) {
		for _, value := range values {
			bits := math.Float64bits(value)
			var bytes [8]byte
			for i := range bytes {
				bytes[i] = byte(bits >> (8 * i))
			}
			hash.Write(bytes[:])
		}
	}
	for _, current := range model.Parameters() {
		hash.Write([]byte(current.Name))
		writeNumbers(current.Values)
	}
	writeCompressed := func(rows *lowprecision.Rows) {
		snapshot := rows.Snapshot()
		writeNumbers(snapshot.Float64Values)
		for _, value := range snapshot.Float32Values {
			writeNumbers([]float64{float64(value)})
		}
		for _, value := range snapshot.Scales {
			writeNumbers([]float64{float64(value)})
		}
		for _, value := range snapshot.Int8Values {
			hash.Write([]byte{byte(value)})
		}
		hash.Write(snapshot.FP4Values)
	}
	for _, layer := range model.allLayers() {
		if layer.IsCompressed() {
			writeCompressed(layer.CompressedWeights)
		}
	}
	if model.TokenEmbedding.IsCompressed() {
		writeCompressed(model.TokenEmbedding.CompressedTable)
	}
	return hash.Sum64()
}

func (model *Model) SaveGenerationState(path string) error {
	if model.lastScores == nil {
		return fmt.Errorf("nothing to save, feed the model at least 1 token first")
	}
	saved := generationFile{
		Format:             generationFileFormat,
		ModelFingerprint:   model.Fingerprint(),
		GeneratedPositions: model.generatedPositions,
		LastScores:         model.lastScores,
	}
	for _, block := range model.Blocks {
		switch blockAttention := block.Attention.(type) {
		case *attention.SelfAttention:
			saved.Blocks = append(saved.Blocks, blockGenerationSnapshot{Kind: StandardAttention, Standard: blockAttention.GenerationSnapshot()})
		case *attention.CompressedAttention:
			saved.Blocks = append(saved.Blocks, blockGenerationSnapshot{Kind: CompressedSparseAttention, Compressed: blockAttention.GenerationSnapshot()})
		default:
			return fmt.Errorf("don't know how to save the cache of attention type %T", block.Attention)
		}
	}

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	writer := bufio.NewWriter(file)
	if err := gob.NewEncoder(writer).Encode(saved); err != nil {
		file.Close()
		return err
	}
	if err := writer.Flush(); err != nil {
		file.Close()
		return err
	}
	return file.Close()
}

func (model *Model) LoadGenerationState(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	var saved generationFile
	if err := gob.NewDecoder(bufio.NewReader(file)).Decode(&saved); err != nil {
		return fmt.Errorf("%s: this is not a saved generation state: %w", path, err)
	}
	if saved.Format != generationFileFormat {
		return fmt.Errorf("%s: unknown generation state format %q", path, saved.Format)
	}
	if saved.ModelFingerprint != model.Fingerprint() {
		return fmt.Errorf("%s: this cache was made by a model with different settings or weights", path)
	}
	if len(saved.Blocks) != len(model.Blocks) {
		return fmt.Errorf("%s: saved %d blocks but the model has %d", path, len(saved.Blocks), len(model.Blocks))
	}

	for blockIndex, block := range model.Blocks {
		snapshot := saved.Blocks[blockIndex]
		switch blockAttention := block.Attention.(type) {
		case *attention.SelfAttention:
			if snapshot.Kind != StandardAttention {
				return fmt.Errorf("%s: block %d was saved as %q attention", path, blockIndex+1, snapshot.Kind)
			}
			if err := blockAttention.RestoreGeneration(snapshot.Standard); err != nil {
				return err
			}
		case *attention.CompressedAttention:
			if snapshot.Kind != CompressedSparseAttention {
				return fmt.Errorf("%s: block %d was saved as %q attention", path, blockIndex+1, snapshot.Kind)
			}
			if err := blockAttention.RestoreGeneration(snapshot.Compressed); err != nil {
				return err
			}
		}
	}
	model.generatedPositions = saved.GeneratedPositions
	model.lastScores = saved.LastScores
	return nil
}
