package transformer

import (
	"fmt"
	"github.com/javanhut/GoTransformers/attention"
	"github.com/javanhut/GoTransformers/lowprecision"
)

type AttentionKind string

const (
	StandardAttention          AttentionKind = "standard"
	CompressedSparseAttention  AttentionKind = "compressedSparse"
	HeavilyCompressedAttention AttentionKind = "heavilyCompressed"
)

type Settings struct {
	VocabularySize        int
	VectorSize            int
	NumberOfBlocks        int
	NumberOfHeads         int
	FeedForwardSize       int
	FeedForwardClampLimit float64

	AttentionPattern        []AttentionKind
	NumberOfKeyValueHeads   int
	ShareKeyAsValue         bool
	QueryRank               int
	NormalizeQueriesAndKeys bool
	UseAttentionSink        bool
	UseRotaryPositions      bool
	RotaryDimensions        int
	RotaryBase              float64
	RotateHalves            bool
	NormEpsilon             float64
	WindowSize              int
	FullAttentionEvery      int
	TopK                    int
	BlocksPerKeyValueGroup  int
	GroupSharingMode        attention.SharingMode

	CompressionRate          int
	HeavyCompressionRate     int
	CompressedWindowSize     int
	CompressedTopK           int
	CompressedOverlap        bool
	NumberOfIndexerHeads     int
	AttendToAllWhileTraining bool

	CachePrecision        lowprecision.Precision
	TrainAtCachePrecision bool

	WeightPrecision lowprecision.Precision

	UseMixtureOfExperts   bool
	NumberOfSharedExperts int
	NumberOfRoutedExperts int
	ExpertsPerToken       int
	ExpertHiddenSize      int
	HashRoutedBlocks      int
	BalanceUpdateRate     float64

	NumberOfResidualStreams int

	MultiTokenPrediction bool
	MultiTokenLossWeight float64

	AdapterRank  int
	AdapterAlpha float64

	ResidualDropout  float64
	AttentionDropout float64
	AdapterDropout   float64

	// TieEmbeddings shares one weight matrix between the token embedding and the
	// output projection (weight tying). It roughly halves the parameters a small
	// model spends on its vocabulary and usually improves quality. Only valid at
	// full precision (tied weights can't be compressed).
	TieEmbeddings bool
}

func SmallSettings(vocabularySize int) Settings {
	return Settings{
		VocabularySize:          vocabularySize,
		VectorSize:              64,
		NumberOfBlocks:          4,
		NumberOfHeads:           4,
		FeedForwardSize:         176,
		UseRotaryPositions:      true,
		BlocksPerKeyValueGroup:  1,
		GroupSharingMode:        attention.BorrowKeysAndValues,
		CompressionRate:         4,
		HeavyCompressionRate:    16,
		CompressedWindowSize:    8,
		CompressedTopK:          4,
		CompressedOverlap:       true,
		NumberOfIndexerHeads:    2,
		CachePrecision:          lowprecision.Float64,
		NumberOfSharedExperts:   1,
		NumberOfRoutedExperts:   4,
		ExpertsPerToken:         2,
		ExpertHiddenSize:        64,
		BalanceUpdateRate:       0.001,
		NumberOfResidualStreams: 1,
		MultiTokenLossWeight:    0.3,
	}
}

func DeepSeekStyleSettings(vocabularySize int) Settings {
	settings := SmallSettings(vocabularySize)
	settings.NumberOfBlocks = 6
	settings.AttentionPattern = []AttentionKind{
		StandardAttention, StandardAttention,
		CompressedSparseAttention, HeavilyCompressedAttention,
		CompressedSparseAttention, HeavilyCompressedAttention,
	}
	settings.WindowSize = 8
	settings.NormalizeQueriesAndKeys = true
	settings.UseAttentionSink = true
	settings.NumberOfKeyValueHeads = 1
	settings.ShareKeyAsValue = true
	settings.RotaryDimensions = 8
	settings.FeedForwardClampLimit = 10
	settings.UseMixtureOfExperts = true
	settings.HashRoutedBlocks = 1
	settings.NumberOfResidualStreams = 4
	settings.MultiTokenPrediction = true
	settings.CachePrecision = lowprecision.FP4
	settings.TrainAtCachePrecision = true
	return settings
}

func (settings Settings) headSize() int {
	return settings.VectorSize / settings.NumberOfHeads
}

func (settings Settings) residualStreams() int {
	if settings.NumberOfResidualStreams < 1 {
		return 1
	}
	return settings.NumberOfResidualStreams
}

func (settings Settings) attentionKindFor(blockIndex int) AttentionKind {
	if len(settings.AttentionPattern) == 0 {
		return StandardAttention
	}
	return settings.AttentionPattern[blockIndex%len(settings.AttentionPattern)]
}

func (settings Settings) Check() error {
	if settings.VocabularySize < 1 {
		return fmt.Errorf("VocabularySize must be at least 1, got %d", settings.VocabularySize)
	}
	if settings.VectorSize < 1 {
		return fmt.Errorf("VectorSize must be at least 1, got %d", settings.VectorSize)
	}
	if settings.NumberOfBlocks < 1 {
		return fmt.Errorf("NumberOfBlocks must be at least 1, got %d", settings.NumberOfBlocks)
	}
	if settings.NumberOfHeads < 1 || settings.VectorSize%settings.NumberOfHeads != 0 {
		return fmt.Errorf("VectorSize %d must split evenly into NumberOfHeads %d", settings.VectorSize, settings.NumberOfHeads)
	}
	if settings.NumberOfKeyValueHeads < 0 || (settings.NumberOfKeyValueHeads > 0 && settings.NumberOfHeads%settings.NumberOfKeyValueHeads != 0) {
		return fmt.Errorf("NumberOfHeads %d must split evenly into NumberOfKeyValueHeads %d (0 means one per head)", settings.NumberOfHeads, settings.NumberOfKeyValueHeads)
	}
	if settings.UseRotaryPositions {
		rotary := settings.RotaryDimensions
		if rotary == 0 {
			rotary = settings.headSize()
		}
		if rotary%2 != 0 || rotary > settings.headSize() || rotary < 0 {
			return fmt.Errorf("rotary positions need an even number of rotary dimensions no bigger than the head size %d, got %d", settings.headSize(), rotary)
		}
	}
	if settings.RotaryBase < 0 || settings.NormEpsilon < 0 {
		return fmt.Errorf("RotaryBase and NormEpsilon can't be negative (0 means the default), got %v and %v", settings.RotaryBase, settings.NormEpsilon)
	}
	if settings.QueryRank < 0 {
		return fmt.Errorf("QueryRank can't be negative, got %d", settings.QueryRank)
	}
	if settings.FeedForwardSize < 1 {
		return fmt.Errorf("FeedForwardSize must be at least 1, got %d", settings.FeedForwardSize)
	}
	if settings.FeedForwardClampLimit < 0 {
		return fmt.Errorf("FeedForwardClampLimit can't be negative, got %v", settings.FeedForwardClampLimit)
	}
	if settings.WindowSize < 0 || settings.FullAttentionEvery < 0 || settings.TopK < 0 {
		return fmt.Errorf("WindowSize, FullAttentionEvery and TopK can't be negative, got %d, %d and %d", settings.WindowSize, settings.FullAttentionEvery, settings.TopK)
	}
	if settings.BlocksPerKeyValueGroup < 1 {
		return fmt.Errorf("BlocksPerKeyValueGroup must be at least 1 (1 means no sharing), got %d", settings.BlocksPerKeyValueGroup)
	}
	if settings.BlocksPerKeyValueGroup > 1 && settings.GroupSharingMode != attention.BorrowKeysAndValues && settings.GroupSharingMode != attention.BorrowKeysValuesAndChoices {
		return fmt.Errorf("GroupSharingMode must be BorrowKeysAndValues or BorrowKeysValuesAndChoices when blocks share keys and values, got %v", settings.GroupSharingMode)
	}
	for blockIndex, kind := range settings.AttentionPattern {
		if kind != StandardAttention && kind != CompressedSparseAttention && kind != HeavilyCompressedAttention {
			return fmt.Errorf("AttentionPattern entry %d is %q, it must be %q, %q or %q", blockIndex, kind, StandardAttention, CompressedSparseAttention, HeavilyCompressedAttention)
		}
		if kind != StandardAttention {
			if settings.CompressionRate < 1 || settings.HeavyCompressionRate < 1 || settings.CompressedWindowSize < 1 || settings.CompressedTopK < 0 {
				return fmt.Errorf("compressed attention needs CompressionRate, HeavyCompressionRate and CompressedWindowSize of at least 1 and CompressedTopK of at least 0, got %d, %d, %d and %d", settings.CompressionRate, settings.HeavyCompressionRate, settings.CompressedWindowSize, settings.CompressedTopK)
			}
			if settings.NumberOfIndexerHeads < 0 {
				return fmt.Errorf("NumberOfIndexerHeads can't be negative, got %d", settings.NumberOfIndexerHeads)
			}
		}
	}
	if settings.WeightPrecision < lowprecision.Float64 || settings.WeightPrecision > lowprecision.FP4 {
		return fmt.Errorf("unknown WeightPrecision %v", settings.WeightPrecision)
	}
	if settings.CachePrecision < lowprecision.Float64 || settings.CachePrecision > lowprecision.FP4 {
		return fmt.Errorf("unknown CachePrecision %v", settings.CachePrecision)
	}
	if settings.UseMixtureOfExperts {
		if settings.NumberOfRoutedExperts < 1 || settings.ExpertsPerToken < 1 || settings.ExpertsPerToken > settings.NumberOfRoutedExperts {
			return fmt.Errorf("mixture of experts needs at least 1 routed expert and between 1 and NumberOfRoutedExperts experts per token, got %d routed and %d per token", settings.NumberOfRoutedExperts, settings.ExpertsPerToken)
		}
		if settings.NumberOfSharedExperts < 0 || settings.ExpertHiddenSize < 1 || settings.HashRoutedBlocks < 0 || settings.BalanceUpdateRate < 0 {
			return fmt.Errorf("mixture of experts needs NumberOfSharedExperts >= 0, ExpertHiddenSize >= 1, HashRoutedBlocks >= 0 and BalanceUpdateRate >= 0")
		}
	}
	if settings.NumberOfResidualStreams < 0 {
		return fmt.Errorf("NumberOfResidualStreams can't be negative (1 means a normal residual connection), got %d", settings.NumberOfResidualStreams)
	}
	for name, rate := range map[string]float64{"ResidualDropout": settings.ResidualDropout, "AttentionDropout": settings.AttentionDropout, "AdapterDropout": settings.AdapterDropout} {
		if rate < 0 || rate >= 1 {
			return fmt.Errorf("%s must be at least 0 and below 1, got %v", name, rate)
		}
	}
	if settings.AdapterRank < 0 || settings.AdapterAlpha < 0 {
		return fmt.Errorf("AdapterRank and AdapterAlpha can't be negative (0 means no adapters), got %d and %v", settings.AdapterRank, settings.AdapterAlpha)
	}
	if settings.MultiTokenLossWeight < 0 {
		return fmt.Errorf("MultiTokenLossWeight can't be negative, got %v", settings.MultiTokenLossWeight)
	}
	if settings.TieEmbeddings && settings.WeightPrecision != lowprecision.Float64 {
		return fmt.Errorf("TieEmbeddings needs full-precision weights (tied weights can't be compressed)")
	}
	return nil
}

func (settings Settings) blockUsesWindow(blockIndex int) bool {
	if settings.WindowSize == 0 {
		return false
	}
	if settings.FullAttentionEvery > 0 && (blockIndex+1)%settings.FullAttentionEvery == 0 {
		return false
	}
	return true
}

func (settings Settings) standardOptions() attention.Options {
	return attention.Options{
		NumberOfHeads:           settings.NumberOfHeads,
		NumberOfKeyValueHeads:   settings.NumberOfKeyValueHeads,
		HideFutureTokens:        true,
		ShareKeyAsValue:         settings.ShareKeyAsValue,
		QueryRank:               settings.QueryRank,
		NormalizeQueriesAndKeys: settings.NormalizeQueriesAndKeys,
		UseAttentionSink:        settings.UseAttentionSink,
		UseRotaryPositions:      settings.UseRotaryPositions,
		RotaryDimensions:        settings.RotaryDimensions,
		RotaryBase:              settings.RotaryBase,
		RotateHalves:            settings.RotateHalves,
		CachePrecision:          settings.CachePrecision,
		TrainAtCachePrecision:   settings.TrainAtCachePrecision,
		AttentionDropout:        settings.AttentionDropout,
	}
}

func (settings Settings) compressedOptions(kind AttentionKind) attention.CompressedOptions {
	options := attention.CompressedOptions{
		NumberOfHeads:         settings.NumberOfHeads,
		WindowSize:            settings.CompressedWindowSize,
		NumberOfIndexerHeads:  settings.NumberOfIndexerHeads,
		UseAttentionSink:      settings.UseAttentionSink,
		UseRotaryPositions:    settings.UseRotaryPositions,
		RotaryDimensions:      settings.RotaryDimensions,
		CachePrecision:        settings.CachePrecision,
		TrainAtCachePrecision: settings.TrainAtCachePrecision,
		AttentionDropout:      settings.AttentionDropout,
	}
	if kind == CompressedSparseAttention {
		options.CompressionRate = settings.CompressionRate
		options.TopK = settings.CompressedTopK
		options.Overlap = settings.CompressedOverlap
	} else {
		options.CompressionRate = settings.HeavyCompressionRate
	}
	return options
}
