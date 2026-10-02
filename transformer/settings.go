package transformer

import (
	"fmt"
	"transformer/attention"
	"transformer/lowprecision"
)

type Settings struct {
	VocabularySize         int
	VectorSize             int
	NumberOfBlocks         int
	NumberOfHeads          int
	FeedForwardSize        int
	FeedForwardClampLimit  float64
	UseRotaryPositions     bool
	WindowSize             int
	FullAttentionEvery     int
	TopK                   int
	BlocksPerKeyValueGroup int
	GroupSharingMode       attention.SharingMode
	CachePrecision         lowprecision.Precision
	TrainAtCachePrecision  bool
}

func SmallSettings(vocabularySize int) Settings {
	return Settings{
		VocabularySize:         vocabularySize,
		VectorSize:             64,
		NumberOfBlocks:         4,
		NumberOfHeads:          4,
		FeedForwardSize:        176,
		UseRotaryPositions:     true,
		BlocksPerKeyValueGroup: 1,
		GroupSharingMode:       attention.BorrowKeysAndValues,
		CachePrecision:         lowprecision.Float64,
	}
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
	if settings.UseRotaryPositions && (settings.VectorSize/settings.NumberOfHeads)%2 != 0 {
		return fmt.Errorf("rotary positions need an even head size, VectorSize %d / NumberOfHeads %d is odd", settings.VectorSize, settings.NumberOfHeads)
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
	if settings.CachePrecision < lowprecision.Float64 || settings.CachePrecision > lowprecision.FP4 {
		return fmt.Errorf("unknown CachePrecision %v", settings.CachePrecision)
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

func (settings Settings) blockOwnsKeysAndValues(blockIndex int) bool {
	return blockIndex%settings.BlocksPerKeyValueGroup == 0
}
