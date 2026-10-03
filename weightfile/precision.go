package weightfile

import (
	"fmt"
	"math"
)

type StoragePrecision int

const (
	Float64 StoragePrecision = iota
	Float32
	BFloat16
)

func (precision StoragePrecision) String() string {
	switch precision {
	case Float64:
		return "Float64"
	case Float32:
		return "Float32"
	case BFloat16:
		return "BFloat16"
	}
	return fmt.Sprintf("StoragePrecision(%d)", int(precision))
}

func (precision StoragePrecision) BytesPerValue() int {
	switch precision {
	case Float32:
		return 4
	case BFloat16:
		return 2
	}
	return 8
}

func (precision StoragePrecision) check() error {
	if precision < Float64 || precision > BFloat16 {
		return fmt.Errorf("unknown storage precision %v, use Float64, Float32 or BFloat16", precision)
	}
	return nil
}

func (precision StoragePrecision) Round(value float64) float64 {
	switch precision {
	case Float32:
		return float64(float32(value))
	case BFloat16:
		return bFloat16BitsToFloat64(float64ToBFloat16Bits(value))
	}
	return value
}

const smallestBFloat16NormalExponent = -126
const bFloat16FractionBits = 7

func float64ToBFloat16Bits(value float64) uint16 {
	if math.IsNaN(value) {
		return 0x7FC0
	}
	if value == 0 || math.IsInf(value, 0) {
		return uint16(math.Float32bits(float32(value)) >> 16)
	}
	_, exponentAboveFraction := math.Frexp(value)
	exponent := max(exponentAboveFraction-1, smallestBFloat16NormalExponent)
	spacingExponent := exponent - bFloat16FractionBits
	stepsOfSpacing := math.RoundToEven(math.Ldexp(value, -spacingExponent))
	rounded := math.Ldexp(stepsOfSpacing, spacingExponent)
	return uint16(math.Float32bits(float32(rounded)) >> 16)
}

func bFloat16BitsToFloat64(bits uint16) float64 {
	return float64(math.Float32frombits(uint32(bits) << 16))
}
