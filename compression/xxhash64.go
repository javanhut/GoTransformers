package compression

import (
	"encoding/binary"
	"math/bits"
)

const (
	xxhashPrime1 uint64 = 11400714785074694791
	xxhashPrime2 uint64 = 14029467366897019727
	xxhashPrime3 uint64 = 1609587929392839161
	xxhashPrime4 uint64 = 9650029242287828579
	xxhashPrime5 uint64 = 2870177450012600261
)

func xxhashRound(accumulator uint64, input uint64) uint64 {
	accumulator += input * xxhashPrime2
	accumulator = bits.RotateLeft64(accumulator, 31)
	return accumulator * xxhashPrime1
}

func xxhashMergeRound(accumulator uint64, value uint64) uint64 {
	accumulator ^= xxhashRound(0, value)
	return accumulator*xxhashPrime1 + xxhashPrime4
}

func xxhash64(data []byte) uint64 {
	position := 0
	var hash uint64
	if len(data) >= 32 {
		seed := uint64(0)
		lane1 := seed + xxhashPrime1 + xxhashPrime2
		lane2 := seed + xxhashPrime2
		lane3 := seed
		lane4 := seed - xxhashPrime1
		for position+32 <= len(data) {
			lane1 = xxhashRound(lane1, binary.LittleEndian.Uint64(data[position:]))
			lane2 = xxhashRound(lane2, binary.LittleEndian.Uint64(data[position+8:]))
			lane3 = xxhashRound(lane3, binary.LittleEndian.Uint64(data[position+16:]))
			lane4 = xxhashRound(lane4, binary.LittleEndian.Uint64(data[position+24:]))
			position += 32
		}
		hash = bits.RotateLeft64(lane1, 1) + bits.RotateLeft64(lane2, 7) + bits.RotateLeft64(lane3, 12) + bits.RotateLeft64(lane4, 18)
		hash = xxhashMergeRound(hash, lane1)
		hash = xxhashMergeRound(hash, lane2)
		hash = xxhashMergeRound(hash, lane3)
		hash = xxhashMergeRound(hash, lane4)
	} else {
		hash = xxhashPrime5
	}
	hash += uint64(len(data))

	for position+8 <= len(data) {
		hash ^= xxhashRound(0, binary.LittleEndian.Uint64(data[position:]))
		hash = bits.RotateLeft64(hash, 27)*xxhashPrime1 + xxhashPrime4
		position += 8
	}
	if position+4 <= len(data) {
		hash ^= uint64(binary.LittleEndian.Uint32(data[position:])) * xxhashPrime1
		hash = bits.RotateLeft64(hash, 23)*xxhashPrime2 + xxhashPrime3
		position += 4
	}
	for position < len(data) {
		hash ^= uint64(data[position]) * xxhashPrime5
		hash = bits.RotateLeft64(hash, 11) * xxhashPrime1
		position++
	}

	hash ^= hash >> 33
	hash *= xxhashPrime2
	hash ^= hash >> 29
	hash *= xxhashPrime3
	hash ^= hash >> 32
	return hash
}
