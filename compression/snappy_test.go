package compression

import (
	"bytes"
	"encoding/binary"
	"math/rand"
	"strings"
	"testing"
)

func TestSnappyHandWrittenElements(t *testing.T) {
	threeHundredBytes := strings.Repeat("0123456789", 30)
	cases := []struct {
		name       string
		compressed []byte
		expected   string
	}{
		{"literal", []byte{5, 4 << 2, 'h', 'e', 'l', 'l', 'o'}, "hello"},
		{"copy with one byte offset, overlapping", []byte{8, 0, 'a', 1 | 3<<2, 1}, "aaaaaaaa"},
		{"copy with one byte offset", []byte{8, 3 << 2, 'a', 'b', 'c', 'd', 1, 4}, "abcdabcd"},
		{"copy with two byte offset", []byte{9, 2 << 2, 'x', 'y', 'z', 2 | 5<<2, 3, 0}, "xyzxyzxyz"},
		{"copy with four byte offset", []byte{5, 1 << 2, 'p', 'q', 3 | 2<<2, 2, 0, 0, 0}, "pqpqp"},
		{"literal with one length byte", append([]byte{100, 60 << 2, 99}, []byte(strings.Repeat("z", 100))...), strings.Repeat("z", 100)},
		{"copy with high offset bits", append(append([]byte{0xb0, 0x02, 61 << 2, 0x2b, 0x01}, []byte(threeHundredBytes)...), 1|1<<5, 0x2c), threeHundredBytes + threeHundredBytes[:4]},
	}
	for _, testCase := range cases {
		decompressed, err := DecompressSnappy(testCase.compressed)
		if err != nil {
			t.Errorf("%s: %v", testCase.name, err)
			continue
		}
		if string(decompressed) != testCase.expected {
			t.Errorf("%s: got %q, want %q", testCase.name, decompressed, testCase.expected)
		}
	}
}

func TestSnappyRejectsBrokenInput(t *testing.T) {
	cases := map[string][]byte{
		"empty":                   {},
		"offset of zero":          {8, 0, 'a', 1 | 3<<2, 0},
		"offset before the start": {8, 0, 'a', 1 | 3<<2, 2},
		"too short":               {9, 4 << 2, 'h', 'e', 'l', 'l', 'o'},
		"too long":                {4, 4 << 2, 'h', 'e', 'l', 'l', 'o'},
		"literal cut off":         {5, 4 << 2, 'h', 'e'},
		"copy offset cut off":     {9, 2 << 2, 'x', 'y', 'z', 2 | 5<<2, 3},
	}
	for name, compressed := range cases {
		if _, err := DecompressSnappy(compressed); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}

func appendSnappyLiteral(output []byte, literal []byte) []byte {
	if len(literal) == 0 {
		return output
	}
	lengthMinusOne := len(literal) - 1
	if lengthMinusOne < 60 {
		output = append(output, byte(lengthMinusOne<<2))
	} else {
		output = append(output, 62<<2, byte(lengthMinusOne), byte(lengthMinusOne>>8), byte(lengthMinusOne>>16))
	}
	return append(output, literal...)
}

func compressSnappyForTest(input []byte) []byte {
	output := binary.AppendUvarint(nil, uint64(len(input)))
	lastPositionOfFourBytes := map[uint32]int{}
	literalStart := 0
	position := 0
	for position+4 <= len(input) {
		key := binary.LittleEndian.Uint32(input[position:])
		previousPosition, found := lastPositionOfFourBytes[key]
		lastPositionOfFourBytes[key] = position
		if !found || position-previousPosition > 65535 {
			position++
			continue
		}
		output = appendSnappyLiteral(output, input[literalStart:position])
		length := 4
		for position+length < len(input) && length < 64 && input[previousPosition+length] == input[position+length] {
			length++
		}
		offset := position - previousPosition
		output = append(output, byte(2|(length-1)<<2), byte(offset), byte(offset>>8))
		position += length
		literalStart = position
	}
	return appendSnappyLiteral(output, input[literalStart:])
}

func TestSnappyRoundTripsThroughTestEncoder(t *testing.T) {
	random := rand.New(rand.NewSource(1))
	randomBytes := make([]byte, 70000)
	random.Read(randomBytes)
	inputs := [][]byte{
		[]byte(""),
		[]byte("a"),
		[]byte(strings.Repeat("the quick brown fox ", 5000)),
		randomBytes,
		append([]byte(strings.Repeat("ab", 40000)), randomBytes[:1000]...),
	}
	for index, input := range inputs {
		compressed := compressSnappyForTest(input)
		decompressed, err := DecompressSnappy(compressed)
		if err != nil {
			t.Fatalf("input %d: %v", index, err)
		}
		if !bytes.Equal(decompressed, input) {
			t.Errorf("input %d: round trip changed the bytes", index)
		}
	}
}
