package compression

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

type zstdVector struct {
	fileName       string
	expectedSHA256 string
	expectedSize   int
}

func readZstdVectors(t *testing.T) []zstdVector {
	t.Helper()
	manifest, err := os.ReadFile(filepath.Join("testdata", "zstdvectors.txt"))
	if err != nil {
		t.Fatal(err)
	}
	var vectors []zstdVector
	lines := strings.Split(strings.TrimSpace(string(manifest)), "\n")
	for _, line := range lines {
		fields := strings.Fields(line)
		if len(fields) != 3 {
			t.Fatalf("bad manifest line %q", line)
		}
		size, err := strconv.Atoi(fields[2])
		if err != nil {
			t.Fatal(err)
		}
		vectors = append(vectors, zstdVector{fileName: fields[0], expectedSHA256: fields[1], expectedSize: size})
	}
	return vectors
}

func readTestFile(t *testing.T, name string) []byte {
	t.Helper()
	contents, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return contents
}

func sha256Text(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func TestZstdDecodesCommandLineVectors(t *testing.T) {
	vectors := readZstdVectors(t)
	if len(vectors) < 40 {
		t.Fatalf("only %d vectors in the manifest", len(vectors))
	}
	for _, vector := range vectors {
		compressed := readTestFile(t, vector.fileName)
		decompressed, err := DecompressZstd(compressed)
		if err != nil {
			t.Errorf("%s: %v", vector.fileName, err)
			continue
		}
		if len(decompressed) != vector.expectedSize || sha256Text(decompressed) != vector.expectedSHA256 {
			t.Errorf("%s: decoded %d bytes with sha256 %s, want %d bytes with %s",
				vector.fileName, len(decompressed), sha256Text(decompressed), vector.expectedSize, vector.expectedSHA256)
		}
	}
}

func TestZstdSmallTextMatchesOriginalExactly(t *testing.T) {
	original := readTestFile(t, "smalltext.txt")
	levels := []string{"level1", "level3", "level9", "level19", "ultra22", "fast3", "nocheck", "streamed"}
	for _, level := range levels {
		decompressed, err := DecompressZstd(readTestFile(t, "smalltext."+level+".zst"))
		if err != nil {
			t.Fatalf("%s: %v", level, err)
		}
		if !bytes.Equal(decompressed, original) {
			t.Errorf("%s: decoded text differs from the original", level)
		}
	}
}

func TestZstdConcatenatedAndSkippableFrames(t *testing.T) {
	original := readTestFile(t, "smalltext.txt")
	firstFrame := readTestFile(t, "smalltext.level3.zst")
	secondFrame := readTestFile(t, "onebyte.level19.zst")
	skippableFrame := []byte{0x5a, 0x2a, 0x4d, 0x18}
	skippableFrame = binary.LittleEndian.AppendUint32(skippableFrame, 5)
	skippableFrame = append(skippableFrame, "hello"...)

	var joined []byte
	joined = append(joined, firstFrame...)
	joined = append(joined, skippableFrame...)
	joined = append(joined, secondFrame...)
	joined = append(joined, firstFrame...)
	decompressed, err := DecompressZstd(joined)
	if err != nil {
		t.Fatal(err)
	}
	expected := string(original) + "a" + string(original)
	if string(decompressed) != expected {
		t.Errorf("decoded %d bytes, want %d", len(decompressed), len(expected))
	}
}

func TestZstdRejectsBrokenInput(t *testing.T) {
	compressed := readTestFile(t, "smalltext.level19.zst")
	cases := map[string][]byte{
		"empty":        {},
		"wrong magic":  {1, 2, 3, 4, 5, 6, 7, 8},
		"cut off":      compressed[:len(compressed)/2],
		"stray bytes":  append(append([]byte{}, compressed...), 1, 2),
		"bad checksum": append(append([]byte{}, compressed[:len(compressed)-1]...), compressed[len(compressed)-1]^0xff),
	}
	for name, input := range cases {
		if _, err := DecompressZstd(input); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}

func TestZstdNeverPanicsOnCorruptedInput(t *testing.T) {
	random := rand.New(rand.NewSource(7))
	names := []string{"smalltext.level19.zst", "mixed.level3.zst", "repetitive.fast3.zst", "random.level1.zst", "largetext.level1.zst"}
	for _, name := range names {
		original := readTestFile(t, name)
		for attempt := 0; attempt < 300; attempt++ {
			corrupted := append([]byte{}, original...)
			numberOfFlips := 1 + random.Intn(4)
			for flip := 0; flip < numberOfFlips; flip++ {
				position := random.Intn(len(corrupted))
				corrupted[position] ^= byte(1 + random.Intn(255))
			}
			if attempt%10 == 0 {
				corrupted = corrupted[:random.Intn(len(corrupted))]
			}
			func() {
				defer func() {
					if recovered := recover(); recovered != nil {
						t.Fatalf("%s attempt %d: panic %v", name, attempt, recovered)
					}
				}()
				DecompressZstd(corrupted)
			}()
		}
	}
}

func TestXXHash64KnownValues(t *testing.T) {
	cases := map[string]uint64{
		"":    0xEF46DB3751D8E999,
		"abc": 0x44BC2CF5AD770999,
	}
	for input, expected := range cases {
		if hash := xxhash64([]byte(input)); hash != expected {
			t.Errorf("xxhash64(%q) = %#x, want %#x", input, hash, expected)
		}
	}
}

func BenchmarkZstdLargeText(b *testing.B) {
	compressed, err := os.ReadFile(filepath.Join("testdata", "largetext.level3.zst"))
	if err != nil {
		b.Fatal(err)
	}
	for iteration := 0; iteration < b.N; iteration++ {
		decompressed, err := DecompressZstd(compressed)
		if err != nil {
			b.Fatal(err)
		}
		b.SetBytes(int64(len(decompressed)))
	}
}

func TestZstdNumberOfSequencesHeaderSizes(t *testing.T) {
	cases := []struct {
		header            []byte
		expectedCount     int
		expectedBytesUsed int
	}{
		{[]byte{0}, 0, 1},
		{[]byte{127}, 127, 1},
		{[]byte{128, 200}, 200, 2},
		{[]byte{254, 255}, 0x7EFF, 2},
		{[]byte{255, 0, 0}, 0x7F00, 3},
		{[]byte{255, 0x34, 0x12}, 0x1234 + 0x7F00, 3},
	}
	for _, testCase := range cases {
		count, bytesUsed, err := readNumberOfSequences(testCase.header)
		if err != nil || count != testCase.expectedCount || bytesUsed != testCase.expectedBytesUsed {
			t.Errorf("%v: got %d sequences from %d bytes (%v), want %d from %d", testCase.header, count, bytesUsed, err, testCase.expectedCount, testCase.expectedBytesUsed)
		}
	}
}

func TestZstdHandBuiltFrameWithRepeatedByteLiterals(t *testing.T) {
	frame := []byte{0x28, 0xb5, 0x2f, 0xfd, 0x20, 5}
	blockContent := []byte{1 | 5<<3, 'q', 0}
	blockHeader := 1 | compressedBlock<<1 | len(blockContent)<<3
	frame = append(frame, byte(blockHeader), byte(blockHeader>>8), byte(blockHeader>>16))
	frame = append(frame, blockContent...)
	decompressed, err := DecompressZstd(frame)
	if err != nil || string(decompressed) != "qqqqq" {
		t.Errorf("got %q, %v", decompressed, err)
	}
}
