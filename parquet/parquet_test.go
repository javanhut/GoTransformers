package parquet

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestReadsRealPyarrowFile(t *testing.T) {
	file, err := Open(filepath.Join("testdata", "diamonds.parquet"))
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()

	expectedNames := []string{"carat", "cut", "color", "clarity", "depth", "table", "price", "x", "y", "z", "__index_level_0__"}
	if !reflect.DeepEqual(file.ColumnNames(), expectedNames) {
		t.Errorf("column names = %v", file.ColumnNames())
	}
	if file.NumberOfRows() != 10 || file.NumberOfRowGroups() != 1 || file.NumberOfRowsInRowGroup(0) != 10 {
		t.Errorf("%d rows in %d row groups", file.NumberOfRows(), file.NumberOfRowGroups())
	}
	if !strings.HasPrefix(file.CreatedBy, "parquet-cpp") {
		t.Errorf("created by %q", file.CreatedBy)
	}

	cuts, err := file.ReadStringColumn(0, "cut")
	if err != nil {
		t.Fatal(err)
	}
	expectedCuts := []string{"Ideal", "Premium", "Good", "Premium", "Good", "Very Good", "Very Good", "Very Good", "Fair", "Very Good"}
	if !reflect.DeepEqual(cuts, expectedCuts) {
		t.Errorf("cut = %q", cuts)
	}
	clarities, err := file.ReadAllStrings("clarity")
	if err != nil {
		t.Fatal(err)
	}
	expectedClarities := []string{"SI2", "SI1", "VS1", "VS2", "SI2", "VVS2", "VVS1", "SI1", "VS2", "VS1"}
	if !reflect.DeepEqual(clarities, expectedClarities) {
		t.Errorf("clarity = %q", clarities)
	}
	for index := 0; index < len(clarities); index++ {
		if !utf8.ValidString(clarities[index]) {
			t.Errorf("clarity %d is not valid UTF-8", index)
		}
	}

	prices, err := file.ReadInt64Column(0, "price")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(prices, []int64{326, 326, 327, 334, 335, 336, 336, 337, 337, 338}) {
		t.Errorf("price = %v", prices)
	}
	depths, err := file.ReadAllFloat64s("depth")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(depths, []float64{61.5, 59.8, 56.9, 62.4, 63.3, 62.8, 62.3, 61.9, 65.1, 59.4}) {
		t.Errorf("depth = %v", depths)
	}
	pricesAsFloats, err := file.ReadFloat64Column(0, "price")
	if err != nil || pricesAsFloats[9] != 338 {
		t.Errorf("price as floats = %v, %v", pricesAsFloats, err)
	}
	isNull, err := file.ReadNullColumn(0, "z")
	if err != nil || !reflect.DeepEqual(isNull, make([]bool, 10)) {
		t.Errorf("z nulls = %v, %v", isNull, err)
	}
	columns := file.Columns()
	if columns[1] != (Column{Name: "cut", PhysicalType: "BYTE_ARRAY", Optional: true}) {
		t.Errorf("cut column = %+v", columns[1])
	}
}

func TestReadsSecondRealPyarrowFile(t *testing.T) {
	file, err := Open(filepath.Join("testdata", "timestamps.parquet"))
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	indexTexts, err := file.ReadAllStrings("index")
	if err != nil || !reflect.DeepEqual(indexTexts, []string{"a", "b", "c"}) {
		t.Errorf("index = %q, %v", indexTexts, err)
	}
	integers, err := file.ReadAllInt64s("a")
	if err != nil || !reflect.DeepEqual(integers, []int64{1, 2, 3}) {
		t.Errorf("a = %v, %v", integers, err)
	}
	floats, err := file.ReadAllFloat64s("b")
	if err != nil || !reflect.DeepEqual(floats, []float64{0.1, 0.2, 0.3}) {
		t.Errorf("b = %v, %v", floats, err)
	}
	timestamps, err := file.ReadAllInt64s("c")
	if err != nil || !reflect.DeepEqual(timestamps, []int64{1483225200000000, 1483311600000000, 1483398000000000}) {
		t.Errorf("c = %v, %v", timestamps, err)
	}
}

func makeTestColumns(numberOfRows int) []testColumn {
	words := []string{"the", "cat", "sat", "on", "a", "mat", "über", "naïve", "日本語", "🙂", ""}
	random := rand.New(rand.NewSource(int64(numberOfRows)))
	columns := []testColumn{
		{name: "text", physicalType: byteArrayType, optional: true},
		{name: "id", physicalType: int64Type},
		{name: "small", physicalType: int32Type},
		{name: "score", physicalType: doubleType, optional: true},
		{name: "ratio", physicalType: floatType},
		{name: "flag", physicalType: booleanType},
		{name: "maybe", physicalType: booleanType, optional: true},
		{name: "label", physicalType: byteArrayType},
		{name: "code", physicalType: fixedLengthByteArrayType, typeLength: 3},
	}
	for columnIndex := 0; columnIndex < len(columns); columnIndex++ {
		if columns[columnIndex].optional {
			columns[columnIndex].isNull = make([]bool, numberOfRows)
		}
	}
	for row := 0; row < numberOfRows; row++ {
		numberOfWords := 1 + random.Intn(6)
		var sentence []string
		for wordIndex := 0; wordIndex < numberOfWords; wordIndex++ {
			sentence = append(sentence, words[random.Intn(len(words))])
		}
		columns[0].texts = append(columns[0].texts, strings.Join(sentence, " "))
		columns[0].isNull[row] = row%7 == 3
		columns[1].integers = append(columns[1].integers, int64(row)*1_000_000_007)
		columns[2].integers = append(columns[2].integers, int64(random.Intn(2000)-1000))
		columns[3].floats = append(columns[3].floats, float64(row%13)*0.25)
		columns[3].isNull[row] = row%5 == 0
		columns[4].floats = append(columns[4].floats, float64(row)*0.5)
		columns[5].booleans = append(columns[5].booleans, random.Intn(3) == 0)
		columns[6].booleans = append(columns[6].booleans, row%2 == 0)
		columns[6].isNull[row] = row%3 == 0
		columns[7].texts = append(columns[7].texts, "same label")
		columns[8].texts = append(columns[8].texts, fmt.Sprintf("%03d", row%40))
	}
	return columns
}

func expectedTexts(column testColumn) []string {
	expected := make([]string, len(column.texts))
	for row := 0; row < len(column.texts); row++ {
		if !column.isNullAt(row) {
			expected[row] = column.texts[row]
		}
	}
	return expected
}

func expectedIntegers(column testColumn) []int64 {
	expected := make([]int64, len(column.integers))
	for row := 0; row < len(column.integers); row++ {
		if !column.isNullAt(row) {
			expected[row] = column.integers[row]
		}
	}
	return expected
}

func expectedFloats(column testColumn) []float64 {
	expected := make([]float64, len(column.floats))
	for row := 0; row < len(column.floats); row++ {
		if !column.isNullAt(row) {
			expected[row] = column.floats[row]
		}
	}
	return expected
}

func expectedBooleans(column testColumn) []bool {
	expected := make([]bool, len(column.booleans))
	for row := 0; row < len(column.booleans); row++ {
		if !column.isNullAt(row) {
			expected[row] = column.booleans[row]
		}
	}
	return expected
}

func checkAllColumnsReadBack(t *testing.T, path string, columns []testColumn, numberOfRows int, options testFileOptions) {
	t.Helper()
	file, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	expectedRowGroups := (numberOfRows + options.rowsPerRowGroup - 1) / options.rowsPerRowGroup
	if file.NumberOfRows() != numberOfRows || file.NumberOfRowGroups() != expectedRowGroups {
		t.Fatalf("%d rows in %d row groups, want %d in %d", file.NumberOfRows(), file.NumberOfRowGroups(), numberOfRows, expectedRowGroups)
	}
	for columnIndex := 0; columnIndex < len(columns); columnIndex++ {
		column := columns[columnIndex]
		var got any
		var want any
		switch column.physicalType {
		case byteArrayType, fixedLengthByteArrayType:
			got, err = file.ReadAllStrings(column.name)
			want = expectedTexts(column)
		case int32Type, int64Type:
			got, err = file.ReadAllInt64s(column.name)
			want = expectedIntegers(column)
		case floatType, doubleType:
			got, err = file.ReadAllFloat64s(column.name)
			want = expectedFloats(column)
		case booleanType:
			var allBooleans []bool
			for rowGroupIndex := 0; rowGroupIndex < file.NumberOfRowGroups(); rowGroupIndex++ {
				booleans, readErr := file.ReadBoolColumn(rowGroupIndex, column.name)
				if readErr != nil {
					err = readErr
					break
				}
				allBooleans = append(allBooleans, booleans...)
			}
			got = allBooleans
			want = expectedBooleans(column)
		}
		if err != nil {
			t.Errorf("column %q: %v", column.name, err)
			continue
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("column %q read back differently", column.name)
		}
	}
	var allNulls []bool
	for rowGroupIndex := 0; rowGroupIndex < file.NumberOfRowGroups(); rowGroupIndex++ {
		isNull, err := file.ReadNullColumn(rowGroupIndex, "text")
		if err != nil {
			t.Fatal(err)
		}
		allNulls = append(allNulls, isNull...)
	}
	if !reflect.DeepEqual(allNulls, columns[0].isNull) {
		t.Errorf("text nulls read back differently")
	}
}

func TestReadsWrittenFilesWithEveryCodecEncodingAndPageVersion(t *testing.T) {
	numberOfRows := 1000
	columns := makeTestColumns(numberOfRows)
	codecs := []int{uncompressedCodec, snappyCodec, gzipCodec, zstdCodec}
	encodingChoices := []string{"plain", "dictionary", "dictionary then plain"}
	for _, codec := range codecs {
		for _, encodingChoice := range encodingChoices {
			for _, useDataPageV2 := range []bool{false, true} {
				name := fmt.Sprintf("%s/%s/v2=%v", nameFromList(codecNames, codec), encodingChoice, useDataPageV2)
				t.Run(name, func(t *testing.T) {
					options := testFileOptions{
						codec:               codec,
						useDictionary:       encodingChoice != "plain",
						plainAfterFirstPage: encodingChoice == "dictionary then plain",
						useDataPageV2:       useDataPageV2,
						rowsPerPage:         128,
						rowsPerRowGroup:     400,
					}
					path := writeTestParquetFile(t, columns, numberOfRows, options)
					checkAllColumnsReadBack(t, path, columns, numberOfRows, options)
				})
			}
		}
	}
}

func TestReadsManyRowGroupsOneAtATime(t *testing.T) {
	numberOfRows := 200
	columns := makeTestColumns(numberOfRows)
	options := testFileOptions{codec: snappyCodec, useDictionary: true, rowsPerPage: 3, rowsPerRowGroup: 10}
	path := writeTestParquetFile(t, columns, numberOfRows, options)
	checkAllColumnsReadBack(t, path, columns, numberOfRows, options)

	file, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	if file.NumberOfRowGroups() != 20 {
		t.Fatalf("%d row groups", file.NumberOfRowGroups())
	}
	rowsSeen := 0
	err = file.ForEachRowGroupOfStrings("text", func(rowGroupIndex int, texts []string) error {
		if len(texts) != file.NumberOfRowsInRowGroup(rowGroupIndex) {
			return fmt.Errorf("row group %d gave %d texts", rowGroupIndex, len(texts))
		}
		rowsSeen += len(texts)
		return nil
	})
	if err != nil || rowsSeen != numberOfRows {
		t.Errorf("saw %d rows, %v", rowsSeen, err)
	}
}

func TestReadsLargeZstdPages(t *testing.T) {
	numberOfRows := 3000
	columns := []testColumn{{name: "text", physicalType: byteArrayType, optional: true, isNull: make([]bool, numberOfRows)}}
	random := rand.New(rand.NewSource(5))
	for row := 0; row < numberOfRows; row++ {
		var builder strings.Builder
		for wordIndex := 0; wordIndex < 40; wordIndex++ {
			fmt.Fprintf(&builder, "word%d ", random.Intn(500))
		}
		columns[0].texts = append(columns[0].texts, builder.String())
	}
	options := testFileOptions{codec: zstdCodec, rowsPerPage: numberOfRows, rowsPerRowGroup: numberOfRows}
	path := writeTestParquetFile(t, columns, numberOfRows, options)
	file, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	texts, err := file.ReadAllStrings("text")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(texts, columns[0].texts) {
		t.Errorf("large zstd page read back differently")
	}
}

func TestErrorsAreClear(t *testing.T) {
	numberOfRows := 20
	columns := makeTestColumns(numberOfRows)
	columns = append(columns, testColumn{name: "tags", physicalType: byteArrayType, repeated: true, texts: columns[0].texts})
	options := testFileOptions{codec: uncompressedCodec, rowsPerPage: 10, rowsPerRowGroup: 20}
	path := writeTestParquetFile(t, columns, numberOfRows, options)
	file, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()

	expectations := []struct {
		read         func() error
		errorMention string
	}{
		{func() error { _, err := file.ReadStringColumn(0, "missing"); return err }, "has no column \"missing\""},
		{func() error { _, err := file.ReadStringColumn(0, "id"); return err }, "holds integer values"},
		{func() error { _, err := file.ReadInt64Column(0, "text"); return err }, "holds text values"},
		{func() error { _, err := file.ReadStringColumn(0, "tags"); return err }, "repeated (list) column"},
		{func() error { _, err := file.ReadStringColumn(5, "text"); return err }, "no row group 5"},
		{func() error { _, err := file.ReadBoolColumn(0, "score"); return err }, "not boolean"},
	}
	for _, expectation := range expectations {
		err := expectation.read()
		if err == nil || !strings.Contains(err.Error(), expectation.errorMention) {
			t.Errorf("got error %v, want one mentioning %q", err, expectation.errorMention)
		}
	}
	if !file.Columns()[9].Repeated {
		t.Errorf("tags column isn't marked repeated")
	}
}

func TestRejectsFilesThatAreNotParquet(t *testing.T) {
	directory := t.TempDir()
	cases := map[string][]byte{
		"empty.parquet":     {},
		"text.parquet":      []byte("this is just some text, not parquet at all"),
		"encrypted.parquet": []byte("PAR1\x00\x00\x00\x00\x04\x00\x00\x00PARE"),
		"bigfooter.parquet": []byte("PAR1\x00\x00\x00\x00\xff\xff\x00\x00PAR1"),
	}
	for name, contents := range cases {
		path := filepath.Join(directory, name)
		if err := os.WriteFile(path, contents, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := Open(path); err == nil {
			t.Errorf("%s: opened without error", name)
		}
	}
	if _, err := Open(filepath.Join(directory, "does-not-exist.parquet")); err == nil {
		t.Error("opening a missing file did not fail")
	}
}

func readEverythingWithoutPanicking(t *testing.T, path string, description string) {
	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("%s: panic %v", description, recovered)
		}
	}()
	file, err := Open(path)
	if err != nil {
		return
	}
	defer file.Close()
	columns := file.Columns()
	for rowGroupIndex := 0; rowGroupIndex < file.NumberOfRowGroups(); rowGroupIndex++ {
		for columnIndex := 0; columnIndex < len(columns); columnIndex++ {
			file.ReadStringColumn(rowGroupIndex, columns[columnIndex].Name)
			file.ReadFloat64Column(rowGroupIndex, columns[columnIndex].Name)
			file.ReadBoolColumn(rowGroupIndex, columns[columnIndex].Name)
		}
	}
}

func TestNeverPanicsOnCorruptedFiles(t *testing.T) {
	numberOfRows := 300
	columns := makeTestColumns(numberOfRows)
	writtenPath := writeTestParquetFile(t, columns, numberOfRows, testFileOptions{codec: snappyCodec, useDictionary: true, rowsPerPage: 50, rowsPerRowGroup: 150})
	originals := []string{writtenPath, filepath.Join("testdata", "diamonds.parquet")}
	random := rand.New(rand.NewSource(11))
	corruptedPath := filepath.Join(t.TempDir(), "corrupted.parquet")
	for _, originalPath := range originals {
		original, err := os.ReadFile(originalPath)
		if err != nil {
			t.Fatal(err)
		}
		for attempt := 0; attempt < 400; attempt++ {
			corrupted := append([]byte{}, original...)
			numberOfFlips := 1 + random.Intn(3)
			for flip := 0; flip < numberOfFlips; flip++ {
				corrupted[4+random.Intn(len(corrupted)-12)] ^= byte(1 + random.Intn(255))
			}
			if err := os.WriteFile(corruptedPath, corrupted, 0o644); err != nil {
				t.Fatal(err)
			}
			readEverythingWithoutPanicking(t, corruptedPath, fmt.Sprintf("%s attempt %d", originalPath, attempt))
		}
	}
}
