package datafile

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"
)

func readAllTexts(t *testing.T, source TextSource) []string {
	t.Helper()
	var texts []string
	for {
		text, err := source.NextText()
		if errors.Is(err, io.EOF) {
			return texts
		}
		if err != nil {
			t.Fatal(err)
		}
		texts = append(texts, text)
	}
}

func TestPlainTextPiecesAddUpToTheWholeFile(t *testing.T) {
	var contents strings.Builder
	for lineNumber := 0; lineNumber < 3000; lineNumber++ {
		fmt.Fprintf(&contents, "line %d says héllo wörld\n", lineNumber)
	}
	contents.WriteString("no newline at the end")
	path := writeFile(t, "book.txt", contents.String())

	source, err := OpenPlainTextSource(path, PlainTextSourceOptions{Splitting: SplitIntoPieces, PieceSizeInBytes: 1000})
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	pieces := readAllTexts(t, source)
	if strings.Join(pieces, "") != contents.String() {
		t.Fatal("the pieces don't add up to the whole file")
	}
	for index := 0; index < len(pieces)-1; index++ {
		if !strings.HasSuffix(pieces[index], "\n") || len(pieces[index]) < 1000 {
			t.Fatalf("piece %d is %d bytes and ends %q", index, len(pieces[index]), pieces[index][len(pieces[index])-5:])
		}
	}
}

func TestPlainTextPiecesOfOneHugeLineStayValidText(t *testing.T) {
	contents := strings.Repeat("ab€", 100000)
	path := writeFile(t, "oneline.txt", contents)
	source, err := OpenPlainTextSource(path, PlainTextSourceOptions{PieceSizeInBytes: 1001})
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	pieces := readAllTexts(t, source)
	if strings.Join(pieces, "") != contents {
		t.Fatal("the pieces don't add up to the whole file")
	}
	for index := 0; index < len(pieces); index++ {
		if !utf8.ValidString(pieces[index]) {
			t.Fatalf("piece %d cuts a character in half", index)
		}
	}
	if len(pieces) < 2 {
		t.Errorf("a long line without newlines should still be cut, got %d pieces", len(pieces))
	}
}

func TestPlainTextSplitOnBlankLines(t *testing.T) {
	path := writeFile(t, "stories.txt", "\n\nfirst story\nsecond line\n\n\n  \nsecond story\r\n\r\nthird story")
	source, err := OpenPlainTextSource(path, PlainTextSourceOptions{Splitting: SplitOnBlankLines})
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	documents := readAllTexts(t, source)
	want := []string{"first story\nsecond line", "second story", "third story"}
	if !reflect.DeepEqual(documents, want) {
		t.Errorf("documents = %q", documents)
	}

	emptyPath := writeFile(t, "empty.txt", "")
	emptySource, _ := OpenPlainTextSource(emptyPath, PlainTextSourceOptions{Splitting: SplitOnBlankLines})
	defer emptySource.Close()
	if len(readAllTexts(t, emptySource)) != 0 {
		t.Error("an empty file should have no documents")
	}
}

func TestJSONLTextSource(t *testing.T) {
	path := writeFile(t, "texts.jsonl", "{\"text\": \"first\"}\n\n{\"text\": \"second\", \"id\": 2}\r\n{\"text\": \"third\"}")
	source, err := OpenJSONLTextSource(path, "text")
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	if texts := readAllTexts(t, source); !reflect.DeepEqual(texts, []string{"first", "second", "third"}) {
		t.Errorf("texts = %q", texts)
	}

	badPath := writeFile(t, "bad.jsonl", "{\"text\": \"fine\"}\n{\"body\": \"x\"}\n")
	badSource, _ := OpenJSONLTextSource(badPath, "text")
	defer badSource.Close()
	badSource.NextText()
	if _, err := badSource.NextText(); err == nil || !strings.Contains(err.Error(), "line 2") {
		t.Errorf("expected an error naming line 2, got %v", err)
	}
}

func TestJSONLRecordSourceTurnsChatsIntoText(t *testing.T) {
	path := writeFile(t, "chats.jsonl", `{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}`+"\n")
	conversationToText := func(record JSONRecord) (string, error) {
		messages, isList := record["messages"].([]any)
		if !isList {
			return "", errors.New("no messages")
		}
		var text strings.Builder
		for _, message := range messages {
			fields := JSONRecord(message.(map[string]any))
			role, _ := fields.Text("role")
			content, _ := fields.Text("content")
			text.WriteString(role + ": " + content + "\n")
		}
		return text.String(), nil
	}
	source, err := OpenJSONLRecordSource(path, conversationToText)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	if texts := readAllTexts(t, source); !reflect.DeepEqual(texts, []string{"user: hi\nassistant: hello\n"}) {
		t.Errorf("texts = %q", texts)
	}
}

func TestTrainingTextSourceReadsManyFilesInOrder(t *testing.T) {
	folder := t.TempDir()
	os.WriteFile(filepath.Join(folder, "part2.txt"), []byte("second file\n"), 0o644)
	os.WriteFile(filepath.Join(folder, "part1.txt"), []byte("first file\n"), 0o644)
	os.WriteFile(filepath.Join(folder, "part3.jsonl"), []byte("{\"body\": \"a\"}\n{\"body\": \"b\"}\n"), 0o644)

	source, err := OpenTrainingTextSource([]string{filepath.Join(folder, "part*.txt"), filepath.Join(folder, "*.jsonl")}, "body")
	if err != nil {
		t.Fatal(err)
	}
	texts := readAllTexts(t, source)
	want := []string{"first file\n", "second file\n", "a\n\n", "b\n\n"}
	if !reflect.DeepEqual(texts, want) {
		t.Errorf("texts = %q", texts)
	}
	if _, err := source.NextText(); !errors.Is(err, io.EOF) {
		t.Errorf("after the last file the source should stay at the end, got %v", err)
	}

	if _, err := OpenTrainingTextSource([]string{filepath.Join(folder, "*.parquet")}, "text"); err == nil {
		t.Error("a pattern that matches nothing should be an error")
	}
}

func TestReadTextSample(t *testing.T) {
	sample, err := ReadTextSample(NewSliceTextSource([]string{"abc", "def", "ghi"}), 5)
	if err != nil || sample != "abcdef" {
		t.Errorf("sample = %q, %v", sample, err)
	}
}
