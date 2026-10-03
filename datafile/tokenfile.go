package datafile

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"os"
	"time"
)

const tokenFileMagic = "GOTOKENS"
const tokenFileVersion = 1
const tokenFileHeaderSize = 64
const largestVocabularyForTwoBytesPerToken = 1 << 16

type tokenFileHeader struct {
	bytesPerToken      int
	vocabularySize     int
	numberOfTokens     int64
	numberOfDocuments  int64
	documentEndsOffset int64
}

func encodeTokenFileHeader(header tokenFileHeader) []byte {
	headerBytes := make([]byte, tokenFileHeaderSize)
	copy(headerBytes[0:8], tokenFileMagic)
	binary.LittleEndian.PutUint32(headerBytes[8:12], tokenFileVersion)
	binary.LittleEndian.PutUint32(headerBytes[12:16], uint32(header.bytesPerToken))
	binary.LittleEndian.PutUint64(headerBytes[16:24], uint64(header.vocabularySize))
	binary.LittleEndian.PutUint64(headerBytes[24:32], uint64(header.numberOfTokens))
	binary.LittleEndian.PutUint64(headerBytes[32:40], uint64(header.numberOfDocuments))
	binary.LittleEndian.PutUint64(headerBytes[40:48], uint64(header.documentEndsOffset))
	return headerBytes
}

func decodeTokenFileHeader(headerBytes []byte) (tokenFileHeader, error) {
	if string(headerBytes[0:8]) != tokenFileMagic {
		return tokenFileHeader{}, fmt.Errorf("not a token file (it doesn't start with %q)", tokenFileMagic)
	}
	version := binary.LittleEndian.Uint32(headerBytes[8:12])
	if version != tokenFileVersion {
		return tokenFileHeader{}, fmt.Errorf("token file version %d, but only version %d can be read", version, tokenFileVersion)
	}
	header := tokenFileHeader{
		bytesPerToken:      int(binary.LittleEndian.Uint32(headerBytes[12:16])),
		vocabularySize:     int(binary.LittleEndian.Uint64(headerBytes[16:24])),
		numberOfTokens:     int64(binary.LittleEndian.Uint64(headerBytes[24:32])),
		numberOfDocuments:  int64(binary.LittleEndian.Uint64(headerBytes[32:40])),
		documentEndsOffset: int64(binary.LittleEndian.Uint64(headerBytes[40:48])),
	}
	if header.bytesPerToken != 2 && header.bytesPerToken != 4 {
		return tokenFileHeader{}, fmt.Errorf("token file says %d bytes per token, expected 2 or 4", header.bytesPerToken)
	}
	if header.numberOfTokens < 0 || header.numberOfDocuments < 0 {
		return tokenFileHeader{}, errors.New("token file header has a negative count")
	}
	return header, nil
}

func BytesPerTokenForVocabulary(vocabularySize int) int {
	if vocabularySize <= largestVocabularyForTwoBytesPerToken {
		return 2
	}
	return 4
}

func putToken(destination []byte, tokenID int, bytesPerToken int) {
	if bytesPerToken == 2 {
		binary.LittleEndian.PutUint16(destination, uint16(tokenID))
		return
	}
	binary.LittleEndian.PutUint32(destination, uint32(tokenID))
}

func getToken(source []byte, bytesPerToken int) int {
	if bytesPerToken == 2 {
		return int(binary.LittleEndian.Uint16(source))
	}
	return int(binary.LittleEndian.Uint32(source))
}

type TokenFileWriter struct {
	finalPath      string
	partialPath    string
	file           *os.File
	bufferedWriter *bufio.Writer
	bytesPerToken  int
	vocabularySize int
	numberOfTokens int64
	documentEnds   []int64
	tokenBytes     []byte
	finished       bool
}

func CreateTokenFile(path string, vocabularySize int) (*TokenFileWriter, error) {
	if vocabularySize <= 0 {
		return nil, fmt.Errorf("vocabulary size must be positive, got %d", vocabularySize)
	}
	partialPath := path + ".partial"
	file, err := os.Create(partialPath)
	if err != nil {
		return nil, err
	}
	writer := &TokenFileWriter{
		finalPath:      path,
		partialPath:    partialPath,
		file:           file,
		bufferedWriter: bufio.NewWriterSize(file, 1<<20),
		bytesPerToken:  BytesPerTokenForVocabulary(vocabularySize),
		vocabularySize: vocabularySize,
	}
	if _, err := writer.bufferedWriter.Write(make([]byte, tokenFileHeaderSize)); err != nil {
		writer.Discard()
		return nil, err
	}
	return writer, nil
}

func (writer *TokenFileWriter) BytesPerToken() int {
	return writer.bytesPerToken
}

func (writer *TokenFileWriter) NumberOfTokens() int64 {
	return writer.numberOfTokens
}

func (writer *TokenFileWriter) NumberOfDocuments() int {
	return len(writer.documentEnds)
}

func (writer *TokenFileWriter) WriteTokens(tokenIDs []int) error {
	if writer.finished {
		return errors.New("the token file is already closed")
	}
	neededBytes := len(tokenIDs) * writer.bytesPerToken
	if cap(writer.tokenBytes) < neededBytes {
		writer.tokenBytes = make([]byte, neededBytes)
	}
	tokenBytes := writer.tokenBytes[:neededBytes]
	for index := 0; index < len(tokenIDs); index++ {
		tokenID := tokenIDs[index]
		if tokenID < 0 || tokenID >= writer.vocabularySize {
			return fmt.Errorf("token ID %d is outside the vocabulary of %d tokens", tokenID, writer.vocabularySize)
		}
		putToken(tokenBytes[index*writer.bytesPerToken:], tokenID, writer.bytesPerToken)
	}
	if _, err := writer.bufferedWriter.Write(tokenBytes); err != nil {
		return err
	}
	writer.numberOfTokens += int64(len(tokenIDs))
	return nil
}

func (writer *TokenFileWriter) EndDocument() {
	numberOfDocuments := len(writer.documentEnds)
	if numberOfDocuments > 0 && writer.documentEnds[numberOfDocuments-1] == writer.numberOfTokens {
		return
	}
	writer.documentEnds = append(writer.documentEnds, writer.numberOfTokens)
}

func (writer *TokenFileWriter) writeDocumentEnds() error {
	endBytes := make([]byte, 8)
	for index := 0; index < len(writer.documentEnds); index++ {
		binary.LittleEndian.PutUint64(endBytes, uint64(writer.documentEnds[index]))
		if _, err := writer.bufferedWriter.Write(endBytes); err != nil {
			return err
		}
	}
	return nil
}

func (writer *TokenFileWriter) Close() error {
	if writer.finished {
		return nil
	}
	header := tokenFileHeader{
		bytesPerToken:     writer.bytesPerToken,
		vocabularySize:    writer.vocabularySize,
		numberOfTokens:    writer.numberOfTokens,
		numberOfDocuments: int64(len(writer.documentEnds)),
	}
	if len(writer.documentEnds) > 0 {
		header.documentEndsOffset = tokenFileHeaderSize + writer.numberOfTokens*int64(writer.bytesPerToken)
	}
	err := writer.writeDocumentEnds()
	if err == nil {
		err = writer.bufferedWriter.Flush()
	}
	if err == nil {
		_, err = writer.file.WriteAt(encodeTokenFileHeader(header), 0)
	}
	if err == nil {
		err = writer.file.Sync()
	}
	if err != nil {
		writer.Discard()
		return err
	}
	writer.finished = true
	if err := writer.file.Close(); err != nil {
		os.Remove(writer.partialPath)
		return err
	}
	return os.Rename(writer.partialPath, writer.finalPath)
}

func (writer *TokenFileWriter) Discard() error {
	if writer.finished {
		return nil
	}
	writer.finished = true
	writer.file.Close()
	return os.Remove(writer.partialPath)
}

func WriteTokenFile(path string, tokenIDs []int, vocabularySize int) error {
	writer, err := CreateTokenFile(path, vocabularySize)
	if err != nil {
		return err
	}
	if err := writer.WriteTokens(tokenIDs); err != nil {
		writer.Discard()
		return err
	}
	return writer.Close()
}

type TokenFile struct {
	path               string
	file               *os.File
	bytesPerToken      int
	vocabularySize     int
	firstToken         int64
	numberOfTokens     int64
	numberOfDocuments  int64
	documentEndsOffset int64
	randomNumbers      *rand.Rand
	readBuffer         []byte
}

func OpenTokenFile(path string) (*TokenFile, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	headerBytes := make([]byte, tokenFileHeaderSize)
	if _, err := file.ReadAt(headerBytes, 0); err != nil {
		file.Close()
		return nil, fmt.Errorf("%s: could not read the token file header: %w", path, err)
	}
	header, err := decodeTokenFileHeader(headerBytes)
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	information, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, err
	}
	expectedSize := tokenFileHeaderSize + header.numberOfTokens*int64(header.bytesPerToken) + header.numberOfDocuments*8
	if information.Size() < expectedSize {
		file.Close()
		return nil, fmt.Errorf("%s: the file is %d bytes but its header needs %d, it may be cut short", path, information.Size(), expectedSize)
	}
	tokenFile := &TokenFile{
		path:               path,
		file:               file,
		bytesPerToken:      header.bytesPerToken,
		vocabularySize:     header.vocabularySize,
		firstToken:         0,
		numberOfTokens:     header.numberOfTokens,
		numberOfDocuments:  header.numberOfDocuments,
		documentEndsOffset: header.documentEndsOffset,
	}
	tokenFile.SetRandomSeed(uint64(time.Now().UnixNano()))
	return tokenFile, nil
}

func (tokenFile *TokenFile) Close() error {
	return tokenFile.file.Close()
}

func (tokenFile *TokenFile) Path() string {
	return tokenFile.path
}

func (tokenFile *TokenFile) SetRandomSeed(seed uint64) {
	tokenFile.randomNumbers = rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
}

func (tokenFile *TokenFile) NumberOfTokens() int {
	return int(tokenFile.numberOfTokens)
}

func (tokenFile *TokenFile) BytesPerToken() int {
	return tokenFile.bytesPerToken
}

func (tokenFile *TokenFile) VocabularySize() int {
	return tokenFile.vocabularySize
}

func (tokenFile *TokenFile) ReadTokens(start int, count int) ([]int, error) {
	if count < 0 {
		return nil, fmt.Errorf("can't read %d tokens", count)
	}
	tokenIDs := make([]int, count)
	if err := tokenFile.ReadTokensInto(start, tokenIDs); err != nil {
		return nil, err
	}
	return tokenIDs, nil
}

func (tokenFile *TokenFile) ReadTokensInto(start int, tokenIDs []int) error {
	count := len(tokenIDs)
	if start < 0 || int64(start)+int64(count) > tokenFile.numberOfTokens {
		return fmt.Errorf("asked for tokens %d to %d, but there are only %d", start, start+count, tokenFile.numberOfTokens)
	}
	if count == 0 {
		return nil
	}
	neededBytes := count * tokenFile.bytesPerToken
	if cap(tokenFile.readBuffer) < neededBytes {
		tokenFile.readBuffer = make([]byte, neededBytes)
	}
	tokenBytes := tokenFile.readBuffer[:neededBytes]
	byteOffset := tokenFileHeaderSize + (tokenFile.firstToken+int64(start))*int64(tokenFile.bytesPerToken)
	if _, err := tokenFile.file.ReadAt(tokenBytes, byteOffset); err != nil {
		return fmt.Errorf("%s: reading tokens %d to %d: %w", tokenFile.path, start, start+count, err)
	}
	for index := 0; index < count; index++ {
		tokenIDs[index] = getToken(tokenBytes[index*tokenFile.bytesPerToken:], tokenFile.bytesPerToken)
	}
	return nil
}

func (tokenFile *TokenFile) checkChunkLength(chunkLength int) error {
	if chunkLength <= 0 {
		return fmt.Errorf("chunk length must be positive, got %d", chunkLength)
	}
	if int64(chunkLength) > tokenFile.numberOfTokens {
		return fmt.Errorf("chunk length %d is longer than the %d tokens available", chunkLength, tokenFile.numberOfTokens)
	}
	return nil
}

func (tokenFile *TokenFile) RandomChunk(chunkLength int) ([]int, error) {
	if err := tokenFile.checkChunkLength(chunkLength); err != nil {
		return nil, err
	}
	numberOfPossibleStarts := tokenFile.numberOfTokens - int64(chunkLength) + 1
	start := tokenFile.randomNumbers.Int64N(numberOfPossibleStarts)
	return tokenFile.ReadTokens(int(start), chunkLength)
}

func (tokenFile *TokenFile) RandomChunks(chunkLength int, numberOfChunks int) ([][]int, error) {
	chunks := make([][]int, numberOfChunks)
	for index := 0; index < numberOfChunks; index++ {
		chunk, err := tokenFile.RandomChunk(chunkLength)
		if err != nil {
			return nil, err
		}
		chunks[index] = chunk
	}
	return chunks, nil
}

func (tokenFile *TokenFile) View(start int, count int) (*TokenFile, error) {
	if start < 0 || count < 0 || int64(start)+int64(count) > tokenFile.numberOfTokens {
		return nil, fmt.Errorf("a view of tokens %d to %d doesn't fit in %d tokens", start, start+count, tokenFile.numberOfTokens)
	}
	view := &TokenFile{
		path:               tokenFile.path,
		file:               tokenFile.file,
		bytesPerToken:      tokenFile.bytesPerToken,
		vocabularySize:     tokenFile.vocabularySize,
		firstToken:         tokenFile.firstToken + int64(start),
		numberOfTokens:     int64(count),
		numberOfDocuments:  tokenFile.numberOfDocuments,
		documentEndsOffset: tokenFile.documentEndsOffset,
	}
	view.SetRandomSeed(tokenFile.randomNumbers.Uint64())
	return view, nil
}

func (tokenFile *TokenFile) SplitLast(numberOfEvaluationTokens int) (*TokenFile, *TokenFile, error) {
	if numberOfEvaluationTokens < 0 || int64(numberOfEvaluationTokens) > tokenFile.numberOfTokens {
		return nil, nil, fmt.Errorf("can't hold back %d of %d tokens for evaluation", numberOfEvaluationTokens, tokenFile.numberOfTokens)
	}
	numberOfTrainingTokens := tokenFile.NumberOfTokens() - numberOfEvaluationTokens
	trainingPart, err := tokenFile.View(0, numberOfTrainingTokens)
	if err != nil {
		return nil, nil, err
	}
	evaluationPart, err := tokenFile.View(numberOfTrainingTokens, numberOfEvaluationTokens)
	if err != nil {
		return nil, nil, err
	}
	return trainingPart, evaluationPart, nil
}

func (tokenFile *TokenFile) Split(evaluationFraction float64) (*TokenFile, *TokenFile, error) {
	if evaluationFraction < 0 || evaluationFraction > 1 {
		return nil, nil, fmt.Errorf("evaluation fraction must be between 0 and 1, got %v", evaluationFraction)
	}
	numberOfEvaluationTokens := int(float64(tokenFile.numberOfTokens) * evaluationFraction)
	return tokenFile.SplitLast(numberOfEvaluationTokens)
}

func (tokenFile *TokenFile) NumberOfDocumentsInWholeFile() int {
	return int(tokenFile.numberOfDocuments)
}

func (tokenFile *TokenFile) ReadDocumentEnds() ([]int, error) {
	documentEnds := []int{}
	if tokenFile.numberOfDocuments == 0 {
		return documentEnds, nil
	}
	endBytes := make([]byte, tokenFile.numberOfDocuments*8)
	if _, err := tokenFile.file.ReadAt(endBytes, tokenFile.documentEndsOffset); err != nil {
		return nil, fmt.Errorf("%s: reading document ends: %w", tokenFile.path, err)
	}
	lastTokenOfView := tokenFile.firstToken + tokenFile.numberOfTokens
	for index := 0; index < int(tokenFile.numberOfDocuments); index++ {
		endInWholeFile := int64(binary.LittleEndian.Uint64(endBytes[index*8:]))
		if endInWholeFile > tokenFile.firstToken && endInWholeFile <= lastTokenOfView {
			documentEnds = append(documentEnds, int(endInWholeFile-tokenFile.firstToken))
		}
	}
	return documentEnds, nil
}

type TokenChunkIterator struct {
	tokenFile   *TokenFile
	chunkLength int
	stepLength  int
	nextStart   int64
}

func (tokenFile *TokenFile) Chunks(chunkLength int) *TokenChunkIterator {
	return tokenFile.ChunksWithStep(chunkLength, chunkLength)
}

func (tokenFile *TokenFile) ChunksWithStep(chunkLength int, stepLength int) *TokenChunkIterator {
	return &TokenChunkIterator{
		tokenFile:   tokenFile,
		chunkLength: chunkLength,
		stepLength:  stepLength,
		nextStart:   0,
	}
}

func (iterator *TokenChunkIterator) NumberOfChunks() int {
	if iterator.chunkLength <= 0 || iterator.stepLength <= 0 {
		return 0
	}
	numberOfTokens := iterator.tokenFile.numberOfTokens
	if int64(iterator.chunkLength) > numberOfTokens {
		return 0
	}
	return int((numberOfTokens-int64(iterator.chunkLength))/int64(iterator.stepLength)) + 1
}

func (iterator *TokenChunkIterator) Next() ([]int, error) {
	if iterator.chunkLength <= 0 || iterator.stepLength <= 0 {
		return nil, fmt.Errorf("chunk length %d and step %d must both be positive", iterator.chunkLength, iterator.stepLength)
	}
	if iterator.nextStart+int64(iterator.chunkLength) > iterator.tokenFile.numberOfTokens {
		return nil, io.EOF
	}
	chunk, err := iterator.tokenFile.ReadTokens(int(iterator.nextStart), iterator.chunkLength)
	if err != nil {
		return nil, err
	}
	iterator.nextStart += int64(iterator.stepLength)
	return chunk, nil
}

func (iterator *TokenChunkIterator) NextBatch(maximumNumberOfChunks int) ([][]int, error) {
	var chunks [][]int
	for len(chunks) < maximumNumberOfChunks {
		chunk, err := iterator.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, err
		}
		chunks = append(chunks, chunk)
	}
	if len(chunks) == 0 {
		return nil, io.EOF
	}
	return chunks, nil
}

func (iterator *TokenChunkIterator) Restart() {
	iterator.nextStart = 0
}
