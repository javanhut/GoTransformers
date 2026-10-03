package constrained

import (
	"fmt"
	"slices"
)

type JSONSettings struct {
	MaximumDepth            int
	MaximumWhitespaceInARow int
	MaximumNumberLength     int
	RequireObjectAtTopLevel bool
}

func DefaultJSONSettings() JSONSettings {
	return JSONSettings{MaximumDepth: 16, MaximumWhitespaceInARow: 16, MaximumNumberLength: 32}
}

type jsonMode int

const (
	expectingValue jsonMode = iota
	expectingValueOrArrayEnd
	expectingKey
	expectingKeyOrObjectEnd
	expectingColon
	expectingCommaOrEnd
	finished
	insideString
	insideStringEscape
	insideUnicodeEscape
	insideLiteral
	numberAfterMinus
	numberAfterLeadingZero
	numberIntegerDigits
	numberAfterDot
	numberFractionDigits
	numberAfterExponentMark
	numberAfterExponentSign
	numberExponentDigits
)

type rootValueKind int

const (
	anyRootValue rootValueKind = iota
	onlyStringRootValue
	onlyNumberRootValue
	onlyStringArrayRootValue
	onlyObjectRootValue
)

type JSONAutomaton struct {
	settings          JSONSettings
	allowedRootValue  rootValueKind
	whitespaceAllowed bool

	mode              jsonMode
	openContainers    []byte
	stringIsKey       bool
	unicodeDigitsRead int
	literalLeftToRead string
	utf8BytesLeft     int
	utf8NextLowest    byte
	utf8NextHighest   byte
	whitespaceInARow  int
	numberLength      int
}

func NewJSONAutomaton(settings JSONSettings) *JSONAutomaton {
	automaton := &JSONAutomaton{settings: settings, allowedRootValue: anyRootValue, whitespaceAllowed: true, mode: expectingValue}
	if settings.RequireObjectAtTopLevel {
		automaton.allowedRootValue = onlyObjectRootValue
	}
	return automaton
}

func newJSONStringAutomaton() *JSONAutomaton {
	return &JSONAutomaton{allowedRootValue: onlyStringRootValue, mode: expectingValue}
}

func newJSONNumberAutomaton(settings JSONSettings) *JSONAutomaton {
	return &JSONAutomaton{settings: settings, allowedRootValue: onlyNumberRootValue, mode: expectingValue}
}

func newJSONStringArrayAutomaton(settings JSONSettings) *JSONAutomaton {
	return &JSONAutomaton{settings: settings, allowedRootValue: onlyStringArrayRootValue, whitespaceAllowed: true, mode: expectingValue}
}

func NewJSONConstraint(tokenBytes [][]byte, settings JSONSettings) *TokenConstraint {
	return NewTokenConstraint(tokenBytes, NewJSONAutomaton(settings))
}

func (automaton *JSONAutomaton) Clone() Automaton {
	copied := *automaton
	copied.openContainers = slices.Clone(automaton.openContainers)
	return &copied
}

func (automaton *JSONAutomaton) StateKey() string {
	return fmt.Sprintf("%d|%s|%t|%d|%s|%d|%d|%d|%d|%d", automaton.mode, automaton.openContainers, automaton.stringIsKey,
		automaton.unicodeDigitsRead, automaton.literalLeftToRead, automaton.utf8BytesLeft, automaton.utf8NextLowest,
		automaton.utf8NextHighest, automaton.whitespaceInARow, automaton.numberLength)
}

func (automaton *JSONAutomaton) IsComplete() bool {
	if automaton.mode == finished {
		return true
	}
	if len(automaton.openContainers) > 0 {
		return false
	}
	return automaton.isInCompleteNumber()
}

func (automaton *JSONAutomaton) isInCompleteNumber() bool {
	switch automaton.mode {
	case numberAfterLeadingZero, numberIntegerDigits, numberFractionDigits, numberExponentDigits:
		return true
	}
	return false
}

func (automaton *JSONAutomaton) isInNumber() bool {
	switch automaton.mode {
	case numberAfterMinus, numberAfterLeadingZero, numberIntegerDigits, numberAfterDot, numberFractionDigits,
		numberAfterExponentMark, numberAfterExponentSign, numberExponentDigits:
		return true
	}
	return false
}

func isNumberCharacter(value byte) bool {
	return isDigit(value) || value == '.' || value == '-' || value == '+' || isExponentMark(value)
}

func isJSONWhitespace(value byte) bool {
	return value == ' ' || value == '\t' || value == '\n' || value == '\r'
}

func isDigit(value byte) bool {
	return value >= '0' && value <= '9'
}

func isHexDigit(value byte) bool {
	return isDigit(value) || (value >= 'a' && value <= 'f') || (value >= 'A' && value <= 'F')
}

func isExponentMark(value byte) bool {
	return value == 'e' || value == 'E'
}

func (automaton *JSONAutomaton) isBetweenTokens() bool {
	switch automaton.mode {
	case expectingValue, expectingValueOrArrayEnd, expectingKey, expectingKeyOrObjectEnd, expectingColon, expectingCommaOrEnd, finished:
		return true
	}
	return false
}

func (automaton *JSONAutomaton) AcceptByte(value byte) bool {
	if automaton.isBetweenTokens() && isJSONWhitespace(value) {
		return automaton.acceptWhitespace()
	}
	accepted := automaton.acceptByteForMode(value)
	if accepted && !isJSONWhitespace(value) {
		automaton.whitespaceInARow = 0
	}
	return accepted
}

func (automaton *JSONAutomaton) acceptWhitespace() bool {
	if automaton.mode == finished || !automaton.whitespaceAllowed {
		return false
	}
	maximum := automaton.settings.MaximumWhitespaceInARow
	if maximum > 0 && automaton.whitespaceInARow >= maximum {
		return false
	}
	automaton.whitespaceInARow++
	return true
}

func (automaton *JSONAutomaton) acceptByteForMode(value byte) bool {
	switch automaton.mode {
	case insideString:
		return automaton.acceptStringByte(value)
	case insideStringEscape:
		return automaton.acceptEscapeByte(value)
	case insideUnicodeEscape:
		return automaton.acceptUnicodeDigit(value)
	case insideLiteral:
		return automaton.acceptLiteralByte(value)
	}
	if automaton.isInNumber() {
		return automaton.acceptNumberByte(value)
	}
	return automaton.acceptStructuralByte(value)
}

func (automaton *JSONAutomaton) acceptStructuralByte(value byte) bool {
	switch automaton.mode {
	case expectingValue:
		return automaton.startValue(value)
	case expectingValueOrArrayEnd:
		if value == ']' {
			return automaton.closeContainer()
		}
		return automaton.startValue(value)
	case expectingKeyOrObjectEnd:
		if value == '}' {
			return automaton.closeContainer()
		}
		return automaton.startKey(value)
	case expectingKey:
		return automaton.startKey(value)
	case expectingColon:
		if value != ':' {
			return false
		}
		automaton.mode = expectingValue
		return true
	case expectingCommaOrEnd:
		return automaton.acceptCommaOrEnd(value)
	}
	return false
}

func (automaton *JSONAutomaton) innermostContainer() byte {
	return automaton.openContainers[len(automaton.openContainers)-1]
}

func (automaton *JSONAutomaton) acceptCommaOrEnd(value byte) bool {
	innermost := automaton.innermostContainer()
	if value == ',' && innermost == '{' {
		automaton.mode = expectingKey
		return true
	}
	if value == ',' && innermost == '[' {
		automaton.mode = expectingValue
		return true
	}
	if (value == '}' && innermost == '{') || (value == ']' && innermost == '[') {
		return automaton.closeContainer()
	}
	return false
}

func (automaton *JSONAutomaton) valueAllowedHere(value byte) bool {
	depth := len(automaton.openContainers)
	switch automaton.allowedRootValue {
	case onlyStringRootValue:
		return depth > 0 || value == '"'
	case onlyNumberRootValue:
		return depth > 0 || value == '-' || isDigit(value)
	case onlyObjectRootValue:
		return depth > 0 || value == '{'
	case onlyStringArrayRootValue:
		if depth == 0 {
			return value == '['
		}
		return value == '"'
	}
	return true
}

func (automaton *JSONAutomaton) startValue(value byte) bool {
	if !automaton.valueAllowedHere(value) {
		return false
	}
	switch {
	case value == '"':
		automaton.mode = insideString
		automaton.stringIsKey = false
	case value == '{' || value == '[':
		return automaton.openContainer(value)
	case value == 't':
		automaton.startLiteral("rue")
	case value == 'f':
		automaton.startLiteral("alse")
	case value == 'n':
		automaton.startLiteral("ull")
	case value == '-':
		automaton.startNumber(numberAfterMinus)
	case value == '0':
		automaton.startNumber(numberAfterLeadingZero)
	case isDigit(value):
		automaton.startNumber(numberIntegerDigits)
	default:
		return false
	}
	return true
}

func (automaton *JSONAutomaton) startNumber(firstMode jsonMode) {
	automaton.mode = firstMode
	automaton.numberLength = 1
}

func (automaton *JSONAutomaton) startKey(value byte) bool {
	if value != '"' {
		return false
	}
	automaton.mode = insideString
	automaton.stringIsKey = true
	return true
}

func (automaton *JSONAutomaton) startLiteral(restOfLiteral string) {
	automaton.mode = insideLiteral
	automaton.literalLeftToRead = restOfLiteral
}

func (automaton *JSONAutomaton) openContainer(value byte) bool {
	maximumDepth := automaton.settings.MaximumDepth
	if maximumDepth > 0 && len(automaton.openContainers) >= maximumDepth {
		return false
	}
	automaton.openContainers = append(automaton.openContainers, value)
	if value == '{' {
		automaton.mode = expectingKeyOrObjectEnd
	} else {
		automaton.mode = expectingValueOrArrayEnd
	}
	return true
}

func (automaton *JSONAutomaton) closeContainer() bool {
	automaton.openContainers = automaton.openContainers[:len(automaton.openContainers)-1]
	automaton.finishValue()
	return true
}

func (automaton *JSONAutomaton) finishValue() {
	if len(automaton.openContainers) == 0 {
		automaton.mode = finished
	} else {
		automaton.mode = expectingCommaOrEnd
	}
}

func (automaton *JSONAutomaton) acceptStringByte(value byte) bool {
	if automaton.utf8BytesLeft > 0 {
		return automaton.acceptUTF8ContinuationByte(value)
	}
	switch {
	case value == '"':
		if automaton.stringIsKey {
			automaton.mode = expectingColon
		} else {
			automaton.finishValue()
		}
		return true
	case value == '\\':
		automaton.mode = insideStringEscape
		return true
	case value < 0x20:
		return false
	case value < 0x80:
		return true
	}
	return automaton.startUTF8Character(value)
}

func (automaton *JSONAutomaton) acceptUTF8ContinuationByte(value byte) bool {
	if value < automaton.utf8NextLowest || value > automaton.utf8NextHighest {
		return false
	}
	automaton.utf8BytesLeft--
	automaton.utf8NextLowest = 0x80
	automaton.utf8NextHighest = 0xBF
	return true
}

func (automaton *JSONAutomaton) expectContinuationBytes(count int, nextLowest byte, nextHighest byte) bool {
	automaton.utf8BytesLeft = count
	automaton.utf8NextLowest = nextLowest
	automaton.utf8NextHighest = nextHighest
	return true
}

func (automaton *JSONAutomaton) startUTF8Character(firstByte byte) bool {
	switch {
	case firstByte >= 0xC2 && firstByte <= 0xDF:
		return automaton.expectContinuationBytes(1, 0x80, 0xBF)
	case firstByte == 0xE0:
		return automaton.expectContinuationBytes(2, 0xA0, 0xBF)
	case firstByte == 0xED:
		return automaton.expectContinuationBytes(2, 0x80, 0x9F)
	case firstByte >= 0xE1 && firstByte <= 0xEF:
		return automaton.expectContinuationBytes(2, 0x80, 0xBF)
	case firstByte == 0xF0:
		return automaton.expectContinuationBytes(3, 0x90, 0xBF)
	case firstByte >= 0xF1 && firstByte <= 0xF3:
		return automaton.expectContinuationBytes(3, 0x80, 0xBF)
	case firstByte == 0xF4:
		return automaton.expectContinuationBytes(3, 0x80, 0x8F)
	}
	return false
}

func (automaton *JSONAutomaton) acceptEscapeByte(value byte) bool {
	switch value {
	case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
		automaton.mode = insideString
		return true
	case 'u':
		automaton.mode = insideUnicodeEscape
		automaton.unicodeDigitsRead = 0
		return true
	}
	return false
}

func (automaton *JSONAutomaton) acceptUnicodeDigit(value byte) bool {
	if !isHexDigit(value) {
		return false
	}
	automaton.unicodeDigitsRead++
	if automaton.unicodeDigitsRead == 4 {
		automaton.mode = insideString
		automaton.unicodeDigitsRead = 0
	}
	return true
}

func (automaton *JSONAutomaton) acceptLiteralByte(value byte) bool {
	if value != automaton.literalLeftToRead[0] {
		return false
	}
	automaton.literalLeftToRead = automaton.literalLeftToRead[1:]
	if automaton.literalLeftToRead == "" {
		automaton.finishValue()
	}
	return true
}

func (automaton *JSONAutomaton) numberMustEndNow() bool {
	maximum := automaton.settings.MaximumNumberLength
	return maximum > 0 && automaton.numberLength >= maximum && automaton.isInCompleteNumber()
}

func (automaton *JSONAutomaton) acceptNumberByte(value byte) bool {
	if isNumberCharacter(value) && automaton.numberMustEndNow() {
		return false
	}
	if !automaton.continueOrEndNumber(value) {
		return false
	}
	if automaton.isInNumber() {
		automaton.numberLength++
	} else {
		automaton.numberLength = 0
	}
	return true
}

func (automaton *JSONAutomaton) continueOrEndNumber(value byte) bool {
	switch automaton.mode {
	case numberAfterMinus:
		if value == '0' {
			automaton.mode = numberAfterLeadingZero
			return true
		}
		if isDigit(value) {
			automaton.mode = numberIntegerDigits
			return true
		}
		return false
	case numberAfterLeadingZero:
		return automaton.acceptFractionOrExponentStart(value)
	case numberIntegerDigits:
		if isDigit(value) {
			return true
		}
		return automaton.acceptFractionOrExponentStart(value)
	case numberAfterDot:
		if isDigit(value) {
			automaton.mode = numberFractionDigits
			return true
		}
		return false
	case numberFractionDigits:
		if isDigit(value) {
			return true
		}
		if isExponentMark(value) {
			automaton.mode = numberAfterExponentMark
			return true
		}
		return automaton.endNumber(value)
	case numberAfterExponentMark:
		if value == '+' || value == '-' {
			automaton.mode = numberAfterExponentSign
			return true
		}
		if isDigit(value) {
			automaton.mode = numberExponentDigits
			return true
		}
		return false
	case numberAfterExponentSign:
		if isDigit(value) {
			automaton.mode = numberExponentDigits
			return true
		}
		return false
	case numberExponentDigits:
		if isDigit(value) {
			return true
		}
		return automaton.endNumber(value)
	}
	return false
}

func (automaton *JSONAutomaton) acceptFractionOrExponentStart(value byte) bool {
	if value == '.' {
		automaton.mode = numberAfterDot
		return true
	}
	if isExponentMark(value) {
		automaton.mode = numberAfterExponentMark
		return true
	}
	return automaton.endNumber(value)
}

func (automaton *JSONAutomaton) endNumber(value byte) bool {
	if len(automaton.openContainers) == 0 {
		return false
	}
	modeBefore := automaton.mode
	automaton.mode = expectingCommaOrEnd
	if automaton.AcceptByte(value) {
		return true
	}
	automaton.mode = modeBefore
	return false
}
