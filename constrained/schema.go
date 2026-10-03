package constrained

import (
	"encoding/json"
	"fmt"
)

type ValueKind int

const (
	StringValue ValueKind = iota
	NumberValue
	BooleanValue
	EnumValue
	StringArrayValue
	ObjectValue
)

type ValueSchema struct {
	Kind       ValueKind
	Choices    []string
	Properties []Property
}

type Property struct {
	Name  string
	Value ValueSchema
}

func StringSchema() ValueSchema {
	return ValueSchema{Kind: StringValue}
}

func NumberSchema() ValueSchema {
	return ValueSchema{Kind: NumberValue}
}

func BooleanSchema() ValueSchema {
	return ValueSchema{Kind: BooleanValue}
}

func EnumSchema(choices ...string) ValueSchema {
	return ValueSchema{Kind: EnumValue, Choices: choices}
}

func StringArraySchema() ValueSchema {
	return ValueSchema{Kind: StringArrayValue}
}

func ObjectSchema(properties ...Property) ValueSchema {
	return ValueSchema{Kind: ObjectValue, Properties: properties}
}

func NewSchemaConstraint(tokenBytes [][]byte, schema ValueSchema, settings JSONSettings) (*TokenConstraint, error) {
	automaton, err := NewSchemaAutomaton(schema, settings)
	if err != nil {
		return nil, err
	}
	return NewTokenConstraint(tokenBytes, automaton), nil
}

func NewSchemaAutomaton(schema ValueSchema, settings JSONSettings) (Automaton, error) {
	switch schema.Kind {
	case StringValue:
		return newJSONStringAutomaton(), nil
	case NumberValue:
		return newJSONNumberAutomaton(settings), nil
	case BooleanValue:
		return NewChoiceAutomaton([]string{"true", "false"}), nil
	case EnumValue:
		return newEnumAutomaton(schema.Choices)
	case StringArrayValue:
		return newJSONStringArrayAutomaton(settings), nil
	case ObjectValue:
		return newObjectAutomaton(schema.Properties, settings)
	}
	return nil, fmt.Errorf("unknown value kind %d", schema.Kind)
}

func quotedJSONString(text string) string {
	quoted, _ := json.Marshal(text)
	return string(quoted)
}

func newEnumAutomaton(choices []string) (Automaton, error) {
	if len(choices) == 0 {
		return nil, fmt.Errorf("an enum needs at least 1 choice")
	}
	quotedChoices := make([]string, len(choices))
	for i, choice := range choices {
		quotedChoices[i] = quotedJSONString(choice)
	}
	return NewChoiceAutomaton(quotedChoices), nil
}

func exactText(text string) Automaton {
	return NewChoiceAutomaton([]string{text})
}

func newObjectAutomaton(properties []Property, settings JSONSettings) (Automaton, error) {
	pieces := []Automaton{exactText("{")}
	for i, property := range properties {
		if i > 0 {
			pieces = append(pieces, exactText(","))
		}
		pieces = append(pieces, exactText(quotedJSONString(property.Name)), exactText(":"))
		valueAutomaton, err := NewSchemaAutomaton(property.Value, settings)
		if err != nil {
			return nil, fmt.Errorf("property %q: %w", property.Name, err)
		}
		pieces = append(pieces, valueAutomaton)
	}
	pieces = append(pieces, exactText("}"))
	return newSequenceAutomaton(pieces, settings.MaximumWhitespaceInARow), nil
}

type sequenceAutomaton struct {
	pieces                  []Automaton
	maximumWhitespaceInARow int

	pieceIndex          int
	currentPiece        Automaton
	currentPieceStarted bool
	whitespaceInARow    int
}

func newSequenceAutomaton(pieces []Automaton, maximumWhitespaceInARow int) *sequenceAutomaton {
	return &sequenceAutomaton{pieces: pieces, maximumWhitespaceInARow: maximumWhitespaceInARow, currentPiece: pieces[0].Clone()}
}

func (automaton *sequenceAutomaton) Clone() Automaton {
	copied := *automaton
	copied.currentPiece = automaton.currentPiece.Clone()
	return &copied
}

func (automaton *sequenceAutomaton) StateKey() string {
	return fmt.Sprintf("%d|%t|%d|%s", automaton.pieceIndex, automaton.currentPieceStarted, automaton.whitespaceInARow, automaton.currentPiece.StateKey())
}

func (automaton *sequenceAutomaton) IsComplete() bool {
	return automaton.pieceIndex == len(automaton.pieces)-1 && automaton.currentPiece.IsComplete()
}

func (automaton *sequenceAutomaton) canTakeMoreWhitespace() bool {
	return automaton.maximumWhitespaceInARow <= 0 || automaton.whitespaceInARow < automaton.maximumWhitespaceInARow
}

func (automaton *sequenceAutomaton) AcceptByte(value byte) bool {
	if !automaton.currentPieceStarted && automaton.pieceIndex > 0 && isJSONWhitespace(value) {
		if !automaton.canTakeMoreWhitespace() {
			return false
		}
		automaton.whitespaceInARow++
		return true
	}
	if automaton.currentPiece.AcceptByte(value) {
		automaton.currentPieceStarted = true
		automaton.whitespaceInARow = 0
		return true
	}
	if !automaton.currentPiece.IsComplete() || automaton.pieceIndex+1 >= len(automaton.pieces) {
		return false
	}
	nextPiece := automaton.pieces[automaton.pieceIndex+1].Clone()
	if isJSONWhitespace(value) {
		if !automaton.canTakeMoreWhitespace() {
			return false
		}
		automaton.moveToPiece(automaton.pieceIndex+1, nextPiece)
		automaton.currentPieceStarted = false
		automaton.whitespaceInARow++
		return true
	}
	if !nextPiece.AcceptByte(value) {
		return false
	}
	automaton.moveToPiece(automaton.pieceIndex+1, nextPiece)
	automaton.currentPieceStarted = true
	automaton.whitespaceInARow = 0
	return true
}

func (automaton *sequenceAutomaton) moveToPiece(pieceIndex int, piece Automaton) {
	automaton.pieceIndex = pieceIndex
	automaton.currentPiece = piece
}
