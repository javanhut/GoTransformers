package constrained

import "fmt"

type Automaton interface {
	Clone() Automaton
	AcceptByte(value byte) bool
	IsComplete() bool
	StateKey() string
}

const largestNumberOfRememberedMasks = 256

type TokenConstraint struct {
	tokenBytes           [][]byte
	startingState        Automaton
	currentState         Automaton
	acceptedText         []byte
	allowedTokensByState map[string][]bool
}

func NewTokenConstraint(tokenBytes [][]byte, startingState Automaton) *TokenConstraint {
	return &TokenConstraint{
		tokenBytes:           tokenBytes,
		startingState:        startingState.Clone(),
		currentState:         startingState.Clone(),
		allowedTokensByState: map[string][]bool{},
	}
}

func (constraint *TokenConstraint) Restart() {
	constraint.currentState = constraint.startingState.Clone()
	constraint.acceptedText = nil
}

func (constraint *TokenConstraint) Text() string {
	return string(constraint.acceptedText)
}

func (constraint *TokenConstraint) VocabularySize() int {
	return len(constraint.tokenBytes)
}

func (constraint *TokenConstraint) IsComplete() bool {
	return constraint.currentState.IsComplete()
}

func acceptsEveryByte(state Automaton, bytes []byte) bool {
	for _, value := range bytes {
		if !state.AcceptByte(value) {
			return false
		}
	}
	return true
}

func (constraint *TokenConstraint) IsTokenAllowed(tokenID int) bool {
	if tokenID < 0 || tokenID >= len(constraint.tokenBytes) || len(constraint.tokenBytes[tokenID]) == 0 {
		return false
	}
	return acceptsEveryByte(constraint.currentState.Clone(), constraint.tokenBytes[tokenID])
}

func (constraint *TokenConstraint) AllowedTokens() []bool {
	stateKey := constraint.currentState.StateKey()
	remembered, found := constraint.allowedTokensByState[stateKey]
	if found {
		return remembered
	}
	allowedTokens := constraint.findAllowedTokens()
	if len(constraint.allowedTokensByState) >= largestNumberOfRememberedMasks {
		clear(constraint.allowedTokensByState)
	}
	constraint.allowedTokensByState[stateKey] = allowedTokens
	return allowedTokens
}

func (constraint *TokenConstraint) acceptedFirstBytes() [256]bool {
	var accepted [256]bool
	for value := 0; value < 256; value++ {
		accepted[value] = constraint.currentState.Clone().AcceptByte(byte(value))
	}
	return accepted
}

func (constraint *TokenConstraint) findAllowedTokens() []bool {
	acceptedFirstBytes := constraint.acceptedFirstBytes()
	allowedTokens := make([]bool, len(constraint.tokenBytes))
	for tokenID, bytes := range constraint.tokenBytes {
		if len(bytes) == 0 || !acceptedFirstBytes[bytes[0]] {
			continue
		}
		if len(bytes) == 1 {
			allowedTokens[tokenID] = true
			continue
		}
		allowedTokens[tokenID] = acceptsEveryByte(constraint.currentState.Clone(), bytes)
	}
	return allowedTokens
}

func (constraint *TokenConstraint) AcceptToken(tokenID int) error {
	if !constraint.IsTokenAllowed(tokenID) {
		return fmt.Errorf("token %d is not allowed after %q", tokenID, constraint.acceptedText)
	}
	nextState := constraint.currentState.Clone()
	acceptsEveryByte(nextState, constraint.tokenBytes[tokenID])
	constraint.currentState = nextState
	constraint.acceptedText = append(constraint.acceptedText, constraint.tokenBytes[tokenID]...)
	return nil
}

func (constraint *TokenConstraint) AcceptText(text string) error {
	nextState := constraint.currentState.Clone()
	if !acceptsEveryByte(nextState, []byte(text)) {
		return fmt.Errorf("the text %q is not allowed after %q", text, constraint.acceptedText)
	}
	constraint.currentState = nextState
	constraint.acceptedText = append(constraint.acceptedText, text...)
	return nil
}
