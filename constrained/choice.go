package constrained

type ChoiceAutomaton struct {
	choices     []string
	matchedText string
}

func NewChoiceAutomaton(choices []string) *ChoiceAutomaton {
	return &ChoiceAutomaton{choices: choices}
}

func NewChoiceConstraint(tokenBytes [][]byte, choices []string) *TokenConstraint {
	return NewTokenConstraint(tokenBytes, NewChoiceAutomaton(choices))
}

func (automaton *ChoiceAutomaton) Clone() Automaton {
	copied := *automaton
	return &copied
}

func (automaton *ChoiceAutomaton) StateKey() string {
	return automaton.matchedText
}

func (automaton *ChoiceAutomaton) AcceptByte(value byte) bool {
	matchedLength := len(automaton.matchedText)
	for _, choice := range automaton.choices {
		if len(choice) <= matchedLength || choice[:matchedLength] != automaton.matchedText {
			continue
		}
		if choice[matchedLength] == value {
			automaton.matchedText = choice[:matchedLength+1]
			return true
		}
	}
	return false
}

func (automaton *ChoiceAutomaton) IsComplete() bool {
	for _, choice := range automaton.choices {
		if choice == automaton.matchedText {
			return true
		}
	}
	return false
}
