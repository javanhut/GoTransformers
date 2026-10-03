package chat

import (
	"fmt"
	"github.com/javanhut/GoTransformers/datafile"
	"github.com/javanhut/GoTransformers/tokenizer"
	"github.com/javanhut/GoTransformers/transformer"
)

var roleNames = map[string]string{
	"system":    "system",
	"user":      "user",
	"human":     "user",
	"assistant": "assistant",
	"gpt":       "assistant",
	"model":     "assistant",
	"bot":       "assistant",
	// Tool/function results fed back into the conversation. These are context for
	// the next assistant turn, not learning targets (ExamplesFromConversation only
	// learns "assistant" turns), and the templates render the "tool" role like any
	// other header.
	"tool":              "tool",
	"function":          "tool",
	"observation":       "tool",
	"tool_response":     "tool",
	"function_response": "tool",
	"ipython":           "tool",
}

func messageFromJSON(value any) (Message, error) {
	fields, isObject := value.(map[string]any)
	if !isObject {
		return Message{}, fmt.Errorf("a message is %T, not an object", value)
	}
	message := datafile.JSONRecord(fields)
	roleField, contentField := "role", "content"
	if !message.HasTextFields("role", "content") && message.HasTextFields("from", "value") {
		roleField, contentField = "from", "value"
	}
	role, err := message.Text(roleField)
	if err != nil {
		return Message{}, err
	}
	content, err := message.Text(contentField)
	if err != nil {
		return Message{}, err
	}
	standardRole, known := roleNames[role]
	if !known {
		return Message{}, fmt.Errorf("unknown role %q (expected system, user or assistant)", role)
	}
	return Message{Role: standardRole, Content: content}, nil
}

func ConversationFromJSON(record datafile.JSONRecord) ([]Message, error) {
	for _, field := range []string{"messages", "conversations"} {
		value, found := record[field]
		if !found {
			continue
		}
		list, isList := value.([]any)
		if !isList {
			return nil, fmt.Errorf("field %q is %T, not a list of messages", field, value)
		}
		messages := make([]Message, len(list))
		for index, item := range list {
			message, err := messageFromJSON(item)
			if err != nil {
				return nil, fmt.Errorf("message %d: %w", index+1, err)
			}
			messages[index] = message
		}
		return messages, nil
	}
	pair, err := record.Pair("input", "target")
	if err != nil {
		return nil, fmt.Errorf("has no \"messages\" list, and %w", err)
	}
	return ConversationFromPair(pair), nil
}

func ConversationFromPair(pair datafile.Pair) []Message {
	return []Message{{Role: "user", Content: pair.Input}, {Role: "assistant", Content: pair.Target}}
}

func ReadConversations(path string) ([][]Message, error) {
	var conversations [][]Message
	err := datafile.ForEachJSONLRecord(path, func(lineNumber int, record datafile.JSONRecord) error {
		messages, err := ConversationFromJSON(record)
		if err != nil {
			return err
		}
		conversations = append(conversations, messages)
		return nil
	})
	return conversations, err
}

func ExamplesFromConversation(messages []Message, template Template, textTokenizer *tokenizer.Tokenizer) ([]transformer.Example, error) {
	endOfTurnID, found := textTokenizer.SpecialTokenID(template.EndOfTurnText())
	if !found {
		return nil, fmt.Errorf("the tokenizer has no %q token, so it doesn't match the %s chat template", template.EndOfTurnText(), template.Style)
	}
	var examples []transformer.Example
	for index, message := range messages {
		if message.Role != "assistant" {
			continue
		}
		examples = append(examples, transformer.Example{
			PromptIDs: textTokenizer.Encode(template.Format(messages[:index], true)),
			AnswerIDs: append(textTokenizer.Encode(message.Content), endOfTurnID),
		})
	}
	return examples, nil
}

func ExamplesFromConversations(conversations [][]Message, template Template, textTokenizer *tokenizer.Tokenizer) ([]transformer.Example, error) {
	var examples []transformer.Example
	for number, messages := range conversations {
		conversationExamples, err := ExamplesFromConversation(messages, template, textTokenizer)
		if err != nil {
			return nil, fmt.Errorf("conversation %d: %w", number+1, err)
		}
		examples = append(examples, conversationExamples...)
	}
	return examples, nil
}
