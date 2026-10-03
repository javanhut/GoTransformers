package chat

import (
	"fmt"
	"strings"
	"transformer/tokenizer"
	"transformer/transformer"
	"transformer/vectormath"
	"unicode/utf8"
)

type ReplyOptions struct {
	MaximumNewTokens int
	Temperature      float64
	TopProbability   float64
}

func DefaultReplyOptions() ReplyOptions {
	return ReplyOptions{MaximumNewTokens: 256, Temperature: 0.2, TopProbability: 0.9}
}

type Conversation struct {
	Model     *transformer.Model
	Tokenizer *tokenizer.Tokenizer
	Template  Template
	Messages  []Message

	fedText           string
	tokensReprocessed int
}

func NewConversation(model *transformer.Model, textTokenizer *tokenizer.Tokenizer, template Template, systemPrompt string) *Conversation {
	conversation := &Conversation{Model: model, Tokenizer: textTokenizer, Template: template}
	if systemPrompt != "" {
		conversation.Messages = append(conversation.Messages, Message{Role: "system", Content: systemPrompt})
	}
	return conversation
}

func (conversation *Conversation) TokensReprocessed() int {
	return conversation.tokensReprocessed
}

func (conversation *Conversation) feedNewText() vectormath.Vector {
	fullText := conversation.Template.Format(conversation.Messages, true)
	if conversation.fedText == "" || !strings.HasPrefix(fullText, conversation.fedText) {
		conversation.Model.StartGenerating()
		conversation.fedText = ""
		conversation.tokensReprocessed += len(conversation.Tokenizer.Encode(fullText))
	}
	newText := strings.TrimPrefix(fullText, conversation.fedText)
	scores := conversation.Model.Feed(conversation.Tokenizer.Encode(newText))
	conversation.fedText = fullText
	return scores
}

func (conversation *Conversation) Reply(userText string, options ReplyOptions, onNewText func(piece string)) (string, error) {
	endOfTurnID, found := conversation.Tokenizer.SpecialTokenID(conversation.Template.EndOfTurnText())
	if !found {
		return "", fmt.Errorf("the tokenizer has no %q token, so it doesn't match the %s chat template", conversation.Template.EndOfTurnText(), conversation.Template.Style)
	}
	if options.MaximumNewTokens < 1 {
		return "", fmt.Errorf("MaximumNewTokens must be at least 1, got %d", options.MaximumNewTokens)
	}

	conversation.Messages = append(conversation.Messages, Message{Role: "user", Content: userText})
	scores := conversation.feedNewText()

	var replyIDs []int
	shownText := ""
	for len(replyIDs) < options.MaximumNewTokens {
		nextID := transformer.PickTokenFromTop(scores, options.Temperature, options.TopProbability)
		if nextID == endOfTurnID {
			break
		}
		replyIDs = append(replyIDs, nextID)
		scores = conversation.Model.NextTokenScores(nextID)

		replyText := conversation.Tokenizer.Decode(replyIDs)
		if onNewText != nil && utf8.ValidString(replyText) && len(replyText) > len(shownText) {
			onNewText(replyText[len(shownText):])
			shownText = replyText
		}
	}
	conversation.Model.NextTokenScores(endOfTurnID)

	replyText := conversation.Tokenizer.Decode(replyIDs)
	conversation.Messages = append(conversation.Messages, Message{Role: "assistant", Content: replyText})
	conversation.fedText += replyText + conversation.Template.EndOfTurnText()
	return replyText, nil
}
