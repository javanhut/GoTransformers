package chat

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

type Message struct {
	Role    string
	Content string
}

type TemplateStyle string

const (
	ChatMLStyle TemplateStyle = "ChatML"
	Llama3Style TemplateStyle = "Llama3"
)

type Template struct {
	Style               TemplateStyle
	DefaultSystemPrompt string
}

func ChatML(defaultSystemPrompt string) Template {
	return Template{Style: ChatMLStyle, DefaultSystemPrompt: defaultSystemPrompt}
}

func Llama3() Template {
	return Template{Style: Llama3Style}
}

func (template Template) EndOfTurnText() string {
	if template.Style == Llama3Style {
		return "<|eot_id|>"
	}
	return "<|im_end|>"
}

func (template Template) withDefaultSystemPrompt(messages []Message) []Message {
	if template.DefaultSystemPrompt == "" || (len(messages) > 0 && messages[0].Role == "system") {
		return messages
	}
	return append([]Message{{Role: "system", Content: template.DefaultSystemPrompt}}, messages...)
}

func (template Template) Format(messages []Message, startAssistantReply bool) string {
	var text strings.Builder
	switch template.Style {
	case Llama3Style:
		text.WriteString("<|begin_of_text|>")
		for _, message := range template.withDefaultSystemPrompt(messages) {
			text.WriteString("<|start_header_id|>" + message.Role + "<|end_header_id|>\n\n" + message.Content + "<|eot_id|>")
		}
		if startAssistantReply {
			text.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
		}
	default:
		for _, message := range template.withDefaultSystemPrompt(messages) {
			text.WriteString("<|im_start|>" + message.Role + "\n" + message.Content + "<|im_end|>\n")
		}
		if startAssistantReply {
			text.WriteString("<|im_start|>assistant\n")
		}
	}
	return text.String()
}

var chatMLDefaultSystemPrompt = regexp.MustCompile(`<\|im_start\|>system\n([^'<]+)<\|im_end\|>`)

func TemplateFromSource(source string) (Template, error) {
	switch {
	case strings.Contains(source, "<|im_start|>"):
		defaultSystemPrompt := ""
		if match := chatMLDefaultSystemPrompt.FindStringSubmatch(source); match != nil {
			defaultSystemPrompt = match[1]
		}
		return ChatML(defaultSystemPrompt), nil
	case strings.Contains(source, "<|start_header_id|>"):
		return Llama3(), nil
	}
	return Template{}, fmt.Errorf("unknown chat template, only ChatML (<|im_start|>) and Llama 3 (<|start_header_id|>) styles are supported")
}

func LoadTemplate(modelFolder string) (Template, error) {
	path := filepath.Join(modelFolder, "tokenizer_config.json")
	contents, err := os.ReadFile(path)
	if err != nil {
		return Template{}, err
	}
	var config struct {
		ChatTemplate json.RawMessage `json:"chat_template"`
	}
	if err := json.Unmarshal(contents, &config); err != nil {
		return Template{}, fmt.Errorf("%s: %w", path, err)
	}
	if len(config.ChatTemplate) == 0 {
		return Template{}, fmt.Errorf("%s has no chat_template, so this is probably not an instruct model", path)
	}

	var source string
	if err := json.Unmarshal(config.ChatTemplate, &source); err != nil {
		var namedTemplates []struct {
			Name     string `json:"name"`
			Template string `json:"template"`
		}
		if listErr := json.Unmarshal(config.ChatTemplate, &namedTemplates); listErr != nil {
			return Template{}, fmt.Errorf("%s: chat_template is neither text nor a list of named templates", path)
		}
		for _, named := range namedTemplates {
			if named.Name == "default" || source == "" {
				source = named.Template
			}
		}
	}
	template, err := TemplateFromSource(source)
	if err != nil {
		return Template{}, fmt.Errorf("%s: %w", path, err)
	}
	return template, nil
}
