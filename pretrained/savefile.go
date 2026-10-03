package pretrained

import (
	"github.com/javanhut/GoTransformers/tokenizer"
	"github.com/javanhut/GoTransformers/transformer"
	"github.com/javanhut/GoTransformers/weightfile"
)

func tokenizerPath(path string) string {
	return path + ".tokenizer"
}

func SaveModelAndTokenizer(path string, model *transformer.Model, loadedTokenizer *tokenizer.Tokenizer, precision weightfile.StoragePrecision) error {
	if err := model.SaveWithPrecision(path, precision); err != nil {
		return err
	}
	return loadedTokenizer.SaveToFile(tokenizerPath(path))
}

func LoadModelAndTokenizer(path string) (*transformer.Model, *tokenizer.Tokenizer, error) {
	model, err := transformer.LoadModel(path)
	if err != nil {
		return nil, nil, err
	}
	loadedTokenizer, err := tokenizer.LoadFromFile(tokenizerPath(path))
	if err != nil {
		return nil, nil, err
	}
	return model, loadedTokenizer, nil
}
