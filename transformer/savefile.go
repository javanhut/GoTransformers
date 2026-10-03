package transformer

import (
	"encoding/json"
	"fmt"
	"os"
	"transformer/weightfile"
)

func settingsPath(path string) string {
	return path + ".settings.json"
}

func (model *Model) Save(path string) error {
	settingsText, err := json.MarshalIndent(model.Settings, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(settingsPath(path), settingsText, 0o644); err != nil {
		return err
	}
	return weightfile.SaveBinary(path, model.Parameters())
}

func LoadModel(path string) (*Model, error) {
	settingsText, err := os.ReadFile(settingsPath(path))
	if err != nil {
		return nil, err
	}
	var settings Settings
	if err := json.Unmarshal(settingsText, &settings); err != nil {
		return nil, err
	}
	model, err := NewModel(settings)
	if err != nil {
		return nil, err
	}
	savedValues, err := weightfile.ReadBinary(path)
	if err != nil {
		return nil, err
	}
	if err := model.SetWeights(savedValues); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return model, nil
}
