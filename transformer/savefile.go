package transformer

import (
	"encoding/json"
	"fmt"
	"github.com/javanhut/GoTransformers/weightfile"
	"os"
)

func settingsPath(path string) string {
	return path + ".settings.json"
}

func (model *Model) Save(path string) error {
	return model.SaveWithPrecision(path, weightfile.Float64)
}

func (model *Model) SaveWithPrecision(path string, precision weightfile.StoragePrecision) error {
	settingsText, err := json.MarshalIndent(model.Settings, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(settingsPath(path), settingsText, 0o644); err != nil {
		return err
	}
	return weightfile.SaveBinaryWithPrecision(path, model.parametersToSave(), precision)
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
	savedValues, savedCompressedValues, err := weightfile.ReadBinaryKeepingCompressed(path)
	if err != nil {
		return nil, err
	}
	if err := model.SetWeightsAndCompressedWeights(savedValues, savedCompressedValues); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return model, nil
}
