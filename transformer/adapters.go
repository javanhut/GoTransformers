package transformer

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"transformer/dropout"
	"transformer/parameter"
	"transformer/perceptron"
	"transformer/weightfile"
)

func (model *Model) adapterLayers() []*perceptron.Layer {
	var layers []*perceptron.Layer
	for _, layer := range model.allLayers() {
		if layer != model.OutputLayer {
			layers = append(layers, layer)
		}
	}
	return layers
}

func (model *Model) AddLowRankAdapters(rank int, alpha float64) {
	if rank < 1 {
		panic(fmt.Sprintf("Model.AddLowRankAdapters: rank must be at least 1, got %d", rank))
	}
	if model.Settings.AdapterRank > 0 {
		panic("Model.AddLowRankAdapters: the model already has adapters, merge or remove them first")
	}
	for _, layer := range model.adapterLayers() {
		layer.AddLowRankAdapter(min(rank, layer.NumberOfInputs(), layer.NumberOfOutputs()), alpha)
		layer.Adapter.InputDropout = dropout.New(model.Settings.AdapterDropout)
	}
	model.Settings.AdapterRank = rank
	model.Settings.AdapterAlpha = alpha
	model.ReleaseGradients()
}

func (model *Model) RemoveAdapters() {
	for _, layer := range model.adapterLayers() {
		layer.RemoveLowRankAdapter()
	}
	model.Settings.AdapterRank = 0
	model.Settings.AdapterAlpha = 0
}

func (model *Model) MergeAdapters() {
	for _, layer := range model.adapterLayers() {
		layer.MergeLowRankAdapter()
	}
	model.Settings.AdapterRank = 0
	model.Settings.AdapterAlpha = 0
}

func (model *Model) AdapterParameters() []parameter.Parameter {
	var adapterParameters []parameter.Parameter
	for _, current := range model.Parameters() {
		if strings.Contains(current.Name, ".lora.") {
			adapterParameters = append(adapterParameters, current)
		}
	}
	return adapterParameters
}

type adapterSettings struct {
	Rank  int
	Alpha float64
}

func adapterSettingsPath(path string) string {
	return path + ".adapters.json"
}

func (model *Model) SaveAdapters(path string) error {
	if model.Settings.AdapterRank == 0 {
		return fmt.Errorf("the model has no adapters to save")
	}
	settingsText, err := json.MarshalIndent(adapterSettings{Rank: model.Settings.AdapterRank, Alpha: model.Settings.AdapterAlpha}, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(adapterSettingsPath(path), settingsText, 0o644); err != nil {
		return err
	}
	return weightfile.SaveBinary(path, model.AdapterParameters())
}

func (model *Model) LoadAdapters(path string) error {
	settingsText, err := os.ReadFile(adapterSettingsPath(path))
	if err != nil {
		return err
	}
	var saved adapterSettings
	if err := json.Unmarshal(settingsText, &saved); err != nil {
		return fmt.Errorf("%s: %w", adapterSettingsPath(path), err)
	}
	if model.Settings.AdapterRank == 0 {
		model.AddLowRankAdapters(saved.Rank, saved.Alpha)
	}
	if model.Settings.AdapterRank != saved.Rank || model.Settings.AdapterAlpha != saved.Alpha {
		return fmt.Errorf("%s: saved adapters have rank %d and alpha %v but the model's have rank %d and alpha %v", path, saved.Rank, saved.Alpha, model.Settings.AdapterRank, model.Settings.AdapterAlpha)
	}
	return weightfile.LoadBinary(path, model.AdapterParameters())
}
