package transformer

import (
	"github.com/javanhut/GoTransformers/optimizer"
	"github.com/javanhut/GoTransformers/parameter"
	"math"
	"reflect"
	"testing"
)

func dropoutSettings() Settings {
	settings := tinySettings()
	settings.ResidualDropout = 0.2
	settings.AttentionDropout = 0.2
	return settings
}

func TestModelGradientsWithFrozenDropoutMasks(t *testing.T) {
	const stepSize = 1e-6
	for name, settings := range map[string]Settings{"dropout": dropoutSettings(), "dropout with adapters": dropoutSettings()} {
		model := newModel(t, settings)
		if name == "dropout with adapters" {
			model.Settings.AdapterDropout = 0.3
			model.AddLowRankAdapters(2, 4)
			for _, current := range model.AdapterParameters() {
				for i := range current.Values {
					current.Values[i] = 0.1 * float64(i%7-3)
				}
			}
		}
		tokenIDs := []int{1, 5, 2, 9, 3, 1}

		model.ComputeGradients(tokenIDs)
		for _, current := range model.allDropouts() {
			if current != nil {
				current.RepeatLastMasks = true
			}
		}
		parameters := model.TrainableParameters()
		parameter.ZeroGradients(parameters)
		model.ComputeGradients(tokenIDs)

		lossWithSameMasks := func() float64 {
			model.setDropoutActive(true)
			defer model.setDropoutActive(false)
			return model.lossAndGradients(tokenIDs, 1, false)
		}
		for _, current := range parameters {
			for i := 0; i < len(current.Values); i += 5 {
				original := current.Values[i]
				current.Values[i] = original + stepSize
				higher := lossWithSameMasks()
				current.Values[i] = original - stepSize
				lower := lossWithSameMasks()
				current.Values[i] = original
				numerical := (higher - lower) / (2 * stepSize)
				if math.Abs(numerical-current.Gradients()[i]) > 1e-5*math.Max(1, math.Abs(numerical)) {
					t.Errorf("%s: %s[%d]: backward gave %v, finite difference gave %v", name, current.Name, i, current.Gradients()[i], numerical)
				}
			}
		}
	}
}

func TestDropoutOnlyActsDuringTraining(t *testing.T) {
	model := newModel(t, dropoutSettings())
	tokenIDs := []int{1, 5, 2, 9}
	first := model.Forward(tokenIDs)
	model.TrainStep([]int{1, 2, 3, 4}, optimizer.NewSGD(0))
	second := model.Forward(tokenIDs)
	if !reflect.DeepEqual(first.Values, second.Values) {
		t.Error("Forward outside training should not use dropout, before or after a training step")
	}
	for _, current := range model.allDropouts() {
		if current.IsOn() {
			t.Error("dropout was left switched on after training")
		}
	}
}

func TestDropoutMakesEachTrainingPassDifferent(t *testing.T) {
	model := newModel(t, dropoutSettings())
	tokenIDs := []int{1, 5, 2, 9, 3}
	firstLoss := model.ComputeGradients(tokenIDs)
	secondLoss := model.ComputeGradients(tokenIDs)
	if firstLoss == secondLoss {
		t.Error("two training passes with dropout gave exactly the same loss")
	}
	if model.TrainingLoss(tokenIDs) != model.TrainingLoss(tokenIDs) {
		t.Error("the loss outside training should not change between calls")
	}
}

func TestBadDropoutSettings(t *testing.T) {
	settings := tinySettings()
	settings.ResidualDropout = 1
	if _, err := NewModel(settings); err == nil {
		t.Error("a dropout rate of 1 should be refused")
	}
}
