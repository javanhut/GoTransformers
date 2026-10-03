package optimizer

import (
	"github.com/javanhut/GoTransformers/parameter"
	"math"
	"testing"
)

func closeTo(first float64, second float64) bool {
	return math.Abs(first-second) < 1e-12
}

func TestEmptyScheduleKeepsTheLearningRate(t *testing.T) {
	schedule := LearningRateSchedule{}
	for _, stepIndex := range []int{0, 1, 100, 100000} {
		if fraction := schedule.FractionAt(stepIndex); fraction != 1 {
			t.Errorf("step %d: fraction %v, want 1", stepIndex, fraction)
		}
	}
}

func TestWarmupThenCosine(t *testing.T) {
	schedule := WarmupThenCosine(10, 110, 0.1)
	if err := schedule.Check(); err != nil {
		t.Fatal(err)
	}
	expected := map[int]float64{
		0:   0.1,
		4:   0.5,
		9:   1,
		10:  1,
		60:  0.1 + 0.9*0.5,
		110: 0.1,
		500: 0.1,
	}
	for stepIndex, wanted := range expected {
		if fraction := schedule.FractionAt(stepIndex); !closeTo(fraction, wanted) {
			t.Errorf("step %d: fraction %v, want %v", stepIndex, fraction, wanted)
		}
	}
	for stepIndex := 11; stepIndex < 110; stepIndex++ {
		if schedule.FractionAt(stepIndex) > schedule.FractionAt(stepIndex-1) {
			t.Fatalf("cosine decay went up between steps %d and %d", stepIndex-1, stepIndex)
		}
	}
}

func TestWarmupThenLinear(t *testing.T) {
	schedule := WarmupThenLinear(0, 100, 0)
	expected := map[int]float64{0: 1, 25: 0.75, 50: 0.5, 99: 0.01, 100: 0}
	for stepIndex, wanted := range expected {
		if fraction := schedule.FractionAt(stepIndex); !closeTo(fraction, wanted) {
			t.Errorf("step %d: fraction %v, want %v", stepIndex, fraction, wanted)
		}
	}
}

func TestWarmupStableDecay(t *testing.T) {
	schedule := WarmupStableDecay(5, 100, 20, 0)
	expected := map[int]float64{0: 0.2, 4: 1, 5: 1, 79: 1, 80: 1, 90: 0.5, 100: 0}
	for stepIndex, wanted := range expected {
		if fraction := schedule.FractionAt(stepIndex); !closeTo(fraction, wanted) {
			t.Errorf("step %d: fraction %v, want %v", stepIndex, fraction, wanted)
		}
	}
}

func TestBadSchedulesAreRejected(t *testing.T) {
	badSchedules := []LearningRateSchedule{
		{WarmupSteps: -1},
		{FinalFraction: 2},
		{DecayShape: "zigzag"},
		{WarmupSteps: 10, TotalSteps: 10, DecayShape: CosineDecay},
	}
	for _, schedule := range badSchedules {
		if schedule.Check() == nil {
			t.Errorf("schedule %+v was accepted", schedule)
		}
	}
}

func TestClipGradientsScalesToTheMaximumNorm(t *testing.T) {
	first := parameter.WithGradients("first", []float64{0, 0})
	second := parameter.WithGradients("second", []float64{0})
	copy(first.Gradients(), []float64{3, 0})
	copy(second.Gradients(), []float64{4})
	parameters := []parameter.Parameter{first, second}

	norm := ClipGradients(parameters, 10)
	if !closeTo(norm, 5) || first.Gradients()[0] != 3 {
		t.Fatalf("norm %v below the maximum should change nothing, gradients now %v %v", norm, first.Gradients(), second.Gradients())
	}
	norm = ClipGradients(parameters, 1)
	if !closeTo(norm, 5) {
		t.Fatalf("norm before clipping was %v, want 5", norm)
	}
	if clippedNorm := GradientNorm(parameters); math.Abs(clippedNorm-1) > 1e-6 {
		t.Fatalf("norm after clipping is %v, want 1", clippedNorm)
	}
	if !closeTo(first.Gradients()[0]/second.Gradients()[0], 0.75) {
		t.Fatalf("clipping changed the gradient direction: %v %v", first.Gradients(), second.Gradients())
	}
}

func TestLearningRatesCanBeScaled(t *testing.T) {
	muon := NewMuon(0.02)
	muon.AdamW.LearningRate = 0.003
	peak := LearningRates(muon)
	ScaleLearningRates(muon, peak, 0.5)
	if !closeTo(muon.LearningRate, 0.01) || !closeTo(muon.AdamW.LearningRate, 0.0015) {
		t.Fatalf("scaled Muon rates are %v and %v", muon.LearningRate, muon.AdamW.LearningRate)
	}
	adamW := NewAdamW(0.001, 0.1)
	ScaleLearningRates(adamW, LearningRates(adamW), 0.25)
	if !closeTo(adamW.LearningRate, 0.00025) {
		t.Fatalf("scaled AdamW rate is %v", adamW.LearningRate)
	}
}
