package gputraining

import (
	"github.com/javanhut/GoTransformers/gpu"
	"github.com/javanhut/GoTransformers/transformer"
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
	"testing"
)

func openTwoDevices(t *testing.T) []*gpu.Device {
	t.Helper()
	devices, err := gpu.ListDevices()
	if err != nil || len(devices) == 0 {
		t.Skipf("no Vulkan GPU available: %v", err)
	}
	var opened []*gpu.Device
	for copyNumber := range 2 {
		device, err := gpu.Open(devices[copyNumber%len(devices)].Index)
		if err != nil {
			t.Skipf("could not open a second GPU device: %v", err)
		}
		t.Cleanup(device.Close)
		opened = append(opened, device)
	}
	return opened
}

func TestDataParallelMatchesOneGPU(t *testing.T) {
	devices := openTwoDevices(t)
	settings := smallTestSettings()
	settings.TieOutputToEmbedding = true
	singleModel, parallelModel := twinModels(t, settings)
	options := DefaultTrainerOptions(testLearningRate, testWeightDecay)
	options.MaximumGradientNorm = 0.5
	single, err := NewTrainer(devices[0], singleModel, options)
	if err != nil {
		t.Fatal(err)
	}
	defer single.Close()
	parallel, err := NewDataParallelTrainer(devices, parallelModel, options)
	if err != nil {
		t.Fatal(err)
	}
	defer parallel.Close()
	parallel.ResynchronizeEvery = 2

	vectormath.SetRandomSeed(21)
	examples := []transformer.Example{
		{AnswerIDs: randomSequence(12, settings.VocabularySize)},
		{PromptIDs: randomSequence(4, settings.VocabularySize), AnswerIDs: randomSequence(6, settings.VocabularySize)},
		{AnswerIDs: randomSequence(7, settings.VocabularySize)},
	}
	for step := 1; step <= 4; step++ {
		singleLoss := single.TrainOnExamples(examples)
		parallelLoss := parallel.TrainOnExamples(examples)
		if math.Abs(singleLoss-parallelLoss)/singleLoss > 1e-4 {
			t.Fatalf("step %d: loss on one GPU %v, on two %v", step, singleLoss, parallelLoss)
		}
		if math.Abs(single.LastGradientNorm()-parallel.LastGradientNorm())/single.LastGradientNorm() > 1e-3 {
			t.Fatalf("step %d: gradient norm on one GPU %v, on two %v", step, single.LastGradientNorm(), parallel.LastGradientNorm())
		}
	}
	if err := single.CopyWeightsToModel(); err != nil {
		t.Fatal(err)
	}
	if err := parallel.CopyWeightsToModel(); err != nil {
		t.Fatal(err)
	}
	singleWeights := snapshot(singleModel)
	worst := 0.0
	for name, values := range snapshot(parallelModel) {
		for i, value := range values {
			worst = math.Max(worst, math.Abs(value-singleWeights[name][i]))
		}
	}
	if worst > 0.05*testLearningRate {
		t.Fatalf("after 4 steps the weights differ by %v (learning rate %v)", worst, testLearningRate)
	}
	sequences := [][]int{randomSequence(9, settings.VocabularySize), randomSequence(9, settings.VocabularySize), randomSequence(9, settings.VocabularySize)}
	if math.Abs(single.EvaluationLoss(sequences)-parallel.EvaluationLoss(sequences)) > 1e-4 {
		t.Fatalf("evaluation loss on one GPU %v, on two %v", single.EvaluationLoss(sequences), parallel.EvaluationLoss(sequences))
	}
}

func TestDataParallelWithFewerExamplesThanGPUs(t *testing.T) {
	devices := openTwoDevices(t)
	vectormath.SetRandomSeed(2)
	model, err := transformer.NewModel(smallTestSettings())
	if err != nil {
		t.Fatal(err)
	}
	parallel, err := NewDataParallelTrainer(devices, model, DefaultTrainerOptions(0.01, 0))
	if err != nil {
		t.Fatal(err)
	}
	defer parallel.Close()
	sequences := [][]int{{1, 2, 3, 4, 5, 6}}
	before := parallel.EvaluationLoss(sequences)
	for range 30 {
		parallel.TrainBatch(sequences)
	}
	if after := parallel.EvaluationLoss(sequences); after > before*0.5 {
		t.Fatalf("loss went from %v to %v", before, after)
	}
}
