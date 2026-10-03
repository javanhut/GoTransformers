package gputraining

import (
	"github.com/javanhut/GoTransformers/transformer"
	"github.com/javanhut/GoTransformers/vectormath"
	"os"
	"sort"
	"testing"
	"time"
)

func TestProfileTrainingStep(t *testing.T) {
	if os.Getenv("GPUTRAINING_PROFILE") == "" {
		t.Skip("set GPUTRAINING_PROFILE=1 to profile a training step of the 6.9M parameter model")
	}
	device := openTestDevice(t)
	settings := transformer.SmallSettings(4096)
	settings.VectorSize = 256
	settings.NumberOfBlocks = 6
	settings.NumberOfHeads = 8
	settings.FeedForwardSize = 688
	vectormath.SetRandomSeed(1)
	model, err := transformer.NewModel(settings)
	if err != nil {
		t.Fatal(err)
	}
	trainer, err := NewTrainer(device, model, DefaultTrainerOptions(0.001, 0.01))
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	text := randomSequence(20000, 4096)
	for step := 0; step < 30; step++ {
		trainer.TrainBatch(transformer.RandomChunks(text, 129, 4))
	}
	if err := trainer.recorder.MeasureTime(true); err != nil {
		t.Skip(err)
	}
	stepStart := time.Now()
	trainer.TrainBatch(transformer.RandomChunks(text, 129, 4))
	t.Logf("the measured step took %v of wall time", time.Since(stepStart))
	times := trainer.recorder.DispatchTimes()
	totals := map[string]float64{}
	counts := map[string]int{}
	total := 0.0
	for _, measured := range times {
		totals[measured.ProgramName] += measured.Seconds
		counts[measured.ProgramName]++
		total += measured.Seconds
	}
	var names []string
	for name := range totals {
		names = append(names, name)
	}
	sort.Slice(names, func(i, j int) bool { return totals[names[i]] > totals[names[j]] })
	t.Logf("%d dispatches, %.2f ms on the GPU", len(times), total*1000)
	for _, name := range names {
		t.Logf("%-24s %4d dispatches %8.2f ms", name, counts[name], totals[name]*1000)
	}
	for index, measured := range times {
		if index < 30 || (index >= 140 && index < 200) {
			t.Logf("dispatch %d %s %.2f ms", index, measured.ProgramName, measured.Seconds*1000)
		}
	}
}
