package optimizer

import (
	"fmt"
	"math"
)

type DecayShape string

const (
	NoDecay     DecayShape = "none"
	LinearDecay DecayShape = "linear"
	CosineDecay DecayShape = "cosine"
)

type LearningRateSchedule struct {
	WarmupSteps   int
	TotalSteps    int
	DecaySteps    int
	DecayShape    DecayShape
	FinalFraction float64
}

func WarmupThenCosine(warmupSteps int, totalSteps int, finalFraction float64) LearningRateSchedule {
	return LearningRateSchedule{WarmupSteps: warmupSteps, TotalSteps: totalSteps, DecayShape: CosineDecay, FinalFraction: finalFraction}
}

func WarmupThenLinear(warmupSteps int, totalSteps int, finalFraction float64) LearningRateSchedule {
	return LearningRateSchedule{WarmupSteps: warmupSteps, TotalSteps: totalSteps, DecayShape: LinearDecay, FinalFraction: finalFraction}
}

func WarmupStableDecay(warmupSteps int, totalSteps int, decaySteps int, finalFraction float64) LearningRateSchedule {
	return LearningRateSchedule{WarmupSteps: warmupSteps, TotalSteps: totalSteps, DecaySteps: decaySteps, DecayShape: LinearDecay, FinalFraction: finalFraction}
}

func (schedule LearningRateSchedule) Check() error {
	if schedule.WarmupSteps < 0 || schedule.TotalSteps < 0 || schedule.DecaySteps < 0 {
		return fmt.Errorf("learning rate schedule steps can't be negative, got WarmupSteps %d, TotalSteps %d and DecaySteps %d", schedule.WarmupSteps, schedule.TotalSteps, schedule.DecaySteps)
	}
	if schedule.FinalFraction < 0 || schedule.FinalFraction > 1 {
		return fmt.Errorf("learning rate schedule FinalFraction must be between 0 and 1, got %v", schedule.FinalFraction)
	}
	if schedule.DecayShape != "" && schedule.DecayShape != NoDecay && schedule.DecayShape != LinearDecay && schedule.DecayShape != CosineDecay {
		return fmt.Errorf("learning rate schedule DecayShape must be %q, %q or %q, got %q", NoDecay, LinearDecay, CosineDecay, schedule.DecayShape)
	}
	if schedule.decays() && schedule.TotalSteps <= schedule.WarmupSteps {
		return fmt.Errorf("a decaying learning rate schedule needs TotalSteps (%d) above WarmupSteps (%d)", schedule.TotalSteps, schedule.WarmupSteps)
	}
	return nil
}

func (schedule LearningRateSchedule) decays() bool {
	return schedule.DecayShape == LinearDecay || schedule.DecayShape == CosineDecay
}

func (schedule LearningRateSchedule) firstDecayStep() int {
	if schedule.DecaySteps > 0 {
		return max(schedule.WarmupSteps, schedule.TotalSteps-schedule.DecaySteps)
	}
	return schedule.WarmupSteps
}

func (schedule LearningRateSchedule) FractionAt(stepIndex int) float64 {
	if stepIndex < schedule.WarmupSteps {
		return float64(stepIndex+1) / float64(schedule.WarmupSteps)
	}
	if !schedule.decays() {
		return 1
	}
	firstDecayStep := schedule.firstDecayStep()
	if stepIndex < firstDecayStep {
		return 1
	}
	if stepIndex >= schedule.TotalSteps {
		return schedule.FinalFraction
	}
	progress := float64(stepIndex-firstDecayStep) / float64(schedule.TotalSteps-firstDecayStep)
	remaining := 1 - progress
	if schedule.DecayShape == CosineDecay {
		remaining = 0.5 * (1 + math.Cos(math.Pi*progress))
	}
	return schedule.FinalFraction + (1-schedule.FinalFraction)*remaining
}
