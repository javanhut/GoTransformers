package training

import (
	"fmt"
	"math"
	"time"
)

type Trainer interface {
	TrainBatch(sequences [][]int) float64
	EvaluationLoss(sequences [][]int) float64
	LearningRate() float64
	LastGradientNorm() float64
	StepsTaken() int
	SaveModel(path string) error
}

type StepReport struct {
	Step         int
	Loss         float64
	LearningRate float64
	GradientNorm float64
	StepTime     time.Duration
}

type EvaluationReport struct {
	Step                   int
	EvaluationLoss         float64
	BestEvaluationLoss     float64
	BestStep               int
	IsBest                 bool
	SavedBestModel         bool
	StepsWithoutImproving  int
	AverageTrainingLoss    float64
	TrainingStepsSinceLast int
}

type Loop struct {
	Trainer       Trainer
	NumberOfSteps int
	NextBatch     func() [][]int

	EvaluationSequences                  [][]int
	EvaluationBatchSize                  int
	EvaluateEvery                        int
	BestModelPath                        string
	StopAfterEvaluationsWithoutImproving int

	OnStep       func(report StepReport)
	OnEvaluation func(report EvaluationReport)
}

type Result struct {
	StepsTaken         int
	LastLoss           float64
	BestEvaluationLoss float64
	BestStep           int
	StoppedEarly       bool
}

func (loop Loop) check() error {
	if loop.Trainer == nil {
		return fmt.Errorf("training.Loop needs a Trainer")
	}
	if loop.NextBatch == nil {
		return fmt.Errorf("training.Loop needs a NextBatch function")
	}
	if loop.NumberOfSteps < 1 {
		return fmt.Errorf("training.Loop needs NumberOfSteps of at least 1, got %d", loop.NumberOfSteps)
	}
	if loop.EvaluateEvery < 0 || loop.EvaluationBatchSize < 0 || loop.StopAfterEvaluationsWithoutImproving < 0 {
		return fmt.Errorf("EvaluateEvery, EvaluationBatchSize and StopAfterEvaluationsWithoutImproving can't be negative")
	}
	if loop.EvaluateEvery > 0 && len(loop.EvaluationSequences) == 0 {
		return fmt.Errorf("EvaluateEvery is %d but there are no EvaluationSequences", loop.EvaluateEvery)
	}
	if loop.BestModelPath != "" && loop.EvaluateEvery == 0 {
		return fmt.Errorf("BestModelPath needs EvaluateEvery above 0, the best model is chosen by evaluation loss")
	}
	return nil
}

func EvaluationLossInBatches(trainer Trainer, sequences [][]int, batchSize int) float64 {
	if batchSize < 1 {
		batchSize = len(sequences)
	}
	totalLoss := 0.0
	for start := 0; start < len(sequences); start += batchSize {
		end := min(start+batchSize, len(sequences))
		totalLoss += trainer.EvaluationLoss(sequences[start:end]) * float64(end-start)
	}
	return totalLoss / float64(len(sequences))
}

func (loop Loop) Run() (Result, error) {
	if err := loop.check(); err != nil {
		return Result{}, err
	}
	result := Result{BestEvaluationLoss: math.Inf(1), BestStep: -1}
	evaluationsWithoutImproving := 0
	trainingLossSinceEvaluation := 0.0
	stepsSinceEvaluation := 0
	for stepNumber := 1; stepNumber <= loop.NumberOfSteps; stepNumber++ {
		learningRate := loop.Trainer.LearningRate()
		started := time.Now()
		loss := loop.Trainer.TrainBatch(loop.NextBatch())
		stepTime := time.Since(started)
		if math.IsNaN(loss) || math.IsInf(loss, 0) {
			return result, fmt.Errorf("step %d: the training loss became %v", stepNumber, loss)
		}
		result.StepsTaken = stepNumber
		result.LastLoss = loss
		trainingLossSinceEvaluation += loss
		stepsSinceEvaluation++
		if loop.OnStep != nil {
			loop.OnStep(StepReport{Step: stepNumber, Loss: loss, LearningRate: learningRate, GradientNorm: loop.Trainer.LastGradientNorm(), StepTime: stepTime})
		}

		isEvaluationStep := loop.EvaluateEvery > 0 && (stepNumber%loop.EvaluateEvery == 0 || stepNumber == loop.NumberOfSteps)
		if !isEvaluationStep {
			continue
		}
		report := EvaluationReport{
			Step:                   stepNumber,
			EvaluationLoss:         EvaluationLossInBatches(loop.Trainer, loop.EvaluationSequences, loop.EvaluationBatchSize),
			AverageTrainingLoss:    trainingLossSinceEvaluation / float64(stepsSinceEvaluation),
			TrainingStepsSinceLast: stepsSinceEvaluation,
		}
		trainingLossSinceEvaluation = 0
		stepsSinceEvaluation = 0
		if report.EvaluationLoss < result.BestEvaluationLoss {
			result.BestEvaluationLoss = report.EvaluationLoss
			result.BestStep = stepNumber
			report.IsBest = true
			evaluationsWithoutImproving = 0
			if loop.BestModelPath != "" {
				if err := loop.Trainer.SaveModel(loop.BestModelPath); err != nil {
					return result, fmt.Errorf("step %d: saving the best model: %w", stepNumber, err)
				}
				report.SavedBestModel = true
			}
		} else {
			evaluationsWithoutImproving++
		}
		report.BestEvaluationLoss = result.BestEvaluationLoss
		report.BestStep = result.BestStep
		report.StepsWithoutImproving = stepNumber - result.BestStep
		if loop.OnEvaluation != nil {
			loop.OnEvaluation(report)
		}
		if loop.StopAfterEvaluationsWithoutImproving > 0 && evaluationsWithoutImproving >= loop.StopAfterEvaluationsWithoutImproving {
			result.StoppedEarly = true
			return result, nil
		}
	}
	return result, nil
}
