package transformer

import (
	"fmt"
	"github.com/javanhut/GoTransformers/optimizer"
)

type TrainerOptions struct {
	Schedule            optimizer.LearningRateSchedule
	MaximumGradientNorm float64
}

func (options TrainerOptions) Check() error {
	if err := options.Schedule.Check(); err != nil {
		return err
	}
	if options.MaximumGradientNorm < 0 {
		return fmt.Errorf("MaximumGradientNorm can't be negative (0 means no clipping), got %v", options.MaximumGradientNorm)
	}
	return nil
}

type Trainer struct {
	Model     *Model
	Optimizer optimizer.Optimizer
	Options   TrainerOptions

	peakLearningRates []float64
	stepsTaken        int
	lastGradientNorm  float64
}

func NewTrainer(model *Model, chosenOptimizer optimizer.Optimizer, options TrainerOptions) (*Trainer, error) {
	if err := options.Check(); err != nil {
		return nil, err
	}
	return &Trainer{
		Model:             model,
		Optimizer:         chosenOptimizer,
		Options:           options,
		peakLearningRates: optimizer.LearningRates(chosenOptimizer),
	}, nil
}

func (trainer *Trainer) TrainOnExamples(examples []Example) float64 {
	model := trainer.Model
	model.checkCanTrain("Trainer.TrainOnExamples", examples)
	loss := model.computeBatchGradients(examples)
	trainable := model.TrainableParameters()
	trainer.lastGradientNorm = optimizer.ClipGradients(trainable, trainer.Options.MaximumGradientNorm)
	optimizer.ScaleLearningRates(trainer.Optimizer, trainer.peakLearningRates, trainer.Options.Schedule.FractionAt(trainer.stepsTaken))
	trainer.Optimizer.Update(trainable)
	model.UpdateExpertBalance()
	trainer.stepsTaken++
	return loss
}

func (trainer *Trainer) TrainBatch(sequences [][]int) float64 {
	examples := make([]Example, len(sequences))
	for i, sequence := range sequences {
		examples[i] = Example{AnswerIDs: sequence}
	}
	return trainer.TrainOnExamples(examples)
}

func (trainer *Trainer) EvaluationLoss(sequences [][]int) float64 {
	if len(sequences) == 0 {
		panic("Trainer.EvaluationLoss: no sequences given")
	}
	totalLoss := 0.0
	for _, sequence := range sequences {
		totalLoss += trainer.Model.Loss(sequence)
	}
	return totalLoss / float64(len(sequences))
}

func (trainer *Trainer) LearningRate() float64 {
	return trainer.peakLearningRates[0] * trainer.Options.Schedule.FractionAt(trainer.stepsTaken)
}

func (trainer *Trainer) LastGradientNorm() float64 {
	return trainer.lastGradientNorm
}

func (trainer *Trainer) StepsTaken() int {
	return trainer.stepsTaken
}

func (trainer *Trainer) SetStepsTaken(stepsTaken int) {
	trainer.stepsTaken = stepsTaken
}

func (trainer *Trainer) SaveModel(path string) error {
	return trainer.Model.Save(path)
}
