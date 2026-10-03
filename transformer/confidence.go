package transformer

import (
	"fmt"
	"github.com/javanhut/GoTransformers/activationfunction"
	"math"
	"slices"
)

type AnswerScore struct {
	TokenProbabilities     []float64
	LogProbability         float64
	Probability            float64
	AverageTokenConfidence float64
	LowestTokenProbability float64
}

func scoreFromProbabilities(tokenProbabilities []float64) AnswerScore {
	score := AnswerScore{TokenProbabilities: tokenProbabilities, LowestTokenProbability: 1}
	for _, probability := range tokenProbabilities {
		score.LogProbability += math.Log(math.Max(probability, 1e-300))
		if probability < score.LowestTokenProbability {
			score.LowestTokenProbability = probability
		}
	}
	score.Probability = math.Exp(score.LogProbability)
	if len(tokenProbabilities) > 0 {
		score.AverageTokenConfidence = math.Exp(score.LogProbability / float64(len(tokenProbabilities)))
	}
	return score
}

func (model *Model) ScoreAnswer(promptIDs []int, answerIDs []int) AnswerScore {
	if len(promptIDs) == 0 || len(answerIDs) == 0 {
		panic(fmt.Sprintf("Model.ScoreAnswer: needs at least 1 prompt token and 1 answer token, got %d and %d", len(promptIDs), len(answerIDs)))
	}
	example := Example{PromptIDs: promptIDs, AnswerIDs: answerIDs}
	tokenIDs := example.tokenIDs()
	model.checkTokenIDs(tokenIDs)
	scores := model.Forward(tokenIDs[:len(tokenIDs)-1])

	tokenProbabilities := make([]float64, len(answerIDs))
	for i, answerID := range answerIDs {
		predictingRow := len(promptIDs) - 1 + i
		tokenProbabilities[i] = activationfunction.Softmax(scores.Row(predictingRow))[answerID]
	}
	return scoreFromProbabilities(tokenProbabilities)
}

type GenerationOptions struct {
	MaximumNewTokens int
	Temperature      float64
	StopTokenIDs     []int
	ConfidenceTarget float64
	AbstainTokenIDs  []int
	Sampling         SamplingOptions
	Constraint       TokenConstraint
}

func (options GenerationOptions) samplingOptions() SamplingOptions {
	sampling := options.Sampling
	if sampling.Temperature == 0 {
		sampling.Temperature = options.Temperature
	}
	return sampling
}

type GeneratedAnswer struct {
	TokenIDs            []int
	Score               AnswerScore
	Abstained           bool
	RejectedTokenIDs    []int
	ConstraintSatisfied bool
}

func isStopToken(tokenID int, stopTokenIDs []int) bool {
	return slices.Contains(stopTokenIDs, tokenID)
}

func (model *Model) Answer(promptIDs []int, options GenerationOptions) GeneratedAnswer {
	if len(promptIDs) == 0 {
		panic("Model.Answer: the prompt needs at least 1 token")
	}
	if options.MaximumNewTokens < 1 {
		panic(fmt.Sprintf("Model.Answer: MaximumNewTokens must be at least 1, got %d", options.MaximumNewTokens))
	}
	if options.ConfidenceTarget < 0 || options.ConfidenceTarget > 1 {
		panic(fmt.Sprintf("Model.Answer: ConfidenceTarget must be between 0 and 1, got %v", options.ConfidenceTarget))
	}

	sampler := NewSampler(options.samplingOptions(), options.Constraint)
	sampler.StopTokenIDs = options.StopTokenIDs
	sampler.RememberPrompt(promptIDs)
	if options.Constraint != nil {
		options.Constraint.Restart()
	}

	model.StartGenerating()
	scores := model.Feed(promptIDs)
	var generatedIDs []int
	var tokenProbabilities []float64
	for len(generatedIDs) < options.MaximumNewTokens {
		nextID, found := sampler.PickToken(scores)
		if !found {
			break
		}
		tokenProbabilities = append(tokenProbabilities, activationfunction.Softmax(scores)[nextID])
		if isStopToken(nextID, options.StopTokenIDs) {
			break
		}
		if err := sampler.AcceptToken(nextID); err != nil {
			panic(fmt.Sprintf("Model.Answer: the sampler picked a token its constraint refuses: %v", err))
		}
		generatedIDs = append(generatedIDs, nextID)
		if len(generatedIDs) < options.MaximumNewTokens {
			scores = model.NextTokenScores(nextID)
		}
	}

	answer := GeneratedAnswer{TokenIDs: generatedIDs, Score: scoreFromProbabilities(tokenProbabilities)}
	answer.ConstraintSatisfied = options.Constraint == nil || options.Constraint.IsComplete()
	if options.ConfidenceTarget > 0 && answer.Score.Probability < options.ConfidenceTarget {
		answer.Abstained = true
		answer.RejectedTokenIDs = generatedIDs
		answer.TokenIDs = append([]int(nil), options.AbstainTokenIDs...)
	}
	return answer
}
