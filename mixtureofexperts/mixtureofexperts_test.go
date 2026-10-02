package mixtureofexperts

import (
	"math"
	"reflect"
	"testing"
	"transformer/gradientcheck"
	"transformer/vectormath"
)

func TestGradients(t *testing.T) {
	tokenIDs := []int{3, 7, 1, 12, 5}
	versions := map[string]*MixtureOfExperts{
		"routed, no shared":   NewMixtureOfExperts("moe", 4, 6, 0, 4, 2),
		"routed with shared":  NewMixtureOfExperts("moe", 4, 6, 1, 4, 2),
		"hashed, no shared":   hashed(NewMixtureOfExperts("moe", 4, 6, 0, 4, 2)),
		"hashed with shared":  hashed(NewMixtureOfExperts("moe", 4, 6, 1, 4, 2)),
		"routed, one per row": NewMixtureOfExperts("moe", 4, 6, 1, 5, 1),
		"clamped":             NewClampedMixtureOfExperts("moe", 4, 6, 1, 4, 2, 0.5),
	}
	for name, mixture := range versions {
		forward := func(inputs vectormath.Matrix) vectormath.Matrix {
			return mixture.Forward(inputs, tokenIDs)
		}
		inputs := vectormath.NewRandomMatrix(len(tokenIDs), 4, -1, 1)
		for _, problem := range gradientcheck.Compare(forward, mixture.Backward, mixture.Parameters(), inputs) {
			t.Errorf("%s: %s", name, problem)
		}
	}
}

func hashed(mixture *MixtureOfExperts) *MixtureOfExperts {
	mixture.UseHashRouting = true
	return mixture
}

func TestEachTokenUsesExpertsPerToken(t *testing.T) {
	for _, mixture := range []*MixtureOfExperts{NewMixtureOfExperts("moe", 4, 6, 1, 6, 3), hashed(NewMixtureOfExperts("moe", 4, 6, 1, 6, 3))} {
		tokenIDs := []int{0, 1, 2, 3, 4, 5, 6, 7}
		mixture.Forward(vectormath.NewRandomMatrix(len(tokenIDs), 4, -1, 1), tokenIDs)
		for token, chosen := range mixture.lastChosenExperts {
			if len(chosen) != 3 {
				t.Errorf("token %d used %d experts, want 3", token, len(chosen))
			}
			seen := map[int]bool{}
			for _, expert := range chosen {
				if seen[expert] {
					t.Errorf("token %d used expert %d twice", token, expert)
				}
				seen[expert] = true
			}
			gateTotal := 0.0
			for _, gate := range mixture.lastGates[token] {
				gateTotal += gate
			}
			if math.Abs(gateTotal-1) > 1e-12 {
				t.Errorf("token %d gates add up to %v, want 1", token, gateTotal)
			}
		}
		totalLoad := 0
		for _, load := range mixture.LastExpertLoad() {
			totalLoad += load
		}
		if totalLoad != len(tokenIDs)*3 {
			t.Errorf("total load %d, want %d", totalLoad, len(tokenIDs)*3)
		}
	}
}

func TestHashRoutingDependsOnlyOnTokenID(t *testing.T) {
	mixture := hashed(NewMixtureOfExperts("moe", 4, 6, 0, 8, 2))
	mixture.Forward(vectormath.NewRandomMatrix(3, 4, -1, 1), []int{42, 9, 42})
	first := mixture.lastChosenExperts[0]
	if !reflect.DeepEqual(first, mixture.lastChosenExperts[2]) {
		t.Errorf("token 42 went to %v and then %v", first, mixture.lastChosenExperts[2])
	}
	mixture.Forward(vectormath.NewRandomMatrix(1, 4, -1, 1), []int{42})
	if !reflect.DeepEqual(first, mixture.lastChosenExperts[0]) {
		t.Errorf("token 42 went to %v with other inputs but %v before", mixture.lastChosenExperts[0], first)
	}
}

func TestOneRowMatchesManyRows(t *testing.T) {
	for name, mixture := range map[string]*MixtureOfExperts{
		"routed": NewMixtureOfExperts("moe", 4, 6, 1, 4, 2),
		"hashed": hashed(NewMixtureOfExperts("moe", 4, 6, 1, 4, 2)),
	} {
		tokenIDs := []int{4, 8, 15, 16}
		inputs := vectormath.NewRandomMatrix(len(tokenIDs), 4, -1, 1)
		allRows := mixture.Forward(inputs, tokenIDs)
		for row := 0; row < inputs.Rows; row++ {
			oneRow := mixture.Forward(vectormath.MatrixFromRows([]vectormath.Vector{inputs.Row(row)}), []int{tokenIDs[row]})
			for i, value := range oneRow.Row(0) {
				if math.Abs(value-allRows.Get(row, i)) > 1e-12 {
					t.Errorf("%s row %d value %d: one row %v, many rows %v", name, row, i, value, allRows.Get(row, i))
				}
			}
		}
	}
}

func imbalance(load []int) int {
	smallest := load[0]
	largest := load[0]
	for _, value := range load {
		if value < smallest {
			smallest = value
		}
		if value > largest {
			largest = value
		}
	}
	return largest - smallest
}

func TestUpdateBalanceEvensOutLoad(t *testing.T) {
	mixture := NewMixtureOfExperts("moe", 4, 6, 0, 4, 1)
	mixture.RouterLayer.Biases[0] = 3
	inputs := vectormath.NewRandomMatrix(64, 4, -1, 1)

	mixture.Forward(inputs, nil)
	startingLoad := mixture.LastExpertLoad()
	mixture.UpdateBalance(0.05)
	if mixture.BalanceBiases[0] >= 0 {
		t.Errorf("expert 0 is overloaded with %v but its bias went to %v", startingLoad, mixture.BalanceBiases[0])
	}
	for expert := 1; expert < 4; expert++ {
		if startingLoad[expert] < 16 && mixture.BalanceBiases[expert] <= 0 {
			t.Errorf("expert %d is under-loaded with %v but its bias went to %v", expert, startingLoad, mixture.BalanceBiases[expert])
		}
	}

	for step := 0; step < 300; step++ {
		mixture.Forward(inputs, nil)
		mixture.UpdateBalance(0.05)
	}
	totalImbalance := 0
	for step := 0; step < 50; step++ {
		mixture.Forward(inputs, nil)
		totalImbalance += imbalance(mixture.LastExpertLoad())
		mixture.UpdateBalance(0.05)
	}
	averageImbalance := float64(totalImbalance) / 50
	if averageImbalance > float64(imbalance(startingLoad))/2 {
		t.Errorf("load started at %v and the gap between busiest and quietest expert still averages %v", startingLoad, averageImbalance)
	}
}

func TestBalanceBiasesDoNotChangeGates(t *testing.T) {
	mixture := NewMixtureOfExperts("moe", 4, 6, 0, 3, 3)
	inputs := vectormath.NewRandomMatrix(2, 4, -1, 1)
	before := mixture.Forward(inputs, nil)
	mixture.BalanceBiases[1] = 100
	after := mixture.Forward(inputs, nil)
	for i := range before.Values {
		if math.Abs(before.Values[i]-after.Values[i]) > 1e-12 {
			t.Fatalf("with every expert chosen, a balance bias changed output %d from %v to %v", i, before.Values[i], after.Values[i])
		}
	}
}

func expectPanic(t *testing.T, name string, function func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s did not panic", name)
		}
	}()
	function()
}

func TestMistakesPanic(t *testing.T) {
	expectPanic(t, "more experts per token than experts", func() { NewMixtureOfExperts("moe", 4, 6, 0, 2, 3) })
	expectPanic(t, "no routed experts", func() { NewMixtureOfExperts("moe", 4, 6, 1, 0, 1) })
	expectPanic(t, "hash routing without token IDs", func() {
		hashed(NewMixtureOfExperts("moe", 4, 6, 0, 4, 2)).Forward(vectormath.NewMatrix(2, 4), nil)
	})
	expectPanic(t, "wrong vector size", func() { NewMixtureOfExperts("moe", 4, 6, 0, 4, 2).Forward(vectormath.NewMatrix(2, 5), nil) })
	expectPanic(t, "Backward before Forward", func() { NewMixtureOfExperts("moe", 4, 6, 0, 4, 2).Backward(vectormath.NewMatrix(2, 4)) })
}
