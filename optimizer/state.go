package optimizer

import "fmt"

type State struct {
	Kind       string
	StepsTaken int
	Remembered map[string]map[string][]float64
}

type Resumable interface {
	Optimizer
	SaveState() State
	RestoreState(state State) error
}

func copyRemembered(remembered map[string][]float64) map[string][]float64 {
	copied := make(map[string][]float64, len(remembered))
	for name, values := range remembered {
		copied[name] = append([]float64(nil), values...)
	}
	return copied
}

func checkKind(state State, wantedKind string) error {
	if state.Kind != wantedKind {
		return fmt.Errorf("saved optimizer state is for %s, not %s", state.Kind, wantedKind)
	}
	return nil
}

func (sgd *SGD) SaveState() State {
	return State{Kind: "SGD"}
}

func (sgd *SGD) RestoreState(state State) error {
	return checkKind(state, "SGD")
}

func (sgd *SGDWithMomentum) SaveState() State {
	return State{Kind: "SGDWithMomentum", Remembered: map[string]map[string][]float64{
		"velocities": copyRemembered(sgd.velocities),
	}}
}

func (sgd *SGDWithMomentum) RestoreState(state State) error {
	if err := checkKind(state, "SGDWithMomentum"); err != nil {
		return err
	}
	sgd.velocities = copyRemembered(state.Remembered["velocities"])
	return nil
}

func (adam *Adam) SaveState() State {
	return State{Kind: "Adam", StepsTaken: adam.stepsTaken, Remembered: map[string]map[string][]float64{
		"averageGradients":        copyRemembered(adam.averageGradients),
		"averageSquaredGradients": copyRemembered(adam.averageSquaredGradients),
	}}
}

func (adam *Adam) RestoreState(state State) error {
	if err := checkKind(state, "Adam"); err != nil {
		return err
	}
	adam.stepsTaken = state.StepsTaken
	adam.averageGradients = copyRemembered(state.Remembered["averageGradients"])
	adam.averageSquaredGradients = copyRemembered(state.Remembered["averageSquaredGradients"])
	return nil
}

func (adamW *AdamW) SaveState() State {
	return State{Kind: "AdamW", StepsTaken: adamW.stepsTaken, Remembered: map[string]map[string][]float64{
		"averageGradients":        copyRemembered(adamW.averageGradients),
		"averageSquaredGradients": copyRemembered(adamW.averageSquaredGradients),
	}}
}

func (adamW *AdamW) RestoreState(state State) error {
	if err := checkKind(state, "AdamW"); err != nil {
		return err
	}
	adamW.stepsTaken = state.StepsTaken
	adamW.averageGradients = copyRemembered(state.Remembered["averageGradients"])
	adamW.averageSquaredGradients = copyRemembered(state.Remembered["averageSquaredGradients"])
	return nil
}

func (muon *Muon) SaveState() State {
	adamWState := muon.AdamW.SaveState()
	return State{Kind: "Muon", StepsTaken: adamWState.StepsTaken, Remembered: map[string]map[string][]float64{
		"momentumBuffers":               copyRemembered(muon.momentumBuffers),
		"adamW.averageGradients":        adamWState.Remembered["averageGradients"],
		"adamW.averageSquaredGradients": adamWState.Remembered["averageSquaredGradients"],
	}}
}

func (muon *Muon) RestoreState(state State) error {
	if err := checkKind(state, "Muon"); err != nil {
		return err
	}
	muon.momentumBuffers = copyRemembered(state.Remembered["momentumBuffers"])
	return muon.AdamW.RestoreState(State{Kind: "AdamW", StepsTaken: state.StepsTaken, Remembered: map[string]map[string][]float64{
		"averageGradients":        state.Remembered["adamW.averageGradients"],
		"averageSquaredGradients": state.Remembered["adamW.averageSquaredGradients"],
	}})
}
