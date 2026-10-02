package activationfunction

import (
	"math"
	"testing"
)

func TestDerivativesMatchFiniteDifference(t *testing.T) {
	const h = 1e-6
	for name, a := range activationFns {
		for _, x := range []float64{-5, -1.3, -0.2, 0.3, 1.7, 4} {
			num := (a.Forward(x+h) - a.Forward(x-h)) / (2 * h)
			if got := a.Derivative(x); math.Abs(num-got) > 1e-6 {
				t.Errorf("%s'(%v) = %v, finite difference = %v", name, x, got, num)
			}
		}
	}
}

func TestNoNaNAtExtremes(t *testing.T) {
	for name, a := range activationFns {
		for _, x := range []float64{-1000, 1000} {
			if math.IsNaN(a.Forward(x)) || math.IsNaN(a.Derivative(x)) {
				t.Errorf("%s produced NaN at x=%v", name, x)
			}
		}
	}
}

func TestGet(t *testing.T) {
	for _, name := range []string{"ReLU", "Sigmoid", "Tanh", "LeakyReLU", "SiLU", "ELU", "GELU", "GELUExact", "Linear", "Softplus", "Mish", "SELU", "ReLU6", "ReLUSquared", "HardSigmoid", "HardSwish", "HardTanh", "Softsign", "QuickGELU"} {
		a, ok := Get(name)
		if !ok || a.Name != name {
			t.Errorf("Get(%q) = %+v, %v", name, a, ok)
		}
	}
	if _, ok := Get("nope"); ok {
		t.Error(`Get("nope") reported ok`)
	}
}

func TestClampedActivations(t *testing.T) {
	const h = 1e-6
	for _, a := range []Activation{ClampedSiLU(2), ClampedLinear(2)} {
		for _, x := range []float64{-5, -1.3, -0.2, 0.3, 1.7, 4} {
			num := (a.Forward(x+h) - a.Forward(x-h)) / (2 * h)
			if got := a.Derivative(x); math.Abs(num-got) > 1e-6 {
				t.Errorf("%s'(%v) = %v, finite difference = %v", a.Name, x, got, num)
			}
		}
	}
	if ClampedSiLU(2).Forward(100) != silu(2) {
		t.Error("ClampedSiLU(2) did not clamp 100 down to 2")
	}
	if ClampedLinear(2).Forward(-100) != -2 {
		t.Error("ClampedLinear(2) did not clamp -100 up to -2")
	}
}
