package activationfunction

import "math"

type ActivationFn func(float64) float64

// Activation pairs an activation function with its derivative.
// Derivative takes the pre-activation input x, not the output Forward(x).
type Activation struct {
	Name       string
	Forward    ActivationFn
	Derivative ActivationFn
}

func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func reluDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

var ReLU = Activation{
	Name:       "ReLU",
	Forward:    relu,
	Derivative: reluDerivative,
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	s := sigmoid(x)
	return s * (1 - s)
}

var Sigmoid = Activation{
	Name:       "Sigmoid",
	Forward:    sigmoid,
	Derivative: sigmoidDerivative,
}

func tanh(x float64) float64 {
	return math.Tanh(x)
}
func tanhDerivative(x float64) float64 {
	t := tanh(x)
	return 1 - t*t
}

var Tanh = Activation{
	Name:       "Tanh",
	Forward:    tanh,
	Derivative: tanhDerivative,
}

func leakyReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0.01 * x
}
func leakyReLUDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0.01
}

var LeakyReLU = Activation{
	Name:       "LeakyReLU",
	Forward:    leakyReLU,
	Derivative: leakyReLUDerivative,
}

func elu(x float64) float64 {
	if x > 0 {
		return x
	}
	return math.Exp(x) - 1
}
func eluDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return math.Exp(x)
}

var ELU = Activation{
	Name:       "ELU",
	Forward:    elu,
	Derivative: eluDerivative,
}

func silu(x float64) float64 {
	s := sigmoid(x)
	return x * s
}
func siluDerivative(x float64) float64 {
	s := sigmoid(x)
	return s + x*s*(1-s)
}

var SiLU = Activation{
	Name:       "SiLU",
	Forward:    silu,
	Derivative: siluDerivative,
}

// GELU uses the tanh approximation (as in GPT-2/BERT); see GELUExact for
// the exact 0.5*x*(1+erf(x/sqrt(2))) form.
var geluC = math.Sqrt(2 / math.Pi)

const geluK = 0.044715

func gelu(x float64) float64 {
	return 0.5 * x * (1 + math.Tanh(geluC*(x+geluK*x*x*x)))
}

func geluDerivative(x float64) float64 {
	u := geluC * (x + geluK*x*x*x)
	t := math.Tanh(u)
	du := geluC * (1 + 3*geluK*x*x)
	return 0.5*(1+t) + 0.5*x*(1-t*t)*du
}

var GELU = Activation{
	Name:       "GELU",
	Forward:    gelu,
	Derivative: geluDerivative,
}

// geluExact is x * Phi(x), where Phi is the standard normal CDF.
func geluExact(x float64) float64 {
	return 0.5 * x * (1 + math.Erf(x/math.Sqrt2))
}

// geluExactDerivative is Phi(x) + x * phi(x), where phi is the standard normal PDF.
func geluExactDerivative(x float64) float64 {
	cdf := 0.5 * (1 + math.Erf(x/math.Sqrt2))
	pdf := math.Exp(-0.5*x*x) / math.Sqrt(2*math.Pi)
	return cdf + x*pdf
}

var GELUExact = Activation{
	Name:       "GELUExact",
	Forward:    geluExact,
	Derivative: geluExactDerivative,
}

func linear(x float64) float64 {
	return x
}

func linearDerivative(x float64) float64 {
	return 1
}

var Linear = Activation{
	Name:       "Linear",
	Forward:    linear,
	Derivative: linearDerivative,
}

var activationFns = func() map[string]Activation {
	m := make(map[string]Activation)
	for _, a := range []Activation{ReLU, Sigmoid, Tanh, LeakyReLU, SiLU, ELU, GELU, GELUExact, Linear, Softplus, Mish, SELU, ReLU6, ReLUSquared, HardSigmoid, HardSwish, HardTanh, Softsign, QuickGELU} {
		m[a.Name] = a
	}
	return m
}()

// Get returns the activation registered under name.
func Get(name string) (Activation, bool) {
	a, ok := activationFns[name]
	return a, ok
}
