package activationfunction

import (
	"fmt"
	"math"
)

func softplus(x float64) float64 {
	if x > 20 {
		return x
	}
	if x < -20 {
		return math.Exp(x)
	}
	return math.Log1p(math.Exp(x))
}

func softplusDerivative(x float64) float64 {
	return sigmoid(x)
}

var Softplus = Activation{
	Name:       "Softplus",
	Forward:    softplus,
	Derivative: softplusDerivative,
}

func mish(x float64) float64 {
	return x * math.Tanh(softplus(x))
}

func mishDerivative(x float64) float64 {
	t := math.Tanh(softplus(x))
	return t + x*(1-t*t)*sigmoid(x)
}

var Mish = Activation{
	Name:       "Mish",
	Forward:    mish,
	Derivative: mishDerivative,
}

const seluScale = 1.0507009873554805
const seluAlpha = 1.6732632423543772

func selu(x float64) float64 {
	if x > 0 {
		return seluScale * x
	}
	return seluScale * seluAlpha * (math.Exp(x) - 1)
}

func seluDerivative(x float64) float64 {
	if x > 0 {
		return seluScale
	}
	return seluScale * seluAlpha * math.Exp(x)
}

var SELU = Activation{
	Name:       "SELU",
	Forward:    selu,
	Derivative: seluDerivative,
}

func relu6(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 6 {
		return 6
	}
	return x
}

func relu6Derivative(x float64) float64 {
	if x > 0 && x < 6 {
		return 1
	}
	return 0
}

var ReLU6 = Activation{
	Name:       "ReLU6",
	Forward:    relu6,
	Derivative: relu6Derivative,
}

func reluSquared(x float64) float64 {
	if x > 0 {
		return x * x
	}
	return 0
}

func reluSquaredDerivative(x float64) float64 {
	if x > 0 {
		return 2 * x
	}
	return 0
}

var ReLUSquared = Activation{
	Name:       "ReLUSquared",
	Forward:    reluSquared,
	Derivative: reluSquaredDerivative,
}

func hardSigmoid(x float64) float64 {
	if x <= -3 {
		return 0
	}
	if x >= 3 {
		return 1
	}
	return x/6 + 0.5
}

func hardSigmoidDerivative(x float64) float64 {
	if x > -3 && x < 3 {
		return 1.0 / 6
	}
	return 0
}

var HardSigmoid = Activation{
	Name:       "HardSigmoid",
	Forward:    hardSigmoid,
	Derivative: hardSigmoidDerivative,
}

func hardSwish(x float64) float64 {
	return x * hardSigmoid(x)
}

func hardSwishDerivative(x float64) float64 {
	if x <= -3 {
		return 0
	}
	if x >= 3 {
		return 1
	}
	return (2*x + 3) / 6
}

var HardSwish = Activation{
	Name:       "HardSwish",
	Forward:    hardSwish,
	Derivative: hardSwishDerivative,
}

func hardTanh(x float64) float64 {
	if x < -1 {
		return -1
	}
	if x > 1 {
		return 1
	}
	return x
}

func hardTanhDerivative(x float64) float64 {
	if x > -1 && x < 1 {
		return 1
	}
	return 0
}

var HardTanh = Activation{
	Name:       "HardTanh",
	Forward:    hardTanh,
	Derivative: hardTanhDerivative,
}

func softsign(x float64) float64 {
	return x / (1 + math.Abs(x))
}

func softsignDerivative(x float64) float64 {
	bottom := 1 + math.Abs(x)
	return 1 / (bottom * bottom)
}

var Softsign = Activation{
	Name:       "Softsign",
	Forward:    softsign,
	Derivative: softsignDerivative,
}

const quickGELUSharpness = 1.702

func quickGELU(x float64) float64 {
	return x * sigmoid(quickGELUSharpness*x)
}

func quickGELUDerivative(x float64) float64 {
	s := sigmoid(quickGELUSharpness * x)
	return s + quickGELUSharpness*x*s*(1-s)
}

var QuickGELU = Activation{
	Name:       "QuickGELU",
	Forward:    quickGELU,
	Derivative: quickGELUDerivative,
}

func ClampedSiLU(limit float64) Activation {
	return Activation{
		Name: fmt.Sprintf("ClampedSiLU(%g)", limit),
		Forward: func(x float64) float64 {
			if x > limit {
				return silu(limit)
			}
			return silu(x)
		},
		Derivative: func(x float64) float64 {
			if x > limit {
				return 0
			}
			return siluDerivative(x)
		},
	}
}

func ClampedLinear(limit float64) Activation {
	return Activation{
		Name: fmt.Sprintf("ClampedLinear(%g)", limit),
		Forward: func(x float64) float64 {
			if x > limit {
				return limit
			}
			if x < -limit {
				return -limit
			}
			return x
		},
		Derivative: func(x float64) float64 {
			if x > limit || x < -limit {
				return 0
			}
			return 1
		},
	}
}
