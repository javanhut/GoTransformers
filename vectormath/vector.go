package vectormath

import (
	"fmt"
	"math"
)

type Vector []float64

func NewVector(size int) Vector {
	return make(Vector, size)
}

func CopyVector(vector Vector) Vector {
	result := NewVector(len(vector))
	copy(result, vector)
	return result
}

func checkSameSize(functionName string, first Vector, second Vector) {
	if len(first) != len(second) {
		panic(fmt.Sprintf("%s: first vector has %d values but second vector has %d values", functionName, len(first), len(second)))
	}
}

func checkNotEmpty(functionName string, vector Vector) {
	if len(vector) == 0 {
		panic(fmt.Sprintf("%s: vector is empty", functionName))
	}
}

func Add(first Vector, second Vector) Vector {
	checkSameSize("Add", first, second)
	result := NewVector(len(first))
	for i := range first {
		result[i] = first[i] + second[i]
	}
	return result
}

func Subtract(first Vector, second Vector) Vector {
	checkSameSize("Subtract", first, second)
	result := NewVector(len(first))
	for i := range first {
		result[i] = first[i] - second[i]
	}
	return result
}

func MultiplyEach(first Vector, second Vector) Vector {
	checkSameSize("MultiplyEach", first, second)
	result := NewVector(len(first))
	for i := range first {
		result[i] = first[i] * second[i]
	}
	return result
}

func Scale(vector Vector, amount float64) Vector {
	result := NewVector(len(vector))
	for i := range vector {
		result[i] = vector[i] * amount
	}
	return result
}

func DotProduct(first Vector, second Vector) float64 {
	checkSameSize("DotProduct", first, second)
	sum := 0.0
	for i := range first {
		sum += first[i] * second[i]
	}
	return sum
}

func Sum(vector Vector) float64 {
	sum := 0.0
	for _, value := range vector {
		sum += value
	}
	return sum
}

func Magnitude(vector Vector) float64 {
	return math.Sqrt(DotProduct(vector, vector))
}

func Max(vector Vector) float64 {
	return vector[IndexOfMax(vector)]
}

func IndexOfMax(vector Vector) int {
	checkNotEmpty("IndexOfMax", vector)
	largestIndex := 0
	for i, value := range vector {
		if value > vector[largestIndex] {
			largestIndex = i
		}
	}
	return largestIndex
}

func ApplyToEach(vector Vector, function func(float64) float64) Vector {
	result := NewVector(len(vector))
	for i, value := range vector {
		result[i] = function(value)
	}
	return result
}
