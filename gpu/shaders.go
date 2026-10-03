package gpu

import _ "embed"

//go:generate glslc --target-env=vulkan1.0 -O shaders/matrixmultiply.comp -o shaders/matrixmultiply.spv
//go:generate glslc --target-env=vulkan1.0 -O shaders/fewrows.comp -o shaders/fewrows.spv

//go:embed shaders/matrixmultiply.spv
var matrixMultiplyShader []byte

//go:embed shaders/fewrows.spv
var fewRowsShader []byte
