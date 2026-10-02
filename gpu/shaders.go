package gpu

import _ "embed"

//go:generate glslc --target-env=vulkan1.0 -O shaders/matrixmultiply.comp -o shaders/matrixmultiply.spv

//go:embed shaders/matrixmultiply.spv
var matrixMultiplyShader []byte
