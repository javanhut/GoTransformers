package gpu

import (
	"fmt"
	"github.com/javanhut/GoTransformers/vectormath"
	"unsafe"
)

const defaultMinimumWorkForGPUWithResidentWeights = 1024 * 1024

const defaultMinimumWorkForGPUWithFewRowsTimesUntransposedWeights = 32 * 1024 * 1024

const defaultWeightCacheLimitBytes = 1 << 30

type weightCacheKey struct {
	address uintptr
	length  int
	rows    int
	columns int
}

type cachedWeights struct {
	buffer   gpuBuffer
	bytes    uint64
	version  uint64
	lastUsed uint64
	values   []float64
}

func keyFor(weights vectormath.Matrix) weightCacheKey {
	return weightCacheKey{
		address: uintptr(unsafe.Pointer(&weights.Values[0])),
		length:  len(weights.Values),
		rows:    weights.Rows,
		columns: weights.Columns,
	}
}

func (device *Device) forgetWeights(key weightCacheKey) {
	entry := device.weightCache[key]
	device.destroyBuffer(&entry.buffer)
	device.weightCacheBytes -= entry.bytes
	delete(device.weightCache, key)
	device.descriptorsNeedUpdate = true
}

func (device *Device) forgetAllWeights() {
	for key := range device.weightCache {
		device.forgetWeights(key)
	}
}

func (device *Device) forgetLeastRecentlyUsedWeights() bool {
	var oldestKey weightCacheKey
	oldestUse := ^uint64(0)
	found := false
	for key, entry := range device.weightCache {
		if entry.lastUsed < oldestUse {
			oldestKey = key
			oldestUse = entry.lastUsed
			found = true
		}
	}
	if found {
		device.forgetWeights(oldestKey)
	}
	return found
}

func (device *Device) uploadWeights(entry *cachedWeights) error {
	if !device.useStagingBuffers {
		writeValues(&entry.buffer, entry.values)
		return device.flushIfNeeded(&entry.buffer)
	}

	if err := device.makeBufferBigEnough(&device.secondStaging, entry.bytes, uploadBufferKind()); err != nil {
		return err
	}
	writeValues(&device.secondStaging, entry.values)
	if err := device.flushIfNeeded(&device.secondStaging); err != nil {
		return err
	}
	if err := checkResult("vkResetCommandBuffer", vkResetCommandBuffer(device.commandBuffer, 0)); err != nil {
		return err
	}
	beginInfo := &vkCommandBufferBeginInfo{
		structureType: vkStructureTypeCommandBufferBeginInfo,
		flags:         vkCommandBufferUsageOneTimeSubmitBit,
	}
	if err := checkResult("vkBeginCommandBuffer", vkBeginCommandBuffer(device.commandBuffer, unsafe.Pointer(beginInfo))); err != nil {
		return err
	}
	device.addCopy(&device.secondStaging, &entry.buffer, entry.bytes)
	device.addBarrier(vkPipelineStageTransferBit, vkPipelineStageComputeShaderBit, vkAccessTransferWriteBit, vkAccessShaderReadBit)
	if err := checkResult("vkEndCommandBuffer", vkEndCommandBuffer(device.commandBuffer)); err != nil {
		return err
	}
	return device.submitAndWait()
}

func (device *Device) residentWeightsFor(weights vectormath.Matrix) (*cachedWeights, error) {
	key := keyFor(weights)
	version := vectormath.WeightsVersion()
	device.weightUseCounter++

	entry, found := device.weightCache[key]
	if found && entry.version == version {
		entry.lastUsed = device.weightUseCounter
		device.weightCacheHits++
		return entry, nil
	}

	if !found {
		bytes := uint64(len(weights.Values)) * 4
		if bytes > device.maxStorageBufferRange {
			return nil, fmt.Errorf("weights need %d bytes but the GPU allows %d per buffer", bytes, device.maxStorageBufferRange)
		}
		if bytes > device.WeightCacheLimitBytes {
			return nil, nil
		}
		for device.weightCacheBytes+bytes > device.WeightCacheLimitBytes {
			if !device.forgetLeastRecentlyUsedWeights() {
				break
			}
		}
		buffer, err := device.createBuffer(bytes, device.inputBufferKind())
		if err != nil {
			return nil, err
		}
		entry = &cachedWeights{buffer: buffer, bytes: bytes, values: weights.Values}
		device.weightCache[key] = entry
		device.weightCacheBytes += bytes
	}

	if err := device.uploadWeights(entry); err != nil {
		device.forgetWeights(key)
		return nil, err
	}
	entry.version = version
	entry.lastUsed = device.weightUseCounter
	device.weightUploads++
	return entry, nil
}

func (device *Device) multiplyWithWeights(first vectormath.Matrix, weights vectormath.Matrix, rows int, inner int, columns int, weightsAreTransposed bool, multiplyOnCPU func(vectormath.Matrix, vectormath.Matrix) vectormath.Matrix) vectormath.Matrix {
	if !device.KeepWeightsOnGPU {
		return device.multiplyOrUseCPU(first, weights, rows, inner, columns, false, weightsAreTransposed, multiplyOnCPU)
	}
	work := rows * inner * columns
	minimumWork := device.MinimumWorkForGPUWithResidentWeights
	if rows <= largestRowsForFewRowsShader && !weightsAreTransposed {
		minimumWork = device.MinimumWorkForGPUWithFewRowsTimesUntransposedWeights
	}
	if work == 0 || work < minimumWork {
		return multiplyOnCPU(first, weights)
	}
	device.lock.Lock()
	defer device.lock.Unlock()
	if device.closed {
		return multiplyOnCPU(first, weights)
	}
	resident, err := device.residentWeightsFor(weights)
	if err != nil {
		device.lastError = err
		return multiplyOnCPU(first, weights)
	}
	result, err := device.multiplyOnGPU(first, weights, rows, inner, columns, false, weightsAreTransposed, resident)
	if err != nil {
		device.lastError = err
		return multiplyOnCPU(first, weights)
	}
	return result
}

func (device *Device) MatrixTimesTransposedWeights(first vectormath.Matrix, weights vectormath.Matrix) vectormath.Matrix {
	return device.multiplyWithWeights(first, weights, first.Rows, first.Columns, weights.Rows, true, vectormath.CPUBackend{}.MatrixTimesTransposed)
}

func (device *Device) MatrixTimesWeights(first vectormath.Matrix, weights vectormath.Matrix) vectormath.Matrix {
	return device.multiplyWithWeights(first, weights, first.Rows, first.Columns, weights.Columns, false, vectormath.CPUBackend{}.MatrixTimesMatrix)
}

func (device *Device) WeightUploads() int {
	device.lock.Lock()
	defer device.lock.Unlock()
	return device.weightUploads
}

func (device *Device) WeightCacheHits() int {
	device.lock.Lock()
	defer device.lock.Unlock()
	return device.weightCacheHits
}

func (device *Device) CachedWeightBytes() uint64 {
	device.lock.Lock()
	defer device.lock.Unlock()
	return device.weightCacheBytes
}

func (device *Device) ForgetCachedWeights() {
	device.lock.Lock()
	defer device.lock.Unlock()
	if !device.closed {
		device.forgetAllWeights()
	}
}
