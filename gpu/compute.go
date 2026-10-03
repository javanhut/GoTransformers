package gpu

import (
	"fmt"
	"math"
	"runtime"
	"unsafe"
)

type Buffer struct {
	device         *Device
	numberOfFloats int
	storage        gpuBuffer
	staging        gpuBuffer
	freed          bool
}

type Program struct {
	device              *Device
	name                string
	numberOfBuffers     int
	pushConstantWords   int
	shaderModule        uint64
	descriptorSetLayout uint64
	pipelineLayout      uint64
	pipeline            uint64
	freed               bool
}

type pendingDownload struct {
	buffer *Buffer
	values []float64
}

type Recorder struct {
	device           *Device
	commandPool      uint64
	commandBuffer    uintptr
	fence            uint64
	descriptorPools  []uint64
	poolsUsed        int
	setsLeftInPool   int
	recording        bool
	downloads        []pendingDownload
	dispatchesInStep int
	freed            bool

	measuringTime bool
	queryPool     uint64
	dispatchNames []string
	dispatchTimes []DispatchTime
}

type DispatchTime struct {
	ProgramName string
	Seconds     float64
}

const largestTimedDispatches = 8191

const descriptorSetsPerPool = 1024

const largestBindingsPerProgram = 16

func Float(value float64) uint32 {
	return math.Float32bits(float32(value))
}

func GroupsFor(numberOfThreads int, threadsPerGroup int) (uint32, uint32) {
	groups := max((numberOfThreads+threadsPerGroup-1)/threadsPerGroup, 1)
	groupsAcross := min(groups, 65535)
	groupsDown := (groups + groupsAcross - 1) / groupsAcross
	return uint32(groupsAcross), uint32(groupsDown)
}

func (device *Device) storageBufferKind() bufferKind {
	usage := uint32(vkBufferUsageStorageBufferBit | vkBufferUsageTransferSourceBit | vkBufferUsageTransferDestinationBit)
	if device.useStagingBuffers {
		return bufferKind{usage: usage, requiredFlags: vkMemoryPropertyDeviceLocalBit}
	}
	return bufferKind{
		usage:          usage,
		requiredFlags:  vkMemoryPropertyHostVisibleBit,
		preferredFlags: vkMemoryPropertyDeviceLocalBit | vkMemoryPropertyHostCoherentBit | vkMemoryPropertyHostCachedBit,
	}
}

func stagingBufferKind() bufferKind {
	return bufferKind{
		usage:          vkBufferUsageTransferSourceBit | vkBufferUsageTransferDestinationBit,
		requiredFlags:  vkMemoryPropertyHostVisibleBit,
		preferredFlags: vkMemoryPropertyHostCoherentBit | vkMemoryPropertyHostCachedBit,
	}
}

func (device *Device) NewBuffer(numberOfFloats int) (*Buffer, error) {
	if numberOfFloats < 1 {
		return nil, fmt.Errorf("a GPU buffer needs at least 1 value, got %d", numberOfFloats)
	}
	size := uint64(numberOfFloats) * 4
	if size > device.maxStorageBufferRange {
		return nil, fmt.Errorf("a buffer of %d values needs %d bytes but the GPU allows %d per buffer", numberOfFloats, size, device.maxStorageBufferRange)
	}
	device.lock.Lock()
	defer device.lock.Unlock()
	if device.closed {
		return nil, fmt.Errorf("the GPU device is closed")
	}
	storage, err := device.createBuffer(size, device.storageBufferKind())
	if err != nil {
		return nil, err
	}
	buffer := &Buffer{device: device, numberOfFloats: numberOfFloats, storage: storage}
	device.computeBuffers[buffer] = true
	return buffer, nil
}

func (buffer *Buffer) NumberOfFloats() int {
	return buffer.numberOfFloats
}

func (buffer *Buffer) checkUsable(values []float64) {
	if buffer.freed {
		panic("GPU buffer was already freed")
	}
	if values != nil && len(values) > buffer.numberOfFloats {
		panic(fmt.Sprintf("GPU buffer holds %d values but got %d", buffer.numberOfFloats, len(values)))
	}
}

func (buffer *Buffer) hostSide() (*gpuBuffer, error) {
	if !buffer.device.useStagingBuffers {
		return &buffer.storage, nil
	}
	if buffer.staging.buffer == 0 {
		staging, err := buffer.device.createBuffer(uint64(buffer.numberOfFloats)*4, stagingBufferKind())
		if err != nil {
			return nil, err
		}
		buffer.staging = staging
	}
	return &buffer.staging, nil
}

func (buffer *Buffer) Upload(values []float64) error {
	recorder, err := buffer.device.immediateRecorder()
	if err != nil {
		return err
	}
	recorder.Begin()
	if err := recorder.Upload(buffer, values); err != nil {
		return err
	}
	return recorder.Submit()
}

func (buffer *Buffer) Download(values []float64) error {
	recorder, err := buffer.device.immediateRecorder()
	if err != nil {
		return err
	}
	recorder.Begin()
	recorder.Download(buffer, values)
	return recorder.Submit()
}

func (buffer *Buffer) Free() {
	device := buffer.device
	device.lock.Lock()
	defer device.lock.Unlock()
	if buffer.freed {
		return
	}
	buffer.freed = true
	delete(device.computeBuffers, buffer)
	if device.closed {
		return
	}
	device.destroyBuffer(&buffer.storage)
	device.destroyBuffer(&buffer.staging)
}

func (device *Device) NewProgram(name string, spirv []byte, numberOfBuffers int, pushConstantWords int) (*Program, error) {
	if numberOfBuffers < 1 || numberOfBuffers > largestBindingsPerProgram {
		return nil, fmt.Errorf("program %q: needs between 1 and %d buffers, got %d", name, largestBindingsPerProgram, numberOfBuffers)
	}
	if pushConstantWords < 0 || pushConstantWords*4 > 128 {
		return nil, fmt.Errorf("program %q: push constants must fit in 128 bytes, got %d words", name, pushConstantWords)
	}
	if len(spirv) < 4 || len(spirv)%4 != 0 {
		return nil, fmt.Errorf("program %q: the shader is not valid SPIR-V", name)
	}
	device.lock.Lock()
	defer device.lock.Unlock()
	if device.closed {
		return nil, fmt.Errorf("the GPU device is closed")
	}
	program := &Program{device: device, name: name, numberOfBuffers: numberOfBuffers, pushConstantWords: pushConstantWords}

	bindings := make([]vkDescriptorSetLayoutBinding, numberOfBuffers)
	for binding := range bindings {
		bindings[binding] = vkDescriptorSetLayoutBinding{
			binding:         uint32(binding),
			descriptorType:  vkDescriptorTypeStorageBuffer,
			descriptorCount: 1,
			stageFlags:      vkShaderStageComputeBit,
		}
	}
	layoutInfo := &vkDescriptorSetLayoutCreateInfo{
		structureType: vkStructureTypeDescriptorSetLayoutCreateInfo,
		bindingCount:  uint32(numberOfBuffers),
		bindings:      unsafe.Pointer(&bindings[0]),
	}
	result := vkCreateDescriptorSetLayout(device.logicalDevice, unsafe.Pointer(layoutInfo), nil, &program.descriptorSetLayout)
	runtime.KeepAlive(bindings)
	if err := checkResult("vkCreateDescriptorSetLayout", result); err != nil {
		return nil, err
	}

	setLayouts := []uint64{program.descriptorSetLayout}
	pipelineLayoutInfo := &vkPipelineLayoutCreateInfo{
		structureType:  vkStructureTypePipelineLayoutCreateInfo,
		setLayoutCount: 1,
		setLayouts:     unsafe.Pointer(&setLayouts[0]),
	}
	pushConstantRange := &vkPushConstantRange{stageFlags: vkShaderStageComputeBit, size: uint32(pushConstantWords * 4)}
	if pushConstantWords > 0 {
		pipelineLayoutInfo.pushConstantRangeCount = 1
		pipelineLayoutInfo.pushConstantRanges = unsafe.Pointer(pushConstantRange)
	}
	result = vkCreatePipelineLayout(device.logicalDevice, unsafe.Pointer(pipelineLayoutInfo), nil, &program.pipelineLayout)
	runtime.KeepAlive(setLayouts)
	runtime.KeepAlive(pushConstantRange)
	if err := checkResult("vkCreatePipelineLayout", result); err != nil {
		device.destroyProgram(program)
		return nil, err
	}

	shaderWords := make([]uint32, len(spirv)/4)
	copy(unsafe.Slice((*byte)(unsafe.Pointer(&shaderWords[0])), len(spirv)), spirv)
	shaderInfo := &vkShaderModuleCreateInfo{
		structureType: vkStructureTypeShaderModuleCreateInfo,
		codeSize:      uintptr(len(spirv)),
		code:          unsafe.Pointer(&shaderWords[0]),
	}
	result = vkCreateShaderModule(device.logicalDevice, unsafe.Pointer(shaderInfo), nil, &program.shaderModule)
	runtime.KeepAlive(shaderWords)
	if err := checkResult("vkCreateShaderModule", result); err != nil {
		device.destroyProgram(program)
		return nil, err
	}

	entryPointName := makeCString("main")
	pipelineInfo := &vkComputePipelineCreateInfo{
		structureType: vkStructureTypeComputePipelineCreateInfo,
		stage: vkPipelineShaderStageCreateInfo{
			structureType: vkStructureTypePipelineShaderStageCreateInfo,
			stage:         vkShaderStageComputeBit,
			module:        program.shaderModule,
			name:          unsafe.Pointer(entryPointName),
		},
		layout:            program.pipelineLayout,
		basePipelineIndex: -1,
	}
	result = vkCreateComputePipelines(device.logicalDevice, 0, 1, unsafe.Pointer(pipelineInfo), nil, &program.pipeline)
	runtime.KeepAlive(entryPointName)
	if err := checkResult("vkCreateComputePipelines", result); err != nil {
		device.destroyProgram(program)
		return nil, err
	}
	device.computePrograms[program] = true
	return program, nil
}

func (device *Device) destroyProgram(program *Program) {
	if program.pipeline != 0 {
		vkDestroyPipeline(device.logicalDevice, program.pipeline, nil)
	}
	if program.shaderModule != 0 {
		vkDestroyShaderModule(device.logicalDevice, program.shaderModule, nil)
	}
	if program.pipelineLayout != 0 {
		vkDestroyPipelineLayout(device.logicalDevice, program.pipelineLayout, nil)
	}
	if program.descriptorSetLayout != 0 {
		vkDestroyDescriptorSetLayout(device.logicalDevice, program.descriptorSetLayout, nil)
	}
	*program = Program{device: device, name: program.name, freed: true}
}

func (program *Program) Free() {
	device := program.device
	device.lock.Lock()
	defer device.lock.Unlock()
	if program.freed {
		return
	}
	delete(device.computePrograms, program)
	if device.closed {
		program.freed = true
		return
	}
	device.destroyProgram(program)
}

func (device *Device) NewRecorder() (*Recorder, error) {
	device.lock.Lock()
	defer device.lock.Unlock()
	if device.closed {
		return nil, fmt.Errorf("the GPU device is closed")
	}
	return device.newRecorderLocked()
}

func (device *Device) newRecorderLocked() (*Recorder, error) {
	recorder := &Recorder{device: device}
	poolInfo := &vkCommandPoolCreateInfo{
		structureType:    vkStructureTypeCommandPoolCreateInfo,
		flags:            vkCommandPoolCreateResetCommandBufferBit,
		queueFamilyIndex: device.queueFamilyIndex,
	}
	if err := checkResult("vkCreateCommandPool", vkCreateCommandPool(device.logicalDevice, unsafe.Pointer(poolInfo), nil, &recorder.commandPool)); err != nil {
		return nil, err
	}
	allocateInfo := &vkCommandBufferAllocateInfo{
		structureType:      vkStructureTypeCommandBufferAllocateInfo,
		commandPool:        recorder.commandPool,
		level:              vkCommandBufferLevelPrimary,
		commandBufferCount: 1,
	}
	if err := checkResult("vkAllocateCommandBuffers", vkAllocateCommandBuffers(device.logicalDevice, unsafe.Pointer(allocateInfo), &recorder.commandBuffer)); err != nil {
		device.destroyRecorder(recorder)
		return nil, err
	}
	fenceInfo := &vkFenceCreateInfo{structureType: vkStructureTypeFenceCreateInfo}
	if err := checkResult("vkCreateFence", vkCreateFence(device.logicalDevice, unsafe.Pointer(fenceInfo), nil, &recorder.fence)); err != nil {
		device.destroyRecorder(recorder)
		return nil, err
	}
	device.computeRecorders[recorder] = true
	return recorder, nil
}

func (device *Device) immediateRecorder() (*Recorder, error) {
	device.lock.Lock()
	defer device.lock.Unlock()
	if device.closed {
		return nil, fmt.Errorf("the GPU device is closed")
	}
	if device.uploadRecorder == nil {
		recorder, err := device.newRecorderLocked()
		if err != nil {
			return nil, err
		}
		device.uploadRecorder = recorder
	}
	return device.uploadRecorder, nil
}

func (device *Device) destroyRecorder(recorder *Recorder) {
	for _, pool := range recorder.descriptorPools {
		vkDestroyDescriptorPool(device.logicalDevice, pool, nil)
	}
	if recorder.fence != 0 {
		vkDestroyFence(device.logicalDevice, recorder.fence, nil)
	}
	if recorder.queryPool != 0 {
		vkDestroyQueryPool(device.logicalDevice, recorder.queryPool, nil)
	}
	if recorder.commandPool != 0 {
		vkDestroyCommandPool(device.logicalDevice, recorder.commandPool, nil)
	}
	*recorder = Recorder{device: device, freed: true}
}

func (recorder *Recorder) Free() {
	device := recorder.device
	device.lock.Lock()
	defer device.lock.Unlock()
	if recorder.freed {
		return
	}
	delete(device.computeRecorders, recorder)
	if device.uploadRecorder == recorder {
		device.uploadRecorder = nil
	}
	if device.closed {
		recorder.freed = true
		return
	}
	device.destroyRecorder(recorder)
}

func (recorder *Recorder) Begin() {
	if recorder.freed {
		panic("GPU recorder was already freed")
	}
	if recorder.recording {
		panic("GPU recorder: Begin called twice without Submit")
	}
	if err := checkResult("vkResetCommandBuffer", vkResetCommandBuffer(recorder.commandBuffer, 0)); err != nil {
		panic(err)
	}
	for _, pool := range recorder.descriptorPools {
		vkResetDescriptorPool(recorder.device.logicalDevice, pool, 0)
	}
	recorder.poolsUsed = 0
	recorder.setsLeftInPool = 0
	recorder.downloads = nil
	recorder.dispatchesInStep = 0
	beginInfo := &vkCommandBufferBeginInfo{
		structureType: vkStructureTypeCommandBufferBeginInfo,
		flags:         vkCommandBufferUsageOneTimeSubmitBit,
	}
	if err := checkResult("vkBeginCommandBuffer", vkBeginCommandBuffer(recorder.commandBuffer, unsafe.Pointer(beginInfo))); err != nil {
		panic(err)
	}
	recorder.dispatchNames = nil
	if recorder.measuringTime {
		vkCmdResetQueryPool(recorder.commandBuffer, recorder.queryPool, 0, largestTimedDispatches+1)
		vkCmdWriteTimestamp(recorder.commandBuffer, vkPipelineStageTopOfPipeBit, recorder.queryPool, 0)
	}
	recorder.recording = true
}

func (recorder *Recorder) checkRecording() {
	if !recorder.recording {
		panic("GPU recorder: call Begin first")
	}
}

func (recorder *Recorder) addFullBarrier() {
	barrier := &vkMemoryBarrier{
		structureType:         vkStructureTypeMemoryBarrier,
		sourceAccessMask:      vkAccessShaderWriteBit | vkAccessTransferWriteBit,
		destinationAccessMask: vkAccessShaderReadBit | vkAccessShaderWriteBit | vkAccessTransferReadBit | vkAccessTransferWriteBit,
	}
	stages := uint32(vkPipelineStageComputeShaderBit | vkPipelineStageTransferBit)
	vkCmdPipelineBarrier(recorder.commandBuffer, stages, stages, 0, 1, unsafe.Pointer(barrier), 0, nil, 0, nil)
	runtime.KeepAlive(barrier)
}

func (recorder *Recorder) Upload(buffer *Buffer, values []float64) error {
	recorder.checkRecording()
	buffer.checkUsable(values)
	hostSide, err := buffer.hostSide()
	if err != nil {
		return err
	}
	writeValues(hostSide, values)
	if err := buffer.device.flushIfNeeded(hostSide); err != nil {
		return err
	}
	if buffer.device.useStagingBuffers && len(values) > 0 {
		region := &vkBufferCopy{size: uint64(len(values)) * 4}
		vkCmdCopyBuffer(recorder.commandBuffer, buffer.staging.buffer, buffer.storage.buffer, 1, unsafe.Pointer(region))
		runtime.KeepAlive(region)
		recorder.addFullBarrier()
	}
	return nil
}

func (recorder *Recorder) Fill(buffer *Buffer, value float64) {
	recorder.checkRecording()
	buffer.checkUsable(nil)
	vkCmdFillBuffer(recorder.commandBuffer, buffer.storage.buffer, 0, uint64(buffer.numberOfFloats)*4, Float(value))
	recorder.addFullBarrier()
}

func (recorder *Recorder) Copy(source *Buffer, destination *Buffer, numberOfFloats int) {
	recorder.checkRecording()
	if numberOfFloats > source.numberOfFloats || numberOfFloats > destination.numberOfFloats {
		panic(fmt.Sprintf("GPU recorder: can't copy %d values between buffers of %d and %d", numberOfFloats, source.numberOfFloats, destination.numberOfFloats))
	}
	region := &vkBufferCopy{size: uint64(numberOfFloats) * 4}
	vkCmdCopyBuffer(recorder.commandBuffer, source.storage.buffer, destination.storage.buffer, 1, unsafe.Pointer(region))
	runtime.KeepAlive(region)
	recorder.addFullBarrier()
}

func (recorder *Recorder) Download(buffer *Buffer, values []float64) {
	recorder.checkRecording()
	buffer.checkUsable(values)
	if buffer.device.useStagingBuffers && len(values) > 0 {
		if _, err := buffer.hostSide(); err != nil {
			panic(err)
		}
		region := &vkBufferCopy{size: uint64(len(values)) * 4}
		vkCmdCopyBuffer(recorder.commandBuffer, buffer.storage.buffer, buffer.staging.buffer, 1, unsafe.Pointer(region))
		runtime.KeepAlive(region)
		recorder.addFullBarrier()
	}
	recorder.downloads = append(recorder.downloads, pendingDownload{buffer: buffer, values: values})
}

func (recorder *Recorder) newDescriptorSet(program *Program) (uint64, error) {
	device := recorder.device
	if recorder.setsLeftInPool == 0 {
		if recorder.poolsUsed == len(recorder.descriptorPools) {
			poolSize := &vkDescriptorPoolSize{descriptorType: vkDescriptorTypeStorageBuffer, descriptorCount: descriptorSetsPerPool * largestBindingsPerProgram}
			poolInfo := &vkDescriptorPoolCreateInfo{
				structureType: vkStructureTypeDescriptorPoolCreateInfo,
				maxSets:       descriptorSetsPerPool,
				poolSizeCount: 1,
				poolSizes:     unsafe.Pointer(poolSize),
			}
			var pool uint64
			result := vkCreateDescriptorPool(device.logicalDevice, unsafe.Pointer(poolInfo), nil, &pool)
			runtime.KeepAlive(poolSize)
			if err := checkResult("vkCreateDescriptorPool", result); err != nil {
				return 0, err
			}
			recorder.descriptorPools = append(recorder.descriptorPools, pool)
		}
		recorder.poolsUsed++
		recorder.setsLeftInPool = descriptorSetsPerPool
	}
	setLayouts := []uint64{program.descriptorSetLayout}
	allocateInfo := &vkDescriptorSetAllocateInfo{
		structureType:      vkStructureTypeDescriptorSetAllocateInfo,
		descriptorPool:     recorder.descriptorPools[recorder.poolsUsed-1],
		descriptorSetCount: 1,
		setLayouts:         unsafe.Pointer(&setLayouts[0]),
	}
	var descriptorSet uint64
	result := vkAllocateDescriptorSets(device.logicalDevice, unsafe.Pointer(allocateInfo), &descriptorSet)
	runtime.KeepAlive(setLayouts)
	if err := checkResult("vkAllocateDescriptorSets", result); err != nil {
		return 0, err
	}
	recorder.setsLeftInPool--
	return descriptorSet, nil
}

func (recorder *Recorder) Run(program *Program, groupsAcross uint32, groupsDown uint32, groupsDeep uint32, pushConstants []uint32, buffers ...*Buffer) {
	recorder.checkRecording()
	if program.freed {
		panic(fmt.Sprintf("GPU program %q was already freed", program.name))
	}
	if len(buffers) != program.numberOfBuffers {
		panic(fmt.Sprintf("GPU program %q takes %d buffers, got %d", program.name, program.numberOfBuffers, len(buffers)))
	}
	if len(pushConstants) != program.pushConstantWords {
		panic(fmt.Sprintf("GPU program %q takes %d push constant words, got %d", program.name, program.pushConstantWords, len(pushConstants)))
	}
	limits := recorder.device.maxWorkGroupCount
	if groupsAcross < 1 || groupsDown < 1 || groupsDeep < 1 || groupsAcross > limits[0] || groupsDown > limits[1] || groupsDeep > limits[2] {
		panic(fmt.Sprintf("GPU program %q: %dx%dx%d work groups is outside the GPU's limit of %v", program.name, groupsAcross, groupsDown, groupsDeep, limits))
	}
	descriptorSet, err := recorder.newDescriptorSet(program)
	if err != nil {
		panic(err)
	}
	bufferInfos := make([]vkDescriptorBufferInfo, len(buffers))
	writes := make([]vkWriteDescriptorSet, len(buffers))
	for binding, buffer := range buffers {
		buffer.checkUsable(nil)
		bufferInfos[binding] = vkDescriptorBufferInfo{buffer: buffer.storage.buffer, rangeSize: vkWholeSize}
		writes[binding] = vkWriteDescriptorSet{
			structureType:      vkStructureTypeWriteDescriptorSet,
			destinationSet:     descriptorSet,
			destinationBinding: uint32(binding),
			descriptorCount:    1,
			descriptorType:     vkDescriptorTypeStorageBuffer,
			bufferInfo:         unsafe.Pointer(&bufferInfos[binding]),
		}
	}
	vkUpdateDescriptorSets(recorder.device.logicalDevice, uint32(len(writes)), unsafe.Pointer(&writes[0]), 0, nil)
	runtime.KeepAlive(bufferInfos)

	vkCmdBindPipeline(recorder.commandBuffer, vkPipelineBindPointCompute, program.pipeline)
	descriptorSetPointer := new(uint64)
	*descriptorSetPointer = descriptorSet
	vkCmdBindDescriptorSets(recorder.commandBuffer, vkPipelineBindPointCompute, program.pipelineLayout, 0, 1, descriptorSetPointer, 0, nil)
	if len(pushConstants) > 0 {
		pushCopy := append([]uint32(nil), pushConstants...)
		vkCmdPushConstants(recorder.commandBuffer, program.pipelineLayout, vkShaderStageComputeBit, 0, uint32(len(pushCopy)*4), unsafe.Pointer(&pushCopy[0]))
		runtime.KeepAlive(pushCopy)
	}
	vkCmdDispatch(recorder.commandBuffer, groupsAcross, groupsDown, groupsDeep)
	recorder.addFullBarrier()
	if recorder.measuringTime && len(recorder.dispatchNames) < largestTimedDispatches {
		recorder.dispatchNames = append(recorder.dispatchNames, program.name)
		vkCmdWriteTimestamp(recorder.commandBuffer, vkPipelineStageBottomOfPipeBit, recorder.queryPool, uint32(len(recorder.dispatchNames)))
	}
	recorder.dispatchesInStep++
}

func (recorder *Recorder) DispatchesRecorded() int {
	return recorder.dispatchesInStep
}

func (recorder *Recorder) Submit() error {
	recorder.checkRecording()
	device := recorder.device
	barrier := &vkMemoryBarrier{
		structureType:         vkStructureTypeMemoryBarrier,
		sourceAccessMask:      vkAccessShaderWriteBit | vkAccessTransferWriteBit,
		destinationAccessMask: vkAccessHostReadBit,
	}
	vkCmdPipelineBarrier(recorder.commandBuffer, vkPipelineStageComputeShaderBit|vkPipelineStageTransferBit, vkPipelineStageHostBit, 0, 1, unsafe.Pointer(barrier), 0, nil, 0, nil)
	runtime.KeepAlive(barrier)
	recorder.recording = false
	if err := checkResult("vkEndCommandBuffer", vkEndCommandBuffer(recorder.commandBuffer)); err != nil {
		return err
	}

	device.lock.Lock()
	if device.closed {
		device.lock.Unlock()
		return fmt.Errorf("the GPU device is closed")
	}
	commandBuffers := []uintptr{recorder.commandBuffer}
	submitInfo := &vkSubmitInfo{
		structureType:      vkStructureTypeSubmitInfo,
		commandBufferCount: 1,
		commandBuffers:     unsafe.Pointer(&commandBuffers[0]),
	}
	result := vkQueueSubmit(device.queue, 1, unsafe.Pointer(submitInfo), recorder.fence)
	runtime.KeepAlive(commandBuffers)
	device.lock.Unlock()
	if err := checkResult("vkQueueSubmit", result); err != nil {
		return err
	}
	fence := new(uint64)
	*fence = recorder.fence
	if err := checkResult("vkWaitForFences", vkWaitForFences(device.logicalDevice, 1, fence, 1, ^uint64(0))); err != nil {
		return err
	}
	if err := checkResult("vkResetFences", vkResetFences(device.logicalDevice, 1, fence)); err != nil {
		return err
	}

	for _, download := range recorder.downloads {
		hostSide, err := download.buffer.hostSide()
		if err != nil {
			return err
		}
		if err := device.invalidateIfNeeded(hostSide); err != nil {
			return err
		}
		readValues(hostSide, download.values)
	}
	if recorder.measuringTime {
		if err := recorder.readDispatchTimes(); err != nil {
			return err
		}
	}
	recorder.downloads = nil
	return nil
}

func (recorder *Recorder) readDispatchTimes() error {
	recorder.dispatchTimes = nil
	if len(recorder.dispatchNames) == 0 {
		return nil
	}
	ticks := make([]uint64, len(recorder.dispatchNames)+1)
	result := vkGetQueryPoolResults(recorder.device.logicalDevice, recorder.queryPool, 0, uint32(len(ticks)), uintptr(len(ticks)*8), unsafe.Pointer(&ticks[0]), 8, vkQueryResult64Bit|vkQueryResultWaitBit)
	if err := checkResult("vkGetQueryPoolResults", result); err != nil {
		return err
	}
	for index, name := range recorder.dispatchNames {
		elapsedTicks := float64(ticks[index+1] - ticks[index])
		recorder.dispatchTimes = append(recorder.dispatchTimes, DispatchTime{ProgramName: name, Seconds: elapsedTicks * recorder.device.nanosecondsPerTick / 1e9})
	}
	return nil
}

func (recorder *Recorder) MeasureTime(on bool) error {
	if recorder.recording {
		panic("GPU recorder: call MeasureTime before Begin")
	}
	if on && !recorder.device.canMeasureTime {
		return fmt.Errorf("this GPU can't measure time on its compute queue")
	}
	if on && recorder.queryPool == 0 {
		createInfo := &vkQueryPoolCreateInfo{
			structureType: vkStructureTypeQueryPoolCreateInfo,
			queryType:     vkQueryTypeTimestamp,
			queryCount:    largestTimedDispatches + 1,
		}
		if err := checkResult("vkCreateQueryPool", vkCreateQueryPool(recorder.device.logicalDevice, unsafe.Pointer(createInfo), nil, &recorder.queryPool)); err != nil {
			return err
		}
	}
	recorder.measuringTime = on
	return nil
}

func (recorder *Recorder) DispatchTimes() []DispatchTime {
	return append([]DispatchTime(nil), recorder.dispatchTimes...)
}

func (device *Device) freeComputeObjects() {
	for recorder := range device.computeRecorders {
		device.destroyRecorder(recorder)
	}
	for program := range device.computePrograms {
		device.destroyProgram(program)
	}
	for buffer := range device.computeBuffers {
		device.destroyBuffer(&buffer.storage)
		device.destroyBuffer(&buffer.staging)
		buffer.freed = true
	}
	device.computeRecorders = map[*Recorder]bool{}
	device.computePrograms = map[*Program]bool{}
	device.computeBuffers = map[*Buffer]bool{}
	device.uploadRecorder = nil
}
