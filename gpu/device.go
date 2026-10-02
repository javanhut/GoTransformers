package gpu

import (
	"fmt"
	"math/bits"
	"runtime"
	"sync"
	"transformer/vectormath"
	"unsafe"
)

const defaultMinimumWorkForGPU = 128 * 128 * 128

type gpuBuffer struct {
	buffer   uint64
	memory   uint64
	size     uint64
	mapped   unsafe.Pointer
	coherent bool
}

type bufferKind struct {
	usage          uint32
	requiredFlags  uint32
	preferredFlags uint32
}

type Device struct {
	MinimumWorkForGPU int

	info              DeviceInfo
	useStagingBuffers bool
	lock              sync.Mutex
	closed            bool
	lastError         error

	instance         uintptr
	physicalDevice   uintptr
	logicalDevice    uintptr
	queue            uintptr
	queueFamilyIndex uint32
	memoryProperties *vkPhysicalDeviceMemoryProperties

	maxStorageBufferRange uint64
	maxWorkGroupCount     [3]uint32

	shaderModule        uint64
	descriptorSetLayout uint64
	pipelineLayout      uint64
	pipeline            uint64
	descriptorPool      uint64
	descriptorSet       uint64
	commandPool         uint64
	commandBuffer       uintptr
	fence               uint64

	firstBuffer   gpuBuffer
	secondBuffer  gpuBuffer
	resultBuffer  gpuBuffer
	firstStaging  gpuBuffer
	secondStaging gpuBuffer
	resultStaging gpuBuffer

	descriptorsNeedUpdate bool
}

func Open(index int) (*Device, error) {
	return openDevice(index, false)
}

func openDevice(index int, forceStagingBuffers bool) (*Device, error) {
	instance, err := createInstance()
	if err != nil {
		return nil, err
	}
	devices, physicalDevices, err := describeDevices(instance)
	if err != nil {
		vkDestroyInstance(instance, nil)
		return nil, err
	}
	if index < 0 || index >= len(devices) {
		vkDestroyInstance(instance, nil)
		return nil, fmt.Errorf("there is no GPU number %d, found %d GPUs", index, len(devices))
	}

	device := &Device{
		MinimumWorkForGPU: defaultMinimumWorkForGPU,
		info:              devices[index],
		instance:          instance,
		physicalDevice:    physicalDevices[index],
	}
	kind := device.info.Kind
	device.useStagingBuffers = forceStagingBuffers || !(kind == "integrated" || kind == "cpu")

	if err := device.setUp(); err != nil {
		device.Close()
		return nil, fmt.Errorf("could not start GPU %q: %w", device.info.Name, err)
	}
	return device, nil
}

func (device *Device) setUp() error {
	properties := readProperties(device.physicalDevice)
	device.maxStorageBufferRange = uint64(properties.uint32At(propertiesLimitsOffset + limitsMaxStorageBufferRangeOffset))
	for i := 0; i < 3; i++ {
		device.maxWorkGroupCount[i] = properties.uint32At(propertiesLimitsOffset + limitsMaxComputeWorkGroupCountOffset + 4*i)
	}
	if properties.uint32At(propertiesLimitsOffset+limitsMaxComputeWorkGroupInvocationsOffset) < 256 {
		return fmt.Errorf("the GPU can't run 256 threads in a group")
	}
	if properties.uint32At(propertiesLimitsOffset+limitsMaxComputeSharedMemorySizeOffset) < 8192 {
		return fmt.Errorf("the GPU has less than 8KB of shared memory")
	}

	device.memoryProperties = &vkPhysicalDeviceMemoryProperties{}
	vkGetPhysicalDeviceMemoryProperties(device.physicalDevice, unsafe.Pointer(device.memoryProperties))

	if err := device.createLogicalDevice(); err != nil {
		return err
	}
	if err := device.createPipeline(); err != nil {
		return err
	}
	return device.createCommandBuffer()
}

func (device *Device) createLogicalDevice() error {
	queueFamilyIndex, found := findComputeQueueFamily(device.physicalDevice)
	if !found {
		return fmt.Errorf("the GPU has no compute queue")
	}
	device.queueFamilyIndex = queueFamilyIndex

	queuePriority := new(float32)
	*queuePriority = 1
	queueCreateInfo := &vkDeviceQueueCreateInfo{
		structureType:    vkStructureTypeDeviceQueueCreateInfo,
		queueFamilyIndex: queueFamilyIndex,
		queueCount:       1,
		queuePriorities:  unsafe.Pointer(queuePriority),
	}
	createInfo := &vkDeviceCreateInfo{
		structureType:        vkStructureTypeDeviceCreateInfo,
		queueCreateInfoCount: 1,
		queueCreateInfos:     unsafe.Pointer(queueCreateInfo),
	}
	portabilityExtension := "VK_KHR_portability_subset"
	var extensionNames []*byte
	if deviceExtensionIsAvailable(device.physicalDevice, portabilityExtension) {
		extensionNames = append(extensionNames, makeCString(portabilityExtension))
		createInfo.enabledExtensionCount = 1
		createInfo.enabledExtensionNames = unsafe.Pointer(&extensionNames[0])
	}

	result := vkCreateDevice(device.physicalDevice, unsafe.Pointer(createInfo), nil, &device.logicalDevice)
	runtime.KeepAlive(queuePriority)
	runtime.KeepAlive(queueCreateInfo)
	runtime.KeepAlive(extensionNames)
	if err := checkResult("vkCreateDevice", result); err != nil {
		return err
	}
	vkGetDeviceQueue(device.logicalDevice, queueFamilyIndex, 0, &device.queue)
	return nil
}

func (device *Device) createPipeline() error {
	shaderWords := make([]uint32, (len(matrixMultiplyShader)+3)/4)
	copy(unsafe.Slice((*byte)(unsafe.Pointer(&shaderWords[0])), len(shaderWords)*4), matrixMultiplyShader)
	shaderInfo := &vkShaderModuleCreateInfo{
		structureType: vkStructureTypeShaderModuleCreateInfo,
		codeSize:      uintptr(len(matrixMultiplyShader)),
		code:          unsafe.Pointer(&shaderWords[0]),
	}
	result := vkCreateShaderModule(device.logicalDevice, unsafe.Pointer(shaderInfo), nil, &device.shaderModule)
	runtime.KeepAlive(shaderWords)
	if err := checkResult("vkCreateShaderModule", result); err != nil {
		return err
	}

	bindings := make([]vkDescriptorSetLayoutBinding, 3)
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
		bindingCount:  uint32(len(bindings)),
		bindings:      unsafe.Pointer(&bindings[0]),
	}
	result = vkCreateDescriptorSetLayout(device.logicalDevice, unsafe.Pointer(layoutInfo), nil, &device.descriptorSetLayout)
	runtime.KeepAlive(bindings)
	if err := checkResult("vkCreateDescriptorSetLayout", result); err != nil {
		return err
	}

	setLayouts := []uint64{device.descriptorSetLayout}
	pushConstantRange := &vkPushConstantRange{stageFlags: vkShaderStageComputeBit, size: uint32(unsafe.Sizeof(pushConstants{}))}
	pipelineLayoutInfo := &vkPipelineLayoutCreateInfo{
		structureType:          vkStructureTypePipelineLayoutCreateInfo,
		setLayoutCount:         1,
		setLayouts:             unsafe.Pointer(&setLayouts[0]),
		pushConstantRangeCount: 1,
		pushConstantRanges:     unsafe.Pointer(pushConstantRange),
	}
	result = vkCreatePipelineLayout(device.logicalDevice, unsafe.Pointer(pipelineLayoutInfo), nil, &device.pipelineLayout)
	runtime.KeepAlive(setLayouts)
	runtime.KeepAlive(pushConstantRange)
	if err := checkResult("vkCreatePipelineLayout", result); err != nil {
		return err
	}

	entryPointName := makeCString("main")
	pipelineInfo := &vkComputePipelineCreateInfo{
		structureType: vkStructureTypeComputePipelineCreateInfo,
		stage: vkPipelineShaderStageCreateInfo{
			structureType: vkStructureTypePipelineShaderStageCreateInfo,
			stage:         vkShaderStageComputeBit,
			module:        device.shaderModule,
			name:          unsafe.Pointer(entryPointName),
		},
		layout:            device.pipelineLayout,
		basePipelineIndex: -1,
	}
	result = vkCreateComputePipelines(device.logicalDevice, 0, 1, unsafe.Pointer(pipelineInfo), nil, &device.pipeline)
	runtime.KeepAlive(entryPointName)
	if err := checkResult("vkCreateComputePipelines", result); err != nil {
		return err
	}

	poolSize := &vkDescriptorPoolSize{descriptorType: vkDescriptorTypeStorageBuffer, descriptorCount: 3}
	poolInfo := &vkDescriptorPoolCreateInfo{
		structureType: vkStructureTypeDescriptorPoolCreateInfo,
		maxSets:       1,
		poolSizeCount: 1,
		poolSizes:     unsafe.Pointer(poolSize),
	}
	result = vkCreateDescriptorPool(device.logicalDevice, unsafe.Pointer(poolInfo), nil, &device.descriptorPool)
	runtime.KeepAlive(poolSize)
	if err := checkResult("vkCreateDescriptorPool", result); err != nil {
		return err
	}

	allocateInfo := &vkDescriptorSetAllocateInfo{
		structureType:      vkStructureTypeDescriptorSetAllocateInfo,
		descriptorPool:     device.descriptorPool,
		descriptorSetCount: 1,
		setLayouts:         unsafe.Pointer(&setLayouts[0]),
	}
	result = vkAllocateDescriptorSets(device.logicalDevice, unsafe.Pointer(allocateInfo), &device.descriptorSet)
	runtime.KeepAlive(setLayouts)
	return checkResult("vkAllocateDescriptorSets", result)
}

func (device *Device) createCommandBuffer() error {
	poolInfo := &vkCommandPoolCreateInfo{
		structureType:    vkStructureTypeCommandPoolCreateInfo,
		flags:            vkCommandPoolCreateResetCommandBufferBit,
		queueFamilyIndex: device.queueFamilyIndex,
	}
	if err := checkResult("vkCreateCommandPool", vkCreateCommandPool(device.logicalDevice, unsafe.Pointer(poolInfo), nil, &device.commandPool)); err != nil {
		return err
	}
	allocateInfo := &vkCommandBufferAllocateInfo{
		structureType:      vkStructureTypeCommandBufferAllocateInfo,
		commandPool:        device.commandPool,
		level:              vkCommandBufferLevelPrimary,
		commandBufferCount: 1,
	}
	if err := checkResult("vkAllocateCommandBuffers", vkAllocateCommandBuffers(device.logicalDevice, unsafe.Pointer(allocateInfo), &device.commandBuffer)); err != nil {
		return err
	}
	fenceInfo := &vkFenceCreateInfo{structureType: vkStructureTypeFenceCreateInfo}
	return checkResult("vkCreateFence", vkCreateFence(device.logicalDevice, unsafe.Pointer(fenceInfo), nil, &device.fence))
}

func (device *Device) chooseMemoryType(allowedTypes uint32, requiredFlags uint32, preferredFlags uint32) (uint32, uint32, bool) {
	bestIndex := -1
	bestScore := -1
	for index := 0; index < int(device.memoryProperties.memoryTypeCount); index++ {
		if allowedTypes&(1<<index) == 0 {
			continue
		}
		flags := device.memoryProperties.memoryTypes[index].propertyFlags
		if flags&requiredFlags != requiredFlags {
			continue
		}
		score := bits.OnesCount32(flags & preferredFlags)
		if score > bestScore {
			bestIndex = index
			bestScore = score
		}
	}
	if bestIndex < 0 {
		return 0, 0, false
	}
	return uint32(bestIndex), device.memoryProperties.memoryTypes[bestIndex].propertyFlags, true
}

func (device *Device) createBuffer(size uint64, kind bufferKind) (gpuBuffer, error) {
	created := gpuBuffer{size: size}
	bufferInfo := &vkBufferCreateInfo{
		structureType: vkStructureTypeBufferCreateInfo,
		size:          size,
		usage:         kind.usage,
	}
	if err := checkResult("vkCreateBuffer", vkCreateBuffer(device.logicalDevice, unsafe.Pointer(bufferInfo), nil, &created.buffer)); err != nil {
		return created, err
	}

	requirements := &vkMemoryRequirements{}
	vkGetBufferMemoryRequirements(device.logicalDevice, created.buffer, unsafe.Pointer(requirements))
	memoryTypeIndex, memoryFlags, found := device.chooseMemoryType(requirements.memoryTypeBits, kind.requiredFlags, kind.preferredFlags)
	if !found {
		device.destroyBuffer(&created)
		return created, fmt.Errorf("the GPU has no memory type with flags %#x for this buffer", kind.requiredFlags)
	}

	allocateInfo := &vkMemoryAllocateInfo{
		structureType:   vkStructureTypeMemoryAllocateInfo,
		allocationSize:  requirements.size,
		memoryTypeIndex: memoryTypeIndex,
	}
	if err := checkResult("vkAllocateMemory", vkAllocateMemory(device.logicalDevice, unsafe.Pointer(allocateInfo), nil, &created.memory)); err != nil {
		device.destroyBuffer(&created)
		return created, err
	}
	if err := checkResult("vkBindBufferMemory", vkBindBufferMemory(device.logicalDevice, created.buffer, created.memory, 0)); err != nil {
		device.destroyBuffer(&created)
		return created, err
	}

	if memoryFlags&vkMemoryPropertyHostVisibleBit != 0 {
		created.coherent = memoryFlags&vkMemoryPropertyHostCoherentBit != 0
		if err := checkResult("vkMapMemory", vkMapMemory(device.logicalDevice, created.memory, 0, vkWholeSize, 0, &created.mapped)); err != nil {
			device.destroyBuffer(&created)
			return created, err
		}
	}
	return created, nil
}

func (device *Device) destroyBuffer(buffer *gpuBuffer) {
	if buffer.mapped != nil {
		vkUnmapMemory(device.logicalDevice, buffer.memory)
	}
	if buffer.buffer != 0 {
		vkDestroyBuffer(device.logicalDevice, buffer.buffer, nil)
	}
	if buffer.memory != 0 {
		vkFreeMemory(device.logicalDevice, buffer.memory, nil)
	}
	*buffer = gpuBuffer{}
}

func (device *Device) makeBufferBigEnough(buffer *gpuBuffer, neededSize uint64, kind bufferKind) error {
	if buffer.size >= neededSize {
		return nil
	}
	newSize := neededSize
	if buffer.size*2 > newSize {
		newSize = buffer.size * 2
	}
	if newSize > device.maxStorageBufferRange {
		newSize = neededSize
	}
	device.destroyBuffer(buffer)
	created, err := device.createBuffer(newSize, kind)
	if err != nil {
		return err
	}
	*buffer = created
	device.descriptorsNeedUpdate = true
	return nil
}

func (device *Device) makeAllBuffersBigEnough(firstSize uint64, secondSize uint64, resultSize uint64) error {
	if !device.useStagingBuffers {
		inputKind := bufferKind{
			usage:          vkBufferUsageStorageBufferBit,
			requiredFlags:  vkMemoryPropertyHostVisibleBit,
			preferredFlags: vkMemoryPropertyDeviceLocalBit | vkMemoryPropertyHostCoherentBit,
		}
		resultKind := bufferKind{
			usage:          vkBufferUsageStorageBufferBit,
			requiredFlags:  vkMemoryPropertyHostVisibleBit,
			preferredFlags: vkMemoryPropertyDeviceLocalBit | vkMemoryPropertyHostCachedBit,
		}
		if err := device.makeBufferBigEnough(&device.firstBuffer, firstSize, inputKind); err != nil {
			return err
		}
		if err := device.makeBufferBigEnough(&device.secondBuffer, secondSize, inputKind); err != nil {
			return err
		}
		return device.makeBufferBigEnough(&device.resultBuffer, resultSize, resultKind)
	}

	inputKind := bufferKind{
		usage:         vkBufferUsageStorageBufferBit | vkBufferUsageTransferDestinationBit,
		requiredFlags: vkMemoryPropertyDeviceLocalBit,
	}
	resultKind := bufferKind{
		usage:         vkBufferUsageStorageBufferBit | vkBufferUsageTransferSourceBit,
		requiredFlags: vkMemoryPropertyDeviceLocalBit,
	}
	uploadKind := bufferKind{
		usage:          vkBufferUsageTransferSourceBit,
		requiredFlags:  vkMemoryPropertyHostVisibleBit,
		preferredFlags: vkMemoryPropertyHostCoherentBit,
	}
	downloadKind := bufferKind{
		usage:          vkBufferUsageTransferDestinationBit,
		requiredFlags:  vkMemoryPropertyHostVisibleBit,
		preferredFlags: vkMemoryPropertyHostCachedBit | vkMemoryPropertyHostCoherentBit,
	}
	steps := []struct {
		buffer *gpuBuffer
		size   uint64
		kind   bufferKind
	}{
		{&device.firstBuffer, firstSize, inputKind},
		{&device.secondBuffer, secondSize, inputKind},
		{&device.resultBuffer, resultSize, resultKind},
		{&device.firstStaging, firstSize, uploadKind},
		{&device.secondStaging, secondSize, uploadKind},
		{&device.resultStaging, resultSize, downloadKind},
	}
	for _, step := range steps {
		if err := device.makeBufferBigEnough(step.buffer, step.size, step.kind); err != nil {
			return err
		}
	}
	return nil
}

func (device *Device) updateDescriptors() {
	if !device.descriptorsNeedUpdate {
		return
	}
	buffers := []*gpuBuffer{&device.firstBuffer, &device.secondBuffer, &device.resultBuffer}
	bufferInfos := make([]vkDescriptorBufferInfo, len(buffers))
	writes := make([]vkWriteDescriptorSet, len(buffers))
	for binding, buffer := range buffers {
		bufferInfos[binding] = vkDescriptorBufferInfo{buffer: buffer.buffer, rangeSize: vkWholeSize}
		writes[binding] = vkWriteDescriptorSet{
			structureType:      vkStructureTypeWriteDescriptorSet,
			destinationSet:     device.descriptorSet,
			destinationBinding: uint32(binding),
			descriptorCount:    1,
			descriptorType:     vkDescriptorTypeStorageBuffer,
			bufferInfo:         unsafe.Pointer(&bufferInfos[binding]),
		}
	}
	vkUpdateDescriptorSets(device.logicalDevice, uint32(len(writes)), unsafe.Pointer(&writes[0]), 0, nil)
	runtime.KeepAlive(bufferInfos)
	device.descriptorsNeedUpdate = false
}

func (device *Device) flushIfNeeded(buffer *gpuBuffer) error {
	if buffer.coherent {
		return nil
	}
	memoryRange := &vkMappedMemoryRange{structureType: vkStructureTypeMappedMemoryRange, memory: buffer.memory, size: vkWholeSize}
	return checkResult("vkFlushMappedMemoryRanges", vkFlushMappedMemoryRanges(device.logicalDevice, 1, unsafe.Pointer(memoryRange)))
}

func (device *Device) invalidateIfNeeded(buffer *gpuBuffer) error {
	if buffer.coherent {
		return nil
	}
	memoryRange := &vkMappedMemoryRange{structureType: vkStructureTypeMappedMemoryRange, memory: buffer.memory, size: vkWholeSize}
	return checkResult("vkInvalidateMappedMemoryRanges", vkInvalidateMappedMemoryRanges(device.logicalDevice, 1, unsafe.Pointer(memoryRange)))
}

func writeValues(buffer *gpuBuffer, values []float64) {
	if len(values) == 0 {
		return
	}
	gpuValues := unsafe.Slice((*float32)(buffer.mapped), len(values))
	for i, value := range values {
		gpuValues[i] = float32(value)
	}
}

func readValues(buffer *gpuBuffer, values []float64) {
	if len(values) == 0 {
		return
	}
	gpuValues := unsafe.Slice((*float32)(buffer.mapped), len(values))
	for i := range values {
		values[i] = float64(gpuValues[i])
	}
}

type pushConstants struct {
	rows               uint32
	inner              uint32
	columns            uint32
	firstIsTransposed  uint32
	secondIsTransposed uint32
}

func boolToUint32(value bool) uint32 {
	if value {
		return 1
	}
	return 0
}

func (device *Device) addBarrier(sourceStage uint32, destinationStage uint32, sourceAccess uint32, destinationAccess uint32) {
	barrier := &vkMemoryBarrier{
		structureType:         vkStructureTypeMemoryBarrier,
		sourceAccessMask:      sourceAccess,
		destinationAccessMask: destinationAccess,
	}
	vkCmdPipelineBarrier(device.commandBuffer, sourceStage, destinationStage, 0, 1, unsafe.Pointer(barrier), 0, nil, 0, nil)
	runtime.KeepAlive(barrier)
}

func (device *Device) addCopy(source *gpuBuffer, destination *gpuBuffer, size uint64) {
	region := &vkBufferCopy{size: size}
	vkCmdCopyBuffer(device.commandBuffer, source.buffer, destination.buffer, 1, unsafe.Pointer(region))
	runtime.KeepAlive(region)
}

func (device *Device) recordCommands(sizes *pushConstants, firstSize uint64, secondSize uint64, resultSize uint64) error {
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

	if device.useStagingBuffers {
		device.addCopy(&device.firstStaging, &device.firstBuffer, firstSize)
		device.addCopy(&device.secondStaging, &device.secondBuffer, secondSize)
		device.addBarrier(vkPipelineStageTransferBit, vkPipelineStageComputeShaderBit, vkAccessTransferWriteBit, vkAccessShaderReadBit)
	}

	vkCmdBindPipeline(device.commandBuffer, vkPipelineBindPointCompute, device.pipeline)
	descriptorSet := new(uint64)
	*descriptorSet = device.descriptorSet
	vkCmdBindDescriptorSets(device.commandBuffer, vkPipelineBindPointCompute, device.pipelineLayout, 0, 1, descriptorSet, 0, nil)
	vkCmdPushConstants(device.commandBuffer, device.pipelineLayout, vkShaderStageComputeBit, 0, uint32(unsafe.Sizeof(*sizes)), unsafe.Pointer(sizes))

	groupsAcross := (sizes.columns + 63) / 64
	groupsDown := (sizes.rows + 63) / 64
	vkCmdDispatch(device.commandBuffer, groupsAcross, groupsDown, 1)

	if device.useStagingBuffers {
		device.addBarrier(vkPipelineStageComputeShaderBit, vkPipelineStageTransferBit, vkAccessShaderWriteBit, vkAccessTransferReadBit)
		device.addCopy(&device.resultBuffer, &device.resultStaging, resultSize)
		device.addBarrier(vkPipelineStageTransferBit, vkPipelineStageHostBit, vkAccessTransferWriteBit, vkAccessHostReadBit)
	} else {
		device.addBarrier(vkPipelineStageComputeShaderBit, vkPipelineStageHostBit, vkAccessShaderWriteBit, vkAccessHostReadBit)
	}
	return checkResult("vkEndCommandBuffer", vkEndCommandBuffer(device.commandBuffer))
}

func (device *Device) submitAndWait() error {
	commandBuffers := []uintptr{device.commandBuffer}
	submitInfo := &vkSubmitInfo{
		structureType:      vkStructureTypeSubmitInfo,
		commandBufferCount: 1,
		commandBuffers:     unsafe.Pointer(&commandBuffers[0]),
	}
	result := vkQueueSubmit(device.queue, 1, unsafe.Pointer(submitInfo), device.fence)
	runtime.KeepAlive(commandBuffers)
	if err := checkResult("vkQueueSubmit", result); err != nil {
		return err
	}
	fence := new(uint64)
	*fence = device.fence
	if err := checkResult("vkWaitForFences", vkWaitForFences(device.logicalDevice, 1, fence, 1, ^uint64(0))); err != nil {
		return err
	}
	return checkResult("vkResetFences", vkResetFences(device.logicalDevice, 1, fence))
}

func (device *Device) multiplyOnGPU(first vectormath.Matrix, second vectormath.Matrix, rows int, inner int, columns int, firstIsTransposed bool, secondIsTransposed bool) (vectormath.Matrix, error) {
	firstSize := uint64(len(first.Values)) * 4
	secondSize := uint64(len(second.Values)) * 4
	resultSize := uint64(rows) * uint64(columns) * 4
	for _, size := range []uint64{firstSize, secondSize, resultSize} {
		if size > device.maxStorageBufferRange {
			return vectormath.Matrix{}, fmt.Errorf("a matrix needs %d bytes but the GPU allows %d per buffer", size, device.maxStorageBufferRange)
		}
	}
	if uint64((columns+63)/64) > uint64(device.maxWorkGroupCount[0]) || uint64((rows+63)/64) > uint64(device.maxWorkGroupCount[1]) {
		return vectormath.Matrix{}, fmt.Errorf("the result is %dx%d, which is too big for one GPU dispatch", rows, columns)
	}

	if err := device.makeAllBuffersBigEnough(firstSize, secondSize, resultSize); err != nil {
		return vectormath.Matrix{}, err
	}
	device.updateDescriptors()

	uploadFirst := &device.firstBuffer
	uploadSecond := &device.secondBuffer
	download := &device.resultBuffer
	if device.useStagingBuffers {
		uploadFirst = &device.firstStaging
		uploadSecond = &device.secondStaging
		download = &device.resultStaging
	}
	writeValues(uploadFirst, first.Values)
	writeValues(uploadSecond, second.Values)
	if err := device.flushIfNeeded(uploadFirst); err != nil {
		return vectormath.Matrix{}, err
	}
	if err := device.flushIfNeeded(uploadSecond); err != nil {
		return vectormath.Matrix{}, err
	}

	sizes := &pushConstants{
		rows:               uint32(rows),
		inner:              uint32(inner),
		columns:            uint32(columns),
		firstIsTransposed:  boolToUint32(firstIsTransposed),
		secondIsTransposed: boolToUint32(secondIsTransposed),
	}
	if err := device.recordCommands(sizes, firstSize, secondSize, resultSize); err != nil {
		return vectormath.Matrix{}, err
	}
	if err := device.submitAndWait(); err != nil {
		return vectormath.Matrix{}, err
	}

	if err := device.invalidateIfNeeded(download); err != nil {
		return vectormath.Matrix{}, err
	}
	result := vectormath.NewMatrix(rows, columns)
	readValues(download, result.Values)
	return result, nil
}

func (device *Device) multiplyOrUseCPU(first vectormath.Matrix, second vectormath.Matrix, rows int, inner int, columns int, firstIsTransposed bool, secondIsTransposed bool, multiplyOnCPU func(vectormath.Matrix, vectormath.Matrix) vectormath.Matrix) vectormath.Matrix {
	work := rows * inner * columns
	if work == 0 || work < device.MinimumWorkForGPU {
		return multiplyOnCPU(first, second)
	}
	device.lock.Lock()
	defer device.lock.Unlock()
	if device.closed {
		return multiplyOnCPU(first, second)
	}
	result, err := device.multiplyOnGPU(first, second, rows, inner, columns, firstIsTransposed, secondIsTransposed)
	if err != nil {
		device.lastError = err
		return multiplyOnCPU(first, second)
	}
	return result
}

func (device *Device) Name() string {
	return fmt.Sprintf("GPU %s (%s, Vulkan)", device.info.Name, device.info.Kind)
}

func (device *Device) Info() DeviceInfo {
	return device.info
}

func (device *Device) LastError() error {
	device.lock.Lock()
	defer device.lock.Unlock()
	return device.lastError
}

func (device *Device) MatrixTimesMatrix(first vectormath.Matrix, second vectormath.Matrix) vectormath.Matrix {
	return device.multiplyOrUseCPU(first, second, first.Rows, first.Columns, second.Columns, false, false, vectormath.CPUBackend{}.MatrixTimesMatrix)
}

func (device *Device) MatrixTimesTransposed(first vectormath.Matrix, second vectormath.Matrix) vectormath.Matrix {
	return device.multiplyOrUseCPU(first, second, first.Rows, first.Columns, second.Rows, false, true, vectormath.CPUBackend{}.MatrixTimesTransposed)
}

func (device *Device) TransposedTimesMatrix(first vectormath.Matrix, second vectormath.Matrix) vectormath.Matrix {
	return device.multiplyOrUseCPU(first, second, first.Columns, first.Rows, second.Columns, true, false, vectormath.CPUBackend{}.TransposedTimesMatrix)
}

func (device *Device) Close() {
	device.lock.Lock()
	defer device.lock.Unlock()
	if device.closed {
		return
	}
	device.closed = true

	if device.logicalDevice != 0 {
		vkDeviceWaitIdle(device.logicalDevice)
		for _, buffer := range []*gpuBuffer{&device.firstBuffer, &device.secondBuffer, &device.resultBuffer, &device.firstStaging, &device.secondStaging, &device.resultStaging} {
			device.destroyBuffer(buffer)
		}
		if device.fence != 0 {
			vkDestroyFence(device.logicalDevice, device.fence, nil)
		}
		if device.commandPool != 0 {
			vkDestroyCommandPool(device.logicalDevice, device.commandPool, nil)
		}
		if device.descriptorPool != 0 {
			vkDestroyDescriptorPool(device.logicalDevice, device.descriptorPool, nil)
		}
		if device.pipeline != 0 {
			vkDestroyPipeline(device.logicalDevice, device.pipeline, nil)
		}
		if device.pipelineLayout != 0 {
			vkDestroyPipelineLayout(device.logicalDevice, device.pipelineLayout, nil)
		}
		if device.descriptorSetLayout != 0 {
			vkDestroyDescriptorSetLayout(device.logicalDevice, device.descriptorSetLayout, nil)
		}
		if device.shaderModule != 0 {
			vkDestroyShaderModule(device.logicalDevice, device.shaderModule, nil)
		}
		vkDestroyDevice(device.logicalDevice, nil)
	}
	if device.instance != 0 {
		vkDestroyInstance(device.instance, nil)
	}
}
