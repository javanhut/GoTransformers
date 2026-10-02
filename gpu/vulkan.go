package gpu

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

const (
	vkSuccess = 0
	vkTimeout = 2

	vkStructureTypeApplicationInfo               = 0
	vkStructureTypeInstanceCreateInfo            = 1
	vkStructureTypeDeviceQueueCreateInfo         = 2
	vkStructureTypeDeviceCreateInfo              = 3
	vkStructureTypeSubmitInfo                    = 4
	vkStructureTypeMemoryAllocateInfo            = 5
	vkStructureTypeMappedMemoryRange             = 6
	vkStructureTypeFenceCreateInfo               = 8
	vkStructureTypeBufferCreateInfo              = 12
	vkStructureTypeShaderModuleCreateInfo        = 16
	vkStructureTypePipelineShaderStageCreateInfo = 18
	vkStructureTypeComputePipelineCreateInfo     = 29
	vkStructureTypePipelineLayoutCreateInfo      = 30
	vkStructureTypeDescriptorSetLayoutCreateInfo = 32
	vkStructureTypeDescriptorPoolCreateInfo      = 33
	vkStructureTypeDescriptorSetAllocateInfo     = 34
	vkStructureTypeWriteDescriptorSet            = 35
	vkStructureTypeCommandPoolCreateInfo         = 39
	vkStructureTypeCommandBufferAllocateInfo     = 40
	vkStructureTypeCommandBufferBeginInfo        = 42
	vkStructureTypeMemoryBarrier                 = 46

	vkApiVersion1_0 = 1 << 22

	vkInstanceCreateEnumeratePortabilityBit = 0x1

	vkPhysicalDeviceTypeOther      = 0
	vkPhysicalDeviceTypeIntegrated = 1
	vkPhysicalDeviceTypeDiscrete   = 2
	vkPhysicalDeviceTypeVirtual    = 3
	vkPhysicalDeviceTypeCPU        = 4

	vkQueueComputeBit = 0x2

	vkBufferUsageTransferSourceBit      = 0x1
	vkBufferUsageTransferDestinationBit = 0x2
	vkBufferUsageStorageBufferBit       = 0x20

	vkMemoryPropertyDeviceLocalBit  = 0x1
	vkMemoryPropertyHostVisibleBit  = 0x2
	vkMemoryPropertyHostCoherentBit = 0x4
	vkMemoryPropertyHostCachedBit   = 0x8

	vkDescriptorTypeStorageBuffer = 7
	vkShaderStageComputeBit       = 0x20
	vkPipelineBindPointCompute    = 1

	vkCommandPoolCreateResetCommandBufferBit = 0x2
	vkCommandBufferLevelPrimary              = 0
	vkCommandBufferUsageOneTimeSubmitBit     = 0x1

	vkPipelineStageTransferBit      = 0x1000
	vkPipelineStageComputeShaderBit = 0x800
	vkPipelineStageHostBit          = 0x4000

	vkAccessShaderReadBit    = 0x20
	vkAccessShaderWriteBit   = 0x40
	vkAccessTransferReadBit  = 0x800
	vkAccessTransferWriteBit = 0x1000
	vkAccessHostReadBit      = 0x2000

	vkWholeSize = ^uint64(0)
)

type vkApplicationInfo struct {
	structureType      uint32
	next               unsafe.Pointer
	applicationName    unsafe.Pointer
	applicationVersion uint32
	engineName         unsafe.Pointer
	engineVersion      uint32
	apiVersion         uint32
}

type vkInstanceCreateInfo struct {
	structureType         uint32
	next                  unsafe.Pointer
	flags                 uint32
	applicationInfo       unsafe.Pointer
	enabledLayerCount     uint32
	enabledLayerNames     unsafe.Pointer
	enabledExtensionCount uint32
	enabledExtensionNames unsafe.Pointer
}

type vkExtensionProperties struct {
	extensionName [256]byte
	specVersion   uint32
}

type vkQueueFamilyProperties struct {
	queueFlags                  uint32
	queueCount                  uint32
	timestampValidBits          uint32
	minImageTransferGranularity [3]uint32
}

type vkMemoryType struct {
	propertyFlags uint32
	heapIndex     uint32
}

type vkMemoryHeap struct {
	size  uint64
	flags uint32
}

type vkPhysicalDeviceMemoryProperties struct {
	memoryTypeCount uint32
	memoryTypes     [32]vkMemoryType
	memoryHeapCount uint32
	memoryHeaps     [16]vkMemoryHeap
}

const (
	propertiesDeviceTypeOffset = 16
	propertiesDeviceNameOffset = 20
	propertiesDeviceNameLength = 256
	propertiesLimitsOffset     = 296

	limitsMaxStorageBufferRangeOffset          = 28
	limitsMaxComputeSharedMemorySizeOffset     = 216
	limitsMaxComputeWorkGroupCountOffset       = 220
	limitsMaxComputeWorkGroupInvocationsOffset = 232
)

type vkPhysicalDeviceProperties struct {
	bytes [4096]byte
}

func (properties *vkPhysicalDeviceProperties) uint32At(offset int) uint32 {
	return *(*uint32)(unsafe.Pointer(&properties.bytes[offset]))
}

func (properties *vkPhysicalDeviceProperties) deviceName() string {
	nameBytes := properties.bytes[propertiesDeviceNameOffset : propertiesDeviceNameOffset+propertiesDeviceNameLength]
	return cString(nameBytes)
}

type vkDeviceQueueCreateInfo struct {
	structureType    uint32
	next             unsafe.Pointer
	flags            uint32
	queueFamilyIndex uint32
	queueCount       uint32
	queuePriorities  unsafe.Pointer
}

type vkDeviceCreateInfo struct {
	structureType         uint32
	next                  unsafe.Pointer
	flags                 uint32
	queueCreateInfoCount  uint32
	queueCreateInfos      unsafe.Pointer
	enabledLayerCount     uint32
	enabledLayerNames     unsafe.Pointer
	enabledExtensionCount uint32
	enabledExtensionNames unsafe.Pointer
	enabledFeatures       unsafe.Pointer
}

type vkBufferCreateInfo struct {
	structureType         uint32
	next                  unsafe.Pointer
	flags                 uint32
	size                  uint64
	usage                 uint32
	sharingMode           uint32
	queueFamilyIndexCount uint32
	queueFamilyIndices    unsafe.Pointer
}

type vkMemoryRequirements struct {
	size           uint64
	alignment      uint64
	memoryTypeBits uint32
}

type vkMemoryAllocateInfo struct {
	structureType   uint32
	next            unsafe.Pointer
	allocationSize  uint64
	memoryTypeIndex uint32
}

type vkMappedMemoryRange struct {
	structureType uint32
	next          unsafe.Pointer
	memory        uint64
	offset        uint64
	size          uint64
}

type vkShaderModuleCreateInfo struct {
	structureType uint32
	next          unsafe.Pointer
	flags         uint32
	codeSize      uintptr
	code          unsafe.Pointer
}

type vkDescriptorSetLayoutBinding struct {
	binding           uint32
	descriptorType    uint32
	descriptorCount   uint32
	stageFlags        uint32
	immutableSamplers unsafe.Pointer
}

type vkDescriptorSetLayoutCreateInfo struct {
	structureType uint32
	next          unsafe.Pointer
	flags         uint32
	bindingCount  uint32
	bindings      unsafe.Pointer
}

type vkPushConstantRange struct {
	stageFlags uint32
	offset     uint32
	size       uint32
}

type vkPipelineLayoutCreateInfo struct {
	structureType          uint32
	next                   unsafe.Pointer
	flags                  uint32
	setLayoutCount         uint32
	setLayouts             unsafe.Pointer
	pushConstantRangeCount uint32
	pushConstantRanges     unsafe.Pointer
}

type vkPipelineShaderStageCreateInfo struct {
	structureType      uint32
	next               unsafe.Pointer
	flags              uint32
	stage              uint32
	module             uint64
	name               unsafe.Pointer
	specializationInfo unsafe.Pointer
}

type vkComputePipelineCreateInfo struct {
	structureType      uint32
	next               unsafe.Pointer
	flags              uint32
	stage              vkPipelineShaderStageCreateInfo
	layout             uint64
	basePipelineHandle uint64
	basePipelineIndex  int32
}

type vkDescriptorPoolSize struct {
	descriptorType  uint32
	descriptorCount uint32
}

type vkDescriptorPoolCreateInfo struct {
	structureType uint32
	next          unsafe.Pointer
	flags         uint32
	maxSets       uint32
	poolSizeCount uint32
	poolSizes     unsafe.Pointer
}

type vkDescriptorSetAllocateInfo struct {
	structureType      uint32
	next               unsafe.Pointer
	descriptorPool     uint64
	descriptorSetCount uint32
	setLayouts         unsafe.Pointer
}

type vkDescriptorBufferInfo struct {
	buffer    uint64
	offset    uint64
	rangeSize uint64
}

type vkWriteDescriptorSet struct {
	structureType           uint32
	next                    unsafe.Pointer
	destinationSet          uint64
	destinationBinding      uint32
	destinationArrayElement uint32
	descriptorCount         uint32
	descriptorType          uint32
	imageInfo               unsafe.Pointer
	bufferInfo              unsafe.Pointer
	texelBufferView         unsafe.Pointer
}

type vkCommandPoolCreateInfo struct {
	structureType    uint32
	next             unsafe.Pointer
	flags            uint32
	queueFamilyIndex uint32
}

type vkCommandBufferAllocateInfo struct {
	structureType      uint32
	next               unsafe.Pointer
	commandPool        uint64
	level              uint32
	commandBufferCount uint32
}

type vkCommandBufferBeginInfo struct {
	structureType   uint32
	next            unsafe.Pointer
	flags           uint32
	inheritanceInfo unsafe.Pointer
}

type vkBufferCopy struct {
	sourceOffset      uint64
	destinationOffset uint64
	size              uint64
}

type vkMemoryBarrier struct {
	structureType         uint32
	next                  unsafe.Pointer
	sourceAccessMask      uint32
	destinationAccessMask uint32
}

type vkSubmitInfo struct {
	structureType            uint32
	next                     unsafe.Pointer
	waitSemaphoreCount       uint32
	waitSemaphores           unsafe.Pointer
	waitDestinationStageMask unsafe.Pointer
	commandBufferCount       uint32
	commandBuffers           unsafe.Pointer
	signalSemaphoreCount     uint32
	signalSemaphores         unsafe.Pointer
}

type vkFenceCreateInfo struct {
	structureType uint32
	next          unsafe.Pointer
	flags         uint32
}

var (
	vkEnumerateInstanceExtensionProperties   func(layerName unsafe.Pointer, count *uint32, properties unsafe.Pointer) int32
	vkCreateInstance                         func(createInfo unsafe.Pointer, allocator unsafe.Pointer, instance *uintptr) int32
	vkDestroyInstance                        func(instance uintptr, allocator unsafe.Pointer)
	vkEnumeratePhysicalDevices               func(instance uintptr, count *uint32, physicalDevices unsafe.Pointer) int32
	vkGetPhysicalDeviceProperties            func(physicalDevice uintptr, properties unsafe.Pointer)
	vkGetPhysicalDeviceQueueFamilyProperties func(physicalDevice uintptr, count *uint32, properties unsafe.Pointer)
	vkGetPhysicalDeviceMemoryProperties      func(physicalDevice uintptr, properties unsafe.Pointer)
	vkEnumerateDeviceExtensionProperties     func(physicalDevice uintptr, layerName unsafe.Pointer, count *uint32, properties unsafe.Pointer) int32
	vkCreateDevice                           func(physicalDevice uintptr, createInfo unsafe.Pointer, allocator unsafe.Pointer, device *uintptr) int32
	vkDestroyDevice                          func(device uintptr, allocator unsafe.Pointer)
	vkDeviceWaitIdle                         func(device uintptr) int32
	vkGetDeviceQueue                         func(device uintptr, queueFamilyIndex uint32, queueIndex uint32, queue *uintptr)
	vkCreateBuffer                           func(device uintptr, createInfo unsafe.Pointer, allocator unsafe.Pointer, buffer *uint64) int32
	vkDestroyBuffer                          func(device uintptr, buffer uint64, allocator unsafe.Pointer)
	vkGetBufferMemoryRequirements            func(device uintptr, buffer uint64, requirements unsafe.Pointer)
	vkAllocateMemory                         func(device uintptr, allocateInfo unsafe.Pointer, allocator unsafe.Pointer, memory *uint64) int32
	vkFreeMemory                             func(device uintptr, memory uint64, allocator unsafe.Pointer)
	vkBindBufferMemory                       func(device uintptr, buffer uint64, memory uint64, offset uint64) int32
	vkMapMemory                              func(device uintptr, memory uint64, offset uint64, size uint64, flags uint32, data *unsafe.Pointer) int32
	vkUnmapMemory                            func(device uintptr, memory uint64)
	vkFlushMappedMemoryRanges                func(device uintptr, rangeCount uint32, ranges unsafe.Pointer) int32
	vkInvalidateMappedMemoryRanges           func(device uintptr, rangeCount uint32, ranges unsafe.Pointer) int32
	vkCreateShaderModule                     func(device uintptr, createInfo unsafe.Pointer, allocator unsafe.Pointer, shaderModule *uint64) int32
	vkDestroyShaderModule                    func(device uintptr, shaderModule uint64, allocator unsafe.Pointer)
	vkCreateDescriptorSetLayout              func(device uintptr, createInfo unsafe.Pointer, allocator unsafe.Pointer, layout *uint64) int32
	vkDestroyDescriptorSetLayout             func(device uintptr, layout uint64, allocator unsafe.Pointer)
	vkCreatePipelineLayout                   func(device uintptr, createInfo unsafe.Pointer, allocator unsafe.Pointer, layout *uint64) int32
	vkDestroyPipelineLayout                  func(device uintptr, layout uint64, allocator unsafe.Pointer)
	vkCreateComputePipelines                 func(device uintptr, pipelineCache uint64, count uint32, createInfos unsafe.Pointer, allocator unsafe.Pointer, pipelines *uint64) int32
	vkDestroyPipeline                        func(device uintptr, pipeline uint64, allocator unsafe.Pointer)
	vkCreateDescriptorPool                   func(device uintptr, createInfo unsafe.Pointer, allocator unsafe.Pointer, pool *uint64) int32
	vkDestroyDescriptorPool                  func(device uintptr, pool uint64, allocator unsafe.Pointer)
	vkAllocateDescriptorSets                 func(device uintptr, allocateInfo unsafe.Pointer, sets *uint64) int32
	vkUpdateDescriptorSets                   func(device uintptr, writeCount uint32, writes unsafe.Pointer, copyCount uint32, copies unsafe.Pointer)
	vkCreateCommandPool                      func(device uintptr, createInfo unsafe.Pointer, allocator unsafe.Pointer, pool *uint64) int32
	vkDestroyCommandPool                     func(device uintptr, pool uint64, allocator unsafe.Pointer)
	vkAllocateCommandBuffers                 func(device uintptr, allocateInfo unsafe.Pointer, commandBuffers *uintptr) int32
	vkBeginCommandBuffer                     func(commandBuffer uintptr, beginInfo unsafe.Pointer) int32
	vkEndCommandBuffer                       func(commandBuffer uintptr) int32
	vkResetCommandBuffer                     func(commandBuffer uintptr, flags uint32) int32
	vkCmdBindPipeline                        func(commandBuffer uintptr, bindPoint uint32, pipeline uint64)
	vkCmdBindDescriptorSets                  func(commandBuffer uintptr, bindPoint uint32, layout uint64, firstSet uint32, setCount uint32, sets *uint64, dynamicOffsetCount uint32, dynamicOffsets unsafe.Pointer)
	vkCmdPushConstants                       func(commandBuffer uintptr, layout uint64, stageFlags uint32, offset uint32, size uint32, values unsafe.Pointer)
	vkCmdDispatch                            func(commandBuffer uintptr, groupCountX uint32, groupCountY uint32, groupCountZ uint32)
	vkCmdCopyBuffer                          func(commandBuffer uintptr, sourceBuffer uint64, destinationBuffer uint64, regionCount uint32, regions unsafe.Pointer)
	vkCmdPipelineBarrier                     func(commandBuffer uintptr, sourceStageMask uint32, destinationStageMask uint32, dependencyFlags uint32, memoryBarrierCount uint32, memoryBarriers unsafe.Pointer, bufferBarrierCount uint32, bufferBarriers unsafe.Pointer, imageBarrierCount uint32, imageBarriers unsafe.Pointer)
	vkQueueSubmit                            func(queue uintptr, submitCount uint32, submits unsafe.Pointer, fence uint64) int32
	vkCreateFence                            func(device uintptr, createInfo unsafe.Pointer, allocator unsafe.Pointer, fence *uint64) int32
	vkDestroyFence                           func(device uintptr, fence uint64, allocator unsafe.Pointer)
	vkWaitForFences                          func(device uintptr, fenceCount uint32, fences *uint64, waitAll uint32, timeout uint64) int32
	vkResetFences                            func(device uintptr, fenceCount uint32, fences *uint64) int32
)

var (
	loadVulkanOnce  sync.Once
	loadVulkanError error
)

func loadVulkan() error {
	loadVulkanOnce.Do(func() {
		loadVulkanError = loadVulkanFunctions()
	})
	return loadVulkanError
}

func loadVulkanFunctions() error {
	if unsafe.Sizeof(uintptr(0)) != 8 {
		return fmt.Errorf("the GPU backend needs a 64-bit system")
	}
	library, err := openVulkanLibrary()
	if err != nil {
		return err
	}
	functions := map[string]any{
		"vkEnumerateInstanceExtensionProperties":   &vkEnumerateInstanceExtensionProperties,
		"vkCreateInstance":                         &vkCreateInstance,
		"vkDestroyInstance":                        &vkDestroyInstance,
		"vkEnumeratePhysicalDevices":               &vkEnumeratePhysicalDevices,
		"vkGetPhysicalDeviceProperties":            &vkGetPhysicalDeviceProperties,
		"vkGetPhysicalDeviceQueueFamilyProperties": &vkGetPhysicalDeviceQueueFamilyProperties,
		"vkGetPhysicalDeviceMemoryProperties":      &vkGetPhysicalDeviceMemoryProperties,
		"vkEnumerateDeviceExtensionProperties":     &vkEnumerateDeviceExtensionProperties,
		"vkCreateDevice":                           &vkCreateDevice,
		"vkDestroyDevice":                          &vkDestroyDevice,
		"vkDeviceWaitIdle":                         &vkDeviceWaitIdle,
		"vkGetDeviceQueue":                         &vkGetDeviceQueue,
		"vkCreateBuffer":                           &vkCreateBuffer,
		"vkDestroyBuffer":                          &vkDestroyBuffer,
		"vkGetBufferMemoryRequirements":            &vkGetBufferMemoryRequirements,
		"vkAllocateMemory":                         &vkAllocateMemory,
		"vkFreeMemory":                             &vkFreeMemory,
		"vkBindBufferMemory":                       &vkBindBufferMemory,
		"vkMapMemory":                              &vkMapMemory,
		"vkUnmapMemory":                            &vkUnmapMemory,
		"vkFlushMappedMemoryRanges":                &vkFlushMappedMemoryRanges,
		"vkInvalidateMappedMemoryRanges":           &vkInvalidateMappedMemoryRanges,
		"vkCreateShaderModule":                     &vkCreateShaderModule,
		"vkDestroyShaderModule":                    &vkDestroyShaderModule,
		"vkCreateDescriptorSetLayout":              &vkCreateDescriptorSetLayout,
		"vkDestroyDescriptorSetLayout":             &vkDestroyDescriptorSetLayout,
		"vkCreatePipelineLayout":                   &vkCreatePipelineLayout,
		"vkDestroyPipelineLayout":                  &vkDestroyPipelineLayout,
		"vkCreateComputePipelines":                 &vkCreateComputePipelines,
		"vkDestroyPipeline":                        &vkDestroyPipeline,
		"vkCreateDescriptorPool":                   &vkCreateDescriptorPool,
		"vkDestroyDescriptorPool":                  &vkDestroyDescriptorPool,
		"vkAllocateDescriptorSets":                 &vkAllocateDescriptorSets,
		"vkUpdateDescriptorSets":                   &vkUpdateDescriptorSets,
		"vkCreateCommandPool":                      &vkCreateCommandPool,
		"vkDestroyCommandPool":                     &vkDestroyCommandPool,
		"vkAllocateCommandBuffers":                 &vkAllocateCommandBuffers,
		"vkBeginCommandBuffer":                     &vkBeginCommandBuffer,
		"vkEndCommandBuffer":                       &vkEndCommandBuffer,
		"vkResetCommandBuffer":                     &vkResetCommandBuffer,
		"vkCmdBindPipeline":                        &vkCmdBindPipeline,
		"vkCmdBindDescriptorSets":                  &vkCmdBindDescriptorSets,
		"vkCmdPushConstants":                       &vkCmdPushConstants,
		"vkCmdDispatch":                            &vkCmdDispatch,
		"vkCmdCopyBuffer":                          &vkCmdCopyBuffer,
		"vkCmdPipelineBarrier":                     &vkCmdPipelineBarrier,
		"vkQueueSubmit":                            &vkQueueSubmit,
		"vkCreateFence":                            &vkCreateFence,
		"vkDestroyFence":                           &vkDestroyFence,
		"vkWaitForFences":                          &vkWaitForFences,
		"vkResetFences":                            &vkResetFences,
	}
	for name, function := range functions {
		address, err := findFunction(library, name)
		if err != nil || address == 0 {
			return fmt.Errorf("the Vulkan library is missing %s", name)
		}
		purego.RegisterFunc(function, address)
	}
	return nil
}

func cString(bytes []byte) string {
	for i, character := range bytes {
		if character == 0 {
			return string(bytes[:i])
		}
	}
	return string(bytes)
}

func makeCString(text string) *byte {
	bytes := make([]byte, len(text)+1)
	copy(bytes, text)
	return &bytes[0]
}

func checkResult(functionName string, result int32) error {
	if result != vkSuccess {
		return fmt.Errorf("%s failed with Vulkan error %d", functionName, result)
	}
	return nil
}
