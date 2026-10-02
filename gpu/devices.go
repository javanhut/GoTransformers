package gpu

import (
	"fmt"
	"runtime"
	"unsafe"
)

type DeviceInfo struct {
	Index int
	Name  string
	Kind  string
}

func kindName(deviceType uint32) string {
	switch deviceType {
	case vkPhysicalDeviceTypeDiscrete:
		return "discrete"
	case vkPhysicalDeviceTypeIntegrated:
		return "integrated"
	case vkPhysicalDeviceTypeVirtual:
		return "virtual"
	case vkPhysicalDeviceTypeCPU:
		return "cpu"
	}
	return "other"
}

func instanceExtensionIsAvailable(name string) bool {
	var count uint32
	if vkEnumerateInstanceExtensionProperties(nil, &count, nil) != vkSuccess || count == 0 {
		return false
	}
	extensions := make([]vkExtensionProperties, count)
	if vkEnumerateInstanceExtensionProperties(nil, &count, unsafe.Pointer(&extensions[0])) != vkSuccess {
		return false
	}
	for _, extension := range extensions[:count] {
		if cString(extension.extensionName[:]) == name {
			return true
		}
	}
	return false
}

func deviceExtensionIsAvailable(physicalDevice uintptr, name string) bool {
	var count uint32
	if vkEnumerateDeviceExtensionProperties(physicalDevice, nil, &count, nil) != vkSuccess || count == 0 {
		return false
	}
	extensions := make([]vkExtensionProperties, count)
	if vkEnumerateDeviceExtensionProperties(physicalDevice, nil, &count, unsafe.Pointer(&extensions[0])) != vkSuccess {
		return false
	}
	for _, extension := range extensions[:count] {
		if cString(extension.extensionName[:]) == name {
			return true
		}
	}
	return false
}

func createInstance() (uintptr, error) {
	if err := loadVulkan(); err != nil {
		return 0, err
	}
	applicationInfo := &vkApplicationInfo{
		structureType:   vkStructureTypeApplicationInfo,
		applicationName: unsafe.Pointer(makeCString("GoTransformers")),
		engineName:      unsafe.Pointer(makeCString("GoTransformers")),
		apiVersion:      vkApiVersion1_0,
	}
	createInfo := &vkInstanceCreateInfo{
		structureType:   vkStructureTypeInstanceCreateInfo,
		applicationInfo: unsafe.Pointer(applicationInfo),
	}
	portabilityExtension := "VK_KHR_portability_enumeration"
	var extensionNames []*byte
	if instanceExtensionIsAvailable(portabilityExtension) {
		extensionNames = append(extensionNames, makeCString(portabilityExtension))
		createInfo.flags = vkInstanceCreateEnumeratePortabilityBit
		createInfo.enabledExtensionCount = 1
		createInfo.enabledExtensionNames = unsafe.Pointer(&extensionNames[0])
	}

	var instance uintptr
	result := vkCreateInstance(unsafe.Pointer(createInfo), nil, &instance)
	runtime.KeepAlive(applicationInfo)
	runtime.KeepAlive(extensionNames)
	if err := checkResult("vkCreateInstance", result); err != nil {
		return 0, err
	}
	return instance, nil
}

func listPhysicalDevices(instance uintptr) ([]uintptr, error) {
	var count uint32
	if err := checkResult("vkEnumeratePhysicalDevices", vkEnumeratePhysicalDevices(instance, &count, nil)); err != nil {
		return nil, err
	}
	if count == 0 {
		return nil, nil
	}
	physicalDevices := make([]uintptr, count)
	if err := checkResult("vkEnumeratePhysicalDevices", vkEnumeratePhysicalDevices(instance, &count, unsafe.Pointer(&physicalDevices[0]))); err != nil {
		return nil, err
	}
	return physicalDevices[:count], nil
}

func readProperties(physicalDevice uintptr) *vkPhysicalDeviceProperties {
	properties := &vkPhysicalDeviceProperties{}
	vkGetPhysicalDeviceProperties(physicalDevice, unsafe.Pointer(properties))
	return properties
}

func findComputeQueueFamily(physicalDevice uintptr) (uint32, bool) {
	var count uint32
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, nil)
	if count == 0 {
		return 0, false
	}
	families := make([]vkQueueFamilyProperties, count)
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, unsafe.Pointer(&families[0]))
	for index, family := range families[:count] {
		if family.queueFlags&vkQueueComputeBit != 0 && family.queueCount > 0 {
			return uint32(index), true
		}
	}
	return 0, false
}

func describeDevices(instance uintptr) ([]DeviceInfo, []uintptr, error) {
	physicalDevices, err := listPhysicalDevices(instance)
	if err != nil {
		return nil, nil, err
	}
	var devices []DeviceInfo
	var usablePhysicalDevices []uintptr
	for _, physicalDevice := range physicalDevices {
		if _, hasComputeQueue := findComputeQueueFamily(physicalDevice); !hasComputeQueue {
			continue
		}
		properties := readProperties(physicalDevice)
		devices = append(devices, DeviceInfo{
			Index: len(devices),
			Name:  properties.deviceName(),
			Kind:  kindName(properties.uint32At(propertiesDeviceTypeOffset)),
		})
		usablePhysicalDevices = append(usablePhysicalDevices, physicalDevice)
	}
	return devices, usablePhysicalDevices, nil
}

func ListDevices() ([]DeviceInfo, error) {
	instance, err := createInstance()
	if err != nil {
		return nil, err
	}
	defer vkDestroyInstance(instance, nil)
	devices, _, err := describeDevices(instance)
	if err != nil {
		return nil, err
	}
	if len(devices) == 0 {
		return nil, fmt.Errorf("Vulkan is installed but found no GPU that can run compute work")
	}
	return devices, nil
}

func kindRank(kind string) int {
	switch kind {
	case "discrete":
		return 0
	case "integrated":
		return 1
	case "virtual":
		return 2
	case "other":
		return 3
	}
	return 4
}

func OpenBest() (*Device, error) {
	devices, err := ListDevices()
	if err != nil {
		return nil, err
	}
	best := devices[0]
	for _, device := range devices {
		if kindRank(device.Kind) < kindRank(best.Kind) {
			best = device
		}
	}
	return Open(best.Index)
}
