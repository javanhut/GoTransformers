//go:build windows

package gpu

import (
	"fmt"
	"syscall"
)

func openVulkanLibrary() (uintptr, error) {
	library, err := syscall.LoadLibrary("vulkan-1.dll")
	if err != nil {
		return 0, fmt.Errorf("could not find vulkan-1.dll: %w", err)
	}
	return uintptr(library), nil
}

func findFunction(library uintptr, name string) (uintptr, error) {
	return syscall.GetProcAddress(syscall.Handle(library), name)
}
