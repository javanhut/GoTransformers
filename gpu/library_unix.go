//go:build !windows

package gpu

import (
	"fmt"
	"runtime"

	"github.com/ebitengine/purego"
)

func vulkanLibraryNames() []string {
	if runtime.GOOS == "darwin" {
		return []string{"libvulkan.1.dylib", "libvulkan.dylib", "libMoltenVK.dylib"}
	}
	return []string{"libvulkan.so.1", "libvulkan.so"}
}

func openVulkanLibrary() (uintptr, error) {
	var lastError error
	for _, name := range vulkanLibraryNames() {
		library, err := purego.Dlopen(name, purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err == nil {
			return library, nil
		}
		lastError = err
	}
	return 0, fmt.Errorf("could not find a Vulkan library (tried %v): %w", vulkanLibraryNames(), lastError)
}

func findFunction(library uintptr, name string) (uintptr, error) {
	return purego.Dlsym(library, name)
}
