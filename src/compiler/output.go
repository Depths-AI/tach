package compiler

import (
	"fmt"
	"os"
	"path/filepath"
)

func WriteNativeArtifacts(result *Result, target BuildTarget, output string) error {
	if result == nil {
		return fmt.Errorf("nil project result")
	}
	root, err := filepath.Abs(output)
	if err != nil {
		return err
	}
	info, err := os.Lstat(root)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("output must be an existing non-symlink directory: %s", root)
	}
	files := map[string][]byte{}
	if target == TargetWeb {
		files["index.js"] = []byte(result.JavaScript)
		files["index.d.ts"] = []byte(result.TypeScript)
		files["kernel.wgsl"] = []byte(result.WGSL)
	} else if target == TargetSPIRV {
		files["kernel.spv"] = result.SPIRV
	} else {
		return fmt.Errorf("invalid target %q", target)
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		return err
	}
	if len(entries) != 0 {
		return fmt.Errorf("output staging directory must be empty")
	}
	for name, data := range files {
		if err := os.WriteFile(filepath.Join(root, name), data, 0o644); err != nil {
			return err
		}
	}
	return nil
}
