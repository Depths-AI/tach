package driver

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestMaintainedProjectsCompile(t *testing.T) {
	for _, path := range []string{filepath.Join("..", "..", "examples"), filepath.Join("..", "..", "showcase")} {
		t.Run(filepath.Base(path), func(t *testing.T) {
			if _, err := Check(path, 0); err != nil {
				t.Fatal(err)
			}
			project, err := loadProject(path, 0)
			if err != nil {
				t.Fatal(err)
			}
			for _, kernel := range project.Kernels {
				formatted, err := formatSource(kernel.Identity+".tach", kernel.Source)
				if err != nil {
					t.Fatal(err)
				}
				if formatted != kernel.Source {
					t.Errorf("%s is not canonically formatted; run tach fmt", kernel.Identity)
				}
			}
		})
	}
}

func TestDocumentedTachSnippetsCompile(t *testing.T) {
	for _, name := range []string{"README.md", "docs/language.md", "docs/ir.md", "docs/abi.md", "tach/README.md"} {
		data, err := os.ReadFile(filepath.Join("..", "..", filepath.FromSlash(name)))
		if err != nil {
			t.Fatal(err)
		}
		parts := strings.Split(strings.ReplaceAll(string(data), "\r\n", "\n"), "```tach\n")
		if len(parts) == 1 {
			t.Fatalf("%s contains no Tach fences", name)
		}
		for index, part := range parts[1:] {
			source, _, found := strings.Cut(part, "```")
			if !found {
				t.Fatalf("%s Tach fence %d is unterminated", name, index+1)
			}
			files := map[string]string{"module/snippet.tach": source}
			if strings.Contains(source, `import "data/particles"`) {
				files["data/particles.tach"] = ""
			}
			root := projectFixture(t, files)
			if _, err := Check(root, 1); err != nil {
				t.Errorf("%s Tach fence %d: %v", name, index+1, err)
			}
		}
	}
}
