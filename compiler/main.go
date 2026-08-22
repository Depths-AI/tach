package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"tach/driver"
	"tach/foundation"
)

var version = "0.2.0"

func main() {
	if len(os.Args) < 2 {
		fail(fmt.Errorf("private Tach compiler operation is required"))
	}
	var diagnostics foundation.Diagnostics
	var err error
	switch os.Args[1] {
	case "_build":
		diagnostics, err = build(os.Args[2:])
	case "_check":
		diagnostics, err = check(os.Args[2:])
	case "_docs":
		diagnostics, err = docs(os.Args[2:])
	case "_fmt":
		err = format(os.Args[2:])
	case "_version":
		if len(os.Args) != 2 {
			err = fmt.Errorf("_version accepts no arguments")
		} else {
			fmt.Println(version)
		}
	default:
		err = fmt.Errorf("unknown private compiler operation %q", os.Args[1])
	}
	if err != nil {
		if reported, ok := driver.ErrorDiagnostics(err); ok {
			writeDiagnostics(reported)
			os.Exit(1)
		}
		fail(err)
	}
	if len(diagnostics) > 0 {
		writeDiagnostics(diagnostics)
	}
}

func build(args []string) (foundation.Diagnostics, error) {
	flags := flag.NewFlagSet("_build", flag.ContinueOnError)
	output := flags.String("output", "", "")
	verbose := flags.Bool("verbose", false, "")
	workers := flags.Int("workers", 0, "")
	if err := flags.Parse(args); err != nil {
		return nil, err
	}
	if flags.NArg() != 0 || *output == "" {
		return nil, fmt.Errorf("_build requires --output and no positional arguments")
	}
	result, err := driver.Build(".", *workers)
	if err == nil {
		err = driver.WriteNativeArtifacts(result, *output, *verbose)
	}
	if result == nil {
		return nil, err
	}
	return result.Diagnostics, err
}

func check(args []string) (foundation.Diagnostics, error) {
	workers, err := workerArgs("_check", args)
	if err != nil {
		return nil, err
	}
	result, err := driver.Check(".", workers)
	if err == nil {
		payload := struct {
			Project json.RawMessage `json:"project"`
			Runtime json.RawMessage `json:"runtime"`
		}{result.Description, result.MetadataJSON}
		var data []byte
		data, err = json.Marshal(payload)
		if err == nil {
			_, err = os.Stdout.Write(data)
		}
	}
	if result == nil {
		return nil, err
	}
	return result.Diagnostics, err
}

func docs(args []string) (foundation.Diagnostics, error) {
	workers, err := workerArgs("_docs", args)
	if err != nil {
		return nil, err
	}
	result, err := driver.Describe(".", workers)
	if err == nil {
		_, err = os.Stdout.Write(result.Description)
	}
	if result == nil {
		return nil, err
	}
	return result.Diagnostics, err
}

func format(args []string) error {
	workers, err := workerArgs("_fmt", args)
	if err != nil {
		return err
	}
	return driver.Format(".", workers)
}

func workerArgs(operation string, args []string) (int, error) {
	flags := flag.NewFlagSet(operation, flag.ContinueOnError)
	workers := flags.Int("workers", 0, "")
	if err := flags.Parse(args); err != nil {
		return 0, err
	}
	if flags.NArg() != 0 {
		return 0, fmt.Errorf("%s accepts no positional arguments", operation)
	}
	return *workers, nil
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}

func writeDiagnostics(diagnostics foundation.Diagnostics) {
	_ = json.NewEncoder(os.Stderr).Encode(struct {
		Schema      int                    `json:"schema"`
		Diagnostics foundation.Diagnostics `json:"diagnostics"`
	}{1, diagnostics})
}
