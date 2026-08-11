package spirvtest

import (
	"encoding/binary"
	"fmt"
	"math"
)

type exampleCase struct {
	name        string
	kernel      string
	buffers     map[string][]byte
	parameters  []byte
	invocations [3]uint32
	check       func(map[string][]byte) error
}

func exampleCases() []exampleCase {
	controlInput := make([]float32, 64)
	for i := range controlInput {
		controlInput[i] = float32(i + 1)
	}
	forInput := make([]uint32, 256)
	copy(forInput, []uint32{1, 2, 3, 4})

	return []exampleCase{
		{
			name: "atomics", kernel: "accumulate", invocations: invocations(64),
			buffers: map[string][]byte{"counters": make([]byte, 16)},
			check: func(output map[string][]byte) error {
				actual, err := readU32(output, "counters")
				if err != nil {
					return err
				}
				if actual[0] != 64 {
					return fmt.Errorf("counters.total = %d, want 64", actual[0])
				}
				return nil
			},
		},
		{
			name: "bitwise", kernel: "bitwise", invocations: invocations(8),
			buffers: map[string][]byte{"out": u32Bytes(make([]uint32, 8))},
			check: func(output map[string][]byte) error {
				actual, err := readU32(output, "out")
				if err != nil {
					return err
				}
				expected := expectedBitwise()
				for i, value := range actual {
					if value != expected {
						return fmt.Errorf("out[%d] = %d, want %d", i, value, expected)
					}
				}
				return nil
			},
		},
		{
			name: "control", kernel: "transform", invocations: invocations(64),
			buffers:    map[string][]byte{"data": f32Bytes(controlInput)},
			parameters: structBytes(16, f32Field(0, 2), u32Field(4, 64), u32Field(8, 1)),
			check: func(output map[string][]byte) error {
				actual, err := readF32(output, "data")
				if err != nil {
					return err
				}
				for i, input := range controlInput {
					expected := input*2 + 1
					if input > 50 {
						expected = 100
					}
					if actual[i] != expected {
						return fmt.Errorf("data[%d] = %v, want %v", i, actual[i], expected)
					}
				}
				return nil
			},
		},
		{
			name: "for", kernel: "reduceLanes", invocations: invocations(256),
			buffers: map[string][]byte{"data": u32Bytes(forInput)},
			check: func(output map[string][]byte) error {
				actual, err := readU32(output, "data")
				if err != nil {
					return err
				}
				for i, value := range actual {
					expected := uint32(0)
					if i == 0 {
						expected = 10
					}
					if value != expected {
						return fmt.Errorf("data[%d] = %d, want %d", i, value, expected)
					}
				}
				return nil
			},
		},
		{
			name: "math", kernel: "math", invocations: invocations(4),
			buffers: map[string][]byte{"out": f32Bytes(make([]float32, 16))},
			check: func(output map[string][]byte) error {
				actual, err := readF32(output, "out")
				if err != nil {
					return err
				}
				for i := 0; i < 4; i++ {
					expected := expectedMath(i)
					for component := range expected {
						index := i*4 + component
						if !closeFloat(actual[index], expected[component], 0.0005) {
							return fmt.Errorf("out[%d][%d] = %v, want %v", i, component, actual[index], expected[component])
						}
					}
				}
				return nil
			},
		},
		{
			name: "particles", kernel: "integrate", invocations: invocations(2),
			buffers: map[string][]byte{
				"particles": f32Bytes([]float32{
					1, 2, 3, 4, 2, 4, 6, 8,
					-1, -2, -3, -4, 1, 2, 3, 4,
				}),
			},
			parameters: structBytes(16, f32Field(0, 0.5), u32Field(4, 2)),
			check: func(output map[string][]byte) error {
				actual, err := readF32(output, "particles")
				if err != nil {
					return err
				}
				expected := []float32{
					2, 4, 6, 8, 2, 4, 6, 8,
					-0.5, -1, -1.5, -2, 1, 2, 3, 4,
				}
				for i, value := range actual {
					if value != expected[i] {
						return fmt.Errorf("particles float %d = %v, want %v", i, value, expected[i])
					}
				}
				return nil
			},
		},
		{
			name: "scalars", kernel: "scale", invocations: invocations(4),
			buffers:    map[string][]byte{"data": f32Bytes([]float32{1, 2, 3, 4})},
			parameters: structBytes(16, f32Field(0, 2.5)),
			check: func(output map[string][]byte) error {
				actual, err := readF32(output, "data")
				if err != nil {
					return err
				}
				expected := []float32{2.5, 5, 7.5, 10}
				for i, value := range actual {
					if value != expected[i] {
						return fmt.Errorf("data[%d] = %v, want %v", i, value, expected[i])
					}
				}
				return nil
			},
		},
	}
}

func invocations(x uint32) [3]uint32 {
	return [3]uint32{x, 1, 1}
}

func expectedBitwise() uint32 {
	u := uint32(0xff00)
	left := u << (40 & 31)
	logical := u >> (36 & 31)
	signed := int32(-64)
	arithmetic := uint32(signed >> (35 & 31))
	mixed := (left | logical) ^ ^u
	mixed &= 0xffff
	mixed <<= (33 & 31)
	return mixed | arithmetic
}

func expectedMath(index int) []float32 {
	a := []float64{float64(index + 1), 2, 3}
	length := math.Sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
	normalized := []float64{a[0] / length, a[1] / length, a[2] / length}
	cross := []float64{-3, 0, a[0]}
	dx, dy, dz := a[0]-cross[0], a[1]-cross[1], a[2]-cross[2]
	distance := math.Sqrt(dx*dx + dy*dy + dz*dz)
	wave := math.Sin(length) + math.Cos(length) + math.Tan(0.25)
	shaped := math.Sqrt(math.Abs(wave)) + 1/math.Sqrt(length+1)
	expo := math.Exp2(math.Log2(length+1)) + math.Exp(math.Log(length+1))
	powered := math.Pow(length+1, 2)
	rounded := math.Floor(powered) + math.Ceil(distance) + math.Trunc(shaped)
	bounded := math.Max(1, math.Min(math.Min(float64(index), 1024), math.Max(float64(index), 1)))
	return []float32{
		float32(normalized[0]), float32(normalized[1]), float32(normalized[2]),
		float32(shaped + expo + rounded + bounded),
	}
}

func closeFloat(actual, expected float32, tolerance float64) bool {
	return !float32IsNaN(actual) && math.Abs(float64(actual-expected)) <= tolerance
}

func float32IsNaN(value float32) bool {
	return value != value
}

func u32Bytes(values []uint32) []byte {
	data := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(data[i*4:], value)
	}
	return data
}

func f32Bytes(values []float32) []byte {
	data := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(value))
	}
	return data
}

type fieldValue struct {
	offset uint32
	value  uint32
}

func f32Field(offset uint32, value float32) fieldValue {
	return fieldValue{offset: offset, value: math.Float32bits(value)}
}

func u32Field(offset, value uint32) fieldValue {
	return fieldValue{offset: offset, value: value}
}

func structBytes(size int, fields ...fieldValue) []byte {
	data := make([]byte, size)
	for _, field := range fields {
		binary.LittleEndian.PutUint32(data[field.offset:], field.value)
	}
	return data
}

func readU32(output map[string][]byte, name string) ([]uint32, error) {
	data, ok := output[name]
	if !ok {
		return nil, fmt.Errorf("result %q is missing", name)
	}
	if len(data)%4 != 0 {
		return nil, fmt.Errorf("result %q has invalid byte length %d", name, len(data))
	}
	values := make([]uint32, len(data)/4)
	for i := range values {
		values[i] = binary.LittleEndian.Uint32(data[i*4:])
	}
	return values, nil
}

func readF32(output map[string][]byte, name string) ([]float32, error) {
	words, err := readU32(output, name)
	if err != nil {
		return nil, err
	}
	values := make([]float32, len(words))
	for i, word := range words {
		values[i] = math.Float32frombits(word)
	}
	return values, nil
}
