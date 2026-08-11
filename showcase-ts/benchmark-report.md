# Tach TypeScript showcase benchmark report

Generated: 2026-08-11T09:35:26.287Z

## Summary

- Status: **PASSED**
- Profile: **full**
- Workloads: 5/5
- Correctness: **VERIFIED**
- Geometric-mean acceleration: **188.72x**
- Harness duration: 76.42 s
- Adapter: `amd · rdna-2`
- Host: `win32/x64`, Node `v24.14.1`

## Measurement contract

- The native compiler, generated WGSL, shader module, compute pipeline, initial buffer upload, uniform arena, bind group, and JavaScript JIT warmup are completed before timing.
- Every sample records one or more dispatches into one compute pass, submits once, and waits once for queue completion.
- Reported times are medians of separate timed batches. GPU readback and correctness comparison happen after timing.
- GPU values are application-visible batch wall times, so command encoding and submission are included; one-time setup and readback are not.
- The comparison target is the same algorithm in single-threaded TypeScript over typed arrays, not a native SIMD or multithreaded library.

## Results

| Workload | Problem | Samples | Dispatches/batch | WebGPU median | TypeScript median | Acceleration | WebGPU throughput | Correctness |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Particle integration | 262,144 particles × 128 fused steps | 5 | 1 | 4.9 ms | 607.3 ms | **123.94x** | 27391.37 million component updates/s | PASS: maximum absolute error 1.86e-9 |
| Mandelbrot escape | 768 × 768, limit 192 × 4 renders | 5 | 4 | 8.8 ms | 1.43 s | **161.97x** | 268.10 million pixel-dispatches/s | PASS: 99.925% within one iteration; maximum difference 160 |
| Tiled matrix multiply | 256 × 256 matrices × 4 products | 5 | 4 | 5.9 ms | 530.8 ms | **89.97x** | 22.75 GFLOP/s | PASS: maximum absolute error 0.00e+0 |
| Black–Scholes pricing | 1,048,576 options × 8 valuations | 5 | 8 | 8.4 ms | 1.87 s | **222.57x** | 998.64 million option valuations/s | PASS: maximum absolute error 3.39e-5 |
| Procedural RGBA composition | 1920 × 1080 RGBA × 3 frames | 5 | 3 | 13.4 ms | 7.98 s | **595.59x** | 464.24 million RGBA pixels/s | PASS: mean RGB error 0.000; 100.000% of pixels within 8; maximum 1 |
