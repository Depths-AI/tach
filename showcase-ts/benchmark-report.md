# Tach TypeScript showcase benchmark report

Generated: 2026-08-10T16:11:48.615Z

## Summary

- Status: **PASSED**
- Profile: **full**
- Workloads: 5/5
- Correctness: **VERIFIED**
- Geometric-mean acceleration: **98.91x**
- Harness duration: 76.20 s
- Adapter: `amd · rdna-2`
- Host: `win32/x64`, Node `v24.14.1`

## Measurement contract

- The native compiler, generated WGSL, shader module, compute pipeline, initial buffer upload, and JavaScript JIT warmup are completed before timing.
- Every sample records multiple dispatches into one compute pass, submits once, and waits once for queue completion.
- Reported times are medians of separate timed batches. GPU readback and correctness comparison happen after timing.
- GPU values are application-visible batch wall times, so command encoding and submission are included; one-time setup and readback are not.
- The comparison target is the same algorithm in single-threaded TypeScript over typed arrays, not a native SIMD or multithreaded library.

## Results

| Workload | Problem | Samples | Dispatches/batch | WebGPU median | TypeScript median | Acceleration | WebGPU throughput | Correctness |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Particle integration | 262,144 particles × 128 steps | 5 | 128 | 69.0 ms | 515.8 ms | **7.48x** | 1945.18 million component updates/s | PASS: maximum absolute error 1.63e-9 |
| Mandelbrot escape | 768 × 768, limit 192 × 4 renders | 5 | 4 | 8.1 ms | 1.45 s | **179.46x** | 291.27 million pixel-dispatches/s | PASS: 99.925% within one iteration; maximum difference 160 |
| Tiled matrix multiply | 256 × 256 matrices × 4 products | 5 | 4 | 5.8 ms | 448.4 ms | **77.31x** | 23.14 GFLOP/s | PASS: maximum absolute error 0.00e+0 |
| Black–Scholes pricing | 1,048,576 options × 8 valuations | 5 | 8 | 8.0 ms | 1.60 s | **199.40x** | 1048.58 million option valuations/s | PASS: maximum absolute error 3.39e-5 |
| Procedural RGBA composition | 1920 × 1080 RGBA × 3 frames | 5 | 3 | 15.8 ms | 7.23 s | **457.67x** | 393.72 million RGBA pixels/s | PASS: mean RGB error 0.000; 100.000% of pixels within 8; maximum 1 |
