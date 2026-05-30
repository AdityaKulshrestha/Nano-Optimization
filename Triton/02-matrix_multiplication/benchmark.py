import torch

import triton
import triton.language as tl

from main import matmul, is_cuda

DEVICE = triton.runtime.driver.active.get_active_torch_device()
TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")

ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'

configs = []

for fp8_inputs in [False, True]:
    if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
        continue
    
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(2, 33)],     # Different possible values for `x_name`
            line_arg='provider',
            line_vals=['triton'] if fp8_inputs else [ref_lib.lower(), 'triton'],  # Argument name to use for grouping the results with different lines
            line_names=['Triton (FP8)' if fp8_inputs else 'Triton', ref_lib],  # Legend for the lines
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS", 
            plot_name="matmul-performance-" + 
            ("fp16" if not fp8_inputs else "fp8"),
            args={
                "fp8_inputs": fp8_inputs,
            }
        )
    )


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)

    if TORCH_HAS_FP8 and fp8_inputs:
        a = a.to(torch.float8_e5m2)
        b = b.T
        b = b.to(torch.float8_e5m2)

    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)

    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)

    return perf(ms), perf(max_ms), perf(min_ms)




if __name__ == "__main__":
    # We can now run the decorated function above. Pass `print_data

    benchmark.run(print_data=True, save_path='./results/')
