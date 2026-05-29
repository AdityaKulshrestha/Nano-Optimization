import torch
import triton
import triton.language as tl

from main import add


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],       # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],    # Different possible values for x_name,
        x_log = True,              # Whether x axis should be in log scale. 
        line_arg='provider',   # Argument name to use for grouping the results with different lines.    
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('orange', '-')]
        ylabel='GB/s',        # Label for the y-axis.
        plot_name="vector_addition_performance",    # Name for the generated plot.
        args={},    
    )
)


def benchmark(size, provider):
    x = torch.randn(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(labmda: x + y, quantiles=quantiles)
    
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    
    gbps = lambda ms: 3 * x.numel() * x.element_size() / (ms * 1e6) # 3 comes from the fact that we are reading two vectors and writing one vector.
    return gbps(ms), gbps(min_ms), gbps(max_ms)


if __name__ == "__main__":
    # We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or
    # `save_path='/path/to/results/' to save them to disk along with raw CSV data:
    benchmark.run(print_data=True, save_path='./results/')
    triton.testing.run_benchmarks()