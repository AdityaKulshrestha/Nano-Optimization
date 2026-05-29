import torch

import triton
import triton.language as tl


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_vectors(
    x_ptr,      # Pointer to first input vector
    y_ptr,      # Pointer to second input vector
    output_ptr, # Pointer to the output vector 
    n_elements, # Size of the vectors
    BLOCK_SIZE: tl.constexpr, # Number of elements each program should process 
):
    # Program ID is used to determine which part of the vector this program should process
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.

    # This program will take a 1D array, process the data in the form of block size
    # Elements will be accessed in the batches of size BLOCK_SIZE
    # Example: BLOCK_SIZE=64
    # Elements: [0:63, 64:127, 128:191, ...]

    block_start = pid * BLOCK_SIZE # Starting index of the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE) 

    # NOTE - Why offsets are required?
    # Because we need it for the load and store operations. We will use it to load the data from the input vectors and store the results in the output vector.

    # Create a mask to guard memory operations against out of bounds accesses.
    mask = offsets < n_elemnts

    # Load x and y from DRAM, masking out any extra elements in case the input is not a multiple of block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y

    # Write it back to DRAM
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    """
    A wrapper function that prepares the data and launches the Triton kernel.
    """
    output = torch.empty_like(x)

    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = x.numel()

    # The SMPD (Sub-Matrix Per Program) is the number of elements each program will process.
    # We can tune this parameter to achieve better performance. A common choice is 1024.
    # For CUDA it is analoguous to the number of threads per block OR CUDA grid size.

    grid = lambda meta: (triton.cdiv(n_elemnts, meta['BLOCK_SIZE']),)

    # NOTE:
    # - Each torch.tensor object is implicitly converted into a pointer to its first element.
    # - 'triton.jit' ed function can be indexed with a launch grid to obtain a callable GPU kernel
    # - Don't forget to pass meta-parameters as keywords arguments

    add_kernel[grid](x, y, output, n_elments, BLOCK_SIZE=1024)
    # We return a handle to z but, since torch.cuda.synchronize() hasn't been called the kernel is still running asynchronously at this point.
    return output

if __name__ == "__main__":
    torch.manual_seed(0)

    size = 98432
    x = torch.randn(size, device=DEVICE)
    y = torch.randn(size, device=DEVICE)

    output_torch = x + y
    output_triton = add(x, y)

    print(output_torch)
    print(output_triton)
    print(f"Max error: {(output_torch - output_triton).abs().max()}")