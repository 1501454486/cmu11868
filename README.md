# CMU 11-868: Large Language Model Systems

This repository documents my implementations for the assignments of the CMU course **11-868: Large Language Model Systems**.

### Useful Links

- **Course Homepage:** [https://llmsystem.github.io/llmsystem2025spring/](https://llmsystem.github.io/llmsystem2025spring/)
- **Homework Tutorial:** [https://llmsystem.github.io/llmsystemhomework/](https://llmsystem.github.io/llmsystemhomework/)
- **My blog notes** [https://blog.mindorigin.top/AI/llmsys](https://blog.mindorigin.top/AI/llmsys)

---

## Key Fixes and Personal Notes

Here are some important notes and fixes I encountered while completing the assignments. These may be helpful for debugging or avoiding common pitfalls.

1.  **HW1 (`autodiff`): Crucial Gradient Initialization**

    In the `autodiff` part of HW1, it is **essential** to initialize all gradients to zero. Failing to do so can cause all backward tests in HW3 to fail unexpectedly.

2.  **HW2: Correcting CUDA Grid Dimensions**

    In HW2, a small but important modification to the grid dimensions in the CUDA kernel is recommended for correctness. Specifically, swap the order of `m` and `p` when calculating `gridDims`.

    **Original:**
    ```cuda
    dim3 gridDims((m + threadsPerBlock - 1) / threadsPerBlock, (p + threadsPerBlock - 1) / threadsPerBlock, batch);
    ```

    **Modified:**
    ```cuda
    dim3 gridDims((p + threadsPerBlock - 1) / threadsPerBlock, (m + threadsPerBlock - 1) / threadsPerBlock, batch);
    ```

3.  **HW4 (`cuda_kernel_ops`): Replacing `pycuda.autoinit`**

    The `import pycuda.autoinit` in HW4 can be intrusive. It's better to replace it with a more compatible PyTorch-based initialization method.

    - **Remove** the line:
      ```python
      import pycuda.autoinit
      ```
    - **Add** the following snippet to ensure proper CUDA initialization via PyTorch:
      ```python
      import torch
      if torch.cuda.is_available():
          # This line gently ensures PyTorch handles CUDA initialization, 
          # which is less intrusive than pycuda.autoinit.
          _ = torch.tensor([1.0]).cuda() 
      ```
    - Remember to remove any redundant `import torch` statements later in the file.

4.  **HW6: Python Version for Conda Environment**

    For HW6, the required environment uses **Python 3.10**, not 3.9. When creating the Conda environment, be sure to specify the correct version:
    ```bash
    conda create --name your_env_name python=3.10
    ```
