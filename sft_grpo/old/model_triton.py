import modal

cuda_version = "12.4.0"
flavor = "devel"
operation_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operation_sys}"

cuda_dev_image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.11"
).entrypoint([])

triton_image = (
    cuda_dev_image.pip_install(
        "torch", 
        "numpy",
        "triton",
    )
)

app = modal.App(name="triton-test", image=triton_image)

@app.function(
    gpu="A100"
)
def f():
    import torch
    import triton
    import triton.language as tl
    print(torch.cuda.is_available())

    DEVICE = triton.runtime.driver.active.get_active_torch_device()

    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    def add(x: torch.Tensor, y: torch.Tensor):
        output = torch.empty_like(x)
        assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        return output

    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    output_torch = x + y
    output_triton = add(x, y)
    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(output_torch - output_triton))}')

@app.local_entrypoint()
def main():
    f.remote()