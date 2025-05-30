import os
from openai import OpenAI
import modal

def get_llama_client():
    client = OpenAI(
        api_key=os.getenv("LLAMA_API_KEY"),
        base_url=os.getenv("LLAMA_BASE_URL"),
    )
    return client

client = get_llama_client()

def get_completion(prompt, system_prompt=None):
    response = client.chat.completions.create(
        model="Llama-4-Maverick-17B-128E-Instruct-FP8",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
            ],
        max_tokens=2000,
    )
    return response.choices[0].message.content

def get_ops():
    ops = [
        "elementwise_add",
        "elementwise_multiply",
        "matmul",
        "matmul_transpose_a",
        "gelu",
        "relu",
        "leaky_relu",
        "silu",
        "softplus",
        "softsign",
        "tanh",
        "sigmoid",
        "softmax",
        "log_softmax",
        "batch_norm",
        "layer_norm",
        "dropout",
        "average_pooling",
        "max_pooling",
        "reduce_sum",
        "reduce_mean",
        "reduce_max",
        "reduce_product",
        "arg_max",
        "arg_min",
        "conv",
        "scalar_mul",
        "scalar_add",
        "cumsum",
        "cumprod",
        "cum_reverse",
        "cross_entropy_loss",
        "mse_loss",
        "kldiv_loss",
    ]
    return ops


app = modal.App(name="triton-test")
image = modal.Image.debian_slim()
image.pip_install("torch", "triton")

@app.function(gpu="A100", image=image)
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



if __name__ == "__main__":
    main()

