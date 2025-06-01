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





if __name__ == "__main__":
    main()

