from datasets import load_dataset
from openai import OpenAI
from utils import get_client
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading

client = get_client()
SCOUT = "Llama-4-Scout-17B-16E-Instruct-FP8"
MAVERICK = "Llama-4-Maverick-17B-128E-Instruct-FP8"

KERNEL_TERMS = [
    # Memory optimization
    "memory_coalescing", "shared_memory", "register_blocking", "memory_bank_conflicts", 
    "cache_blocking", "global_memory", "local_memory", "texture_memory", "constant_memory",
    
    # Parallelization patterns
    "thread_block_optimization", "warp_primitives", "grid_stride_loops", "block_reduction",
    "warp_reduction", "cooperative_groups", "occupancy_optimization", "thread_coarsening",
    "loop_unrolling",
    
    # Compute optimization
    "vectorized_operations", "fused_multiply_add", "tensor_cores", "mixed_precision",
    "instruction_level_parallelism", "pipeline_optimization", "prefetching",
    "arithmetic_intensity",
    
    # Matrix operations
    "matrix_multiplication", "batched_operations", "transpose_optimization",
    "block_matrix_multiplication", "strided_operations", "broadcast_operations",
    "element_wise_operations", "reduction_operations", "scan_operations",
    
    # Advanced techniques
    "autotuning", "kernel_fusion", "loop_tiling", "software_pipelining",
    "double_buffering", "asynchronous_operations", "stream_processing",
    "dynamic_parallelism", "persistent_kernels", "cooperative_kernels",
    
    # Specific algorithms
    "convolution", "attention_mechanism", "layernorm", "softmax", "gemm",
    "gemv", "axpy", "dot_product", "cross_entropy", "relu_activation",
    "batch_normalization", "dropout", "embedding_lookup", "sparse_operations",
    
    # Triton specific
    "triton_autotuning", "triton_heuristics", "triton_meta_parameters",
    "triton_program_id", "triton_atomic_operations", "triton_masked_operations",
    "triton_pointer_arithmetic"
]

SYSTEM_PROMPT_QUESTION = """
We need to create QA pairs, and we have code snippets already that will act as the
'answer' portion of the QA pairs, and we just need to create the 'question' portion.
You are to simulate being a user that wants a Triton kernel implemented.
You will be shown a code snippet that implements a kernel with Triton,
based on what the kernel does and what optimization techniques it utilizes,
respond with an example user question that would be answered with the 
given Triton kernel.
Only respond with the user question and do not write any explanatory text.
"""

SYSTEM_PROMPT_TERMS = f"""
You are analyzing Triton GPU kernels to identify relevant optimization techniques and kernel characteristics.
Given a Triton kernel code, identify which of the following terms are relevant or used in the implementation:

{', '.join(KERNEL_TERMS)}

Respond with ONLY a comma-separated list of the relevant terms from the above list. Do not include any explanatory text or terms not in the list.
"""

def llama_chat(system_prompt, triton_code, model=MAVERICK):
    prompt_template = f"Here is the Triton Kernel: {triton_code}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_template},
        ],
    )
    return response.choices[0].message.content

def process_dataset():
    ds = load_dataset("GPUMODE/KernelBook")
    ds = ds['train'].select(range(5))

    
    def process_batch(examples):
        maverick_requests = 0
        scout_requests = 0
        
        def get_question_and_terms(triton_code):
            nonlocal maverick_requests, scout_requests
            
            if maverick_requests < scout_requests:
                model = MAVERICK
                maverick_requests += 2  # We make 2 requests per kernel
            else:
                model = SCOUT
                scout_requests += 2
            
            try:
                question = llama_chat(SYSTEM_PROMPT_QUESTION, triton_code, model)
                terms = llama_chat(SYSTEM_PROMPT_TERMS, triton_code, model)
                return question, terms
            except Exception as e:
                print(f"Error: {e}")
                return "", ""
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(get_question_and_terms, examples['triton_code']))
        
        questions, kernel_terms = zip(*results)
        examples['Question'] = list(questions)
        examples['kernel_terms'] = list(kernel_terms)
        return examples
    
    ds = ds.map(process_batch, batched=True, batch_size=50)
    
    ds.push_to_hub("cdreetz/KernelBook-QA")
    
    return ds

if __name__ == "__main__":
    processed_ds = process_dataset()
