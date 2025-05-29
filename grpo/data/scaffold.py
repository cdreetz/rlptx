import json
import random
from typing import List, Dict
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

CURRICULUM = {
    1: ["add", "multiply", "exp", "sum", "max", "transpose", "gemv", "gemm", "softmax"],
    2: ["fused_softmax", "block_gemm", "online_sum", "tiled_transpose"],
    3: ["online_softmax", "block_attention", "optimized_gemm"],
    4: ["vanilla_attention", "flash_attention", "multi_head_attention"]
}

OP_SPECS = {
    "add": {"ref": "torch.add", "inputs": 2, "shapes": "same"},
    "multiply": {"ref": "torch.mul", "inputs": 2, "shapes": "same"},
    "exp": {"ref": "torch.exp", "inputs": 1, "shapes": "same"},
    "sum": {"ref": "torch.sum", "inputs": 1, "shapes": "reduce"},
    "max": {"ref": "torch.max", "inputs": 1, "shapes": "reduce"},
    "transpose": {"ref": "torch.transpose", "inputs": 1, "shapes": "transpose"},
    "gemv": {"ref": "torch.mv", "inputs": 2, "shapes": "mv"},
    "gemm": {"ref": "torch.matmul", "inputs": 2, "shapes": "mm"},
    "softmax": {"ref": "torch.softmax", "inputs": 1, "shapes": "same"},
}

client = OpenAI(
    api_key=os.getenv("LLAMA_API_KEY"),
    base_url="https://api.llama.com/compat/v1/",
)

def generate_prompt(operation: str) -> str:
    system_prompt = f"""Generate a natural way someone might ask for a Triton kernel that does {operation}. 
Make it sound like a real person asking, not a template. Vary the style - sometimes casual, sometimes formal, sometimes specific, sometimes general.
Just return the prompt, nothing else."""
    
    response = client.chat.completions.create(
        model="Llama-4-Maverick-17B-128E-Instruct-FP8",
        messages=[
            {"role": "system", "content": system_prompt}
        ],
    )
    return response.choices[0].message.content.strip()

def generate_shapes():
    sizes = [128, 256, 512, 768, 1024, 1536, 2048, 4096]
    return {
        'M': random.choice(sizes),
        'N': random.choice(sizes), 
        'K': random.choice(sizes)
    }

def get_shapes_for_op(operation: str, base_shapes: Dict) -> Dict:
    M, N, K = base_shapes['M'], base_shapes['N'], base_shapes['K']
    spec = OP_SPECS.get(operation, {"shapes": "same"})
    
    if spec["shapes"] == "same":
        return {"input_shapes": [(N,)], "output_shape": (N,)}
    elif spec["shapes"] == "reduce":
        return {"input_shapes": [(N,)], "output_shape": ()}
    elif spec["shapes"] == "transpose":
        return {"input_shapes": [(M, N)], "output_shape": (N, M)}
    elif spec["shapes"] == "mv":
        return {"input_shapes": [(M, N), (N,)], "output_shape": (M,)}
    elif spec["shapes"] == "mm":
        return {"input_shapes": [(M, K), (K, N)], "output_shape": (M, N)}
    else:
        return {"input_shapes": [(N,)], "output_shape": (N,)}

def generate_curriculum_dataset(samples_per_task: int = 50) -> List[Dict]:
    dataset = []
    
    for tier, operations in CURRICULUM.items():
        print(f"Generating Tier {tier}...")
        for operation in operations:
            print(f"  {operation}: {samples_per_task} samples")
            for i in range(samples_per_task):
                shapes = generate_shapes()
                prompt = generate_prompt(operation)
                op_shapes = get_shapes_for_op(operation, shapes)
                
                dataset.append({
                    'prompt': prompt,
                    'operation': operation,
                    'tier': tier,
                    'reference_op': OP_SPECS.get(operation, {}).get('ref', 'unknown'),
                    **op_shapes
                })
                
                if i % 10 == 0:
                    print(f"    Generated {i+1}/{samples_per_task}")
    
    return dataset

if __name__ == "__main__":
    dataset = generate_curriculum_dataset(samples_per_task=50)
    
    with open("curriculum_dataset_medium.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(dataset)} examples")
