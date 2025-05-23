from datasets import load_dataset
from openai import OpenAI
from util import get_client

client = get_client()

SCOUT = "Llama-4-Scout-17B-16E-Instruct-FP8"
MAVERICK = "Llama-4-Maverick-17B-128E-Instruct-FP8"

SYSTEM_PROMPT = """
We need to create QA pairs, and we have code snippets already that will act as the
'answer' portion of the QA pairs, and we just need to create the 'question' portion.

You are to simulate being a user that wants a Triton kernel implemented.

You will be shown a code snippet that implements a kernel with Triton,
based on what the kernel does and what optimization techniques it utilizes,
respond with an example user question that would be answered with the 
given Triton kernel.

Only respond with the user question and do not write any explanatory text.
"""

def llama_chat(system_prompt, prompt):
    response = client.chat.completions.create(
        model=MAVERICK,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return response


ds = load_dataset("GPUMODE/KernelBook")

# download the kernelbook dataset
# add a new column 'Question'
# for each row, use the value from the 'triton_code' column to pass 
# to the generator to create the 'question' text
# each of the llama models through the api has a 3k requests per minute
# limit, so if you hit rate limits for maverick, switch to scout
# finally once we have processed all rows, upload the dataset to hf hub
