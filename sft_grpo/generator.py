import json
import random
import asyncio
import aiohttp
import os
from pathlib import Path
from datasets import Dataset
from utils import get_ops, get_completion
from datasets import load_dataset
import time

class SFTDatasetGenerator:
    def __init__(self, ops, num_examples=1000, seed=117, max_concurrent=10):
        self.ops = ops
        self.num_examples = num_examples
        self.max_concurrent = max_concurrent  # Reduced from 50 to 10
        self.requests_per_minute = 3000
        self.xml_format = """
<kernel>
@triton.jit
...
</kernel>
<launch_fn>
...
</launch_fn>
"""
        self.post_prompt = """
Write both the kernel method and the corresponding launch method. 
Include the docstring explaining arguments and expected size, shape, and number of dimensions.
Answer in the following format:\n
"""
        random.seed(seed)
        self.queries_system_prompt = """
You are playing the role of user who is going to ask for a triton kernel for a given pytorch operation. Given an operation, respond with a query a user would ask for a triton kernel for that operation.
"""
        self.responses_system_prompt = """
You are playing the role of a triton kernel expert. 
Given a query, respond with a triton kernel for the given operation. 
It is important that kernel and launch functions are correctly wrapped in their tags.
"""

    async def get_completion_async(self, prompt, system_prompt, session, max_retries=3):
        """Async version of get_completion using aiohttp with retry logic"""
        headers = {
            "Authorization": f"Bearer {os.getenv('LLAMA_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000,
        }
        
        # Fix URL construction to avoid double slashes
        base_url = os.getenv("LLAMA_BASE_URL").rstrip('/')
        url = f"{base_url}/chat/completions"
        
        for attempt in range(max_retries):
            try:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["message"]["content"]
                    elif response.status == 429:
                        # Rate limit hit, wait with exponential backoff
                        wait_time = (2 ** attempt) * 5  # 5, 10, 20 seconds
                        print(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise Exception(f"API request failed with status {response.status}: {await response.text()}")
            except asyncio.TimeoutError:
                if attempt == max_retries - 1:
                    raise Exception("Request timed out after all retries")
                wait_time = (2 ** attempt) * 2
                print(f"Timeout, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                await asyncio.sleep(wait_time)
        
        raise Exception(f"Failed after {max_retries} attempts")

    async def get_response_batch_async(self, queries_batch):
        """Get responses for a batch of queries using asyncio"""
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            # Rate limiter: 3000 requests per minute = 50 requests per second
            request_interval = 60.0 / self.requests_per_minute  # 0.02 seconds between requests
            last_request_time = 0
            
            async def get_single_response(query):
                nonlocal last_request_time
                async with semaphore:
                    # Rate limiting: ensure minimum interval between requests
                    current_time = time.time()
                    time_since_last = current_time - last_request_time
                    if time_since_last < request_interval:
                        await asyncio.sleep(request_interval - time_since_last)
                    last_request_time = time.time()
                    
                    try:
                        prompt = query['text'] + self.post_prompt + self.xml_format
                        response = await self.get_completion_async(prompt, self.responses_system_prompt, session)
                        return (query, response, None)
                    except Exception as e:
                        return (query, None, str(e))
            
            tasks = [get_single_response(query) for query in queries_batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions that were returned
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append((queries_batch[i], None, str(result)))
                    print(f"✗ Failed {i+1}/{len(results)}: {result}")
                else:
                    processed_results.append(result)
                    if result[1] is not None:
                        print(f"✓ Completed {i+1}/{len(results)}")
                    else:
                        print(f"✗ Failed {i+1}/{len(results)}: {result[2]}")
            
            return processed_results

    async def create_synthetic_queries_async(self, k):
        """Async version of create_synthetic_queries"""
        selected_ops = random.choices(self.ops, k=k)
        
        print(f"Creating {k} synthetic queries using async requests...")
        
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            # Rate limiter: 3000 requests per minute = 50 requests per second
            request_interval = 60.0 / self.requests_per_minute  # 0.02 seconds between requests
            last_request_time = 0
            
            async def get_single_query(op, idx):
                nonlocal last_request_time
                async with semaphore:
                    # Rate limiting: ensure minimum interval between requests
                    current_time = time.time()
                    time_since_last = current_time - last_request_time
                    if time_since_last < request_interval:
                        await asyncio.sleep(request_interval - time_since_last)
                    last_request_time = time.time()
                    
                    try:
                        prompt = f"Operation: {op}"
                        completion = await self.get_completion_async(prompt, self.queries_system_prompt, session)
                        return (op, idx, completion, None)
                    except Exception as e:
                        return (op, idx, None, str(e))
            
            tasks = [get_single_query(op, i) for i, op in enumerate(selected_ops)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_queries = {}
            for result in results:
                if isinstance(result, Exception):
                    print(f"✗ Exception: {result}")
                    continue
                    
                op, idx, completion, error = result
                if completion is not None:
                    key = f"{op}_{idx}"
                    all_queries[key] = {
                        'text': completion,
                        'type': 'synthetic',
                        'operation': op
                    }
                    print(f"✓ Query for {op}")
                else:
                    print(f"✗ Failed query for {op}: {error}")
        
        print(f"Created {len(all_queries)} synthetic queries")
        return all_queries

    def create_synthetic_queries(self, k):
        """Sync wrapper for async method"""
        return asyncio.run(self.create_synthetic_queries_async(k))

    def create_convert_queries(self, k):
        data = load_dataset('GPUMODE/KernelBook')['train']
        selected_examples = random.sample(list(data), k)
        
        all_queries = {}
        print(f"Creating {k} convert queries...")
        for i, example in enumerate(selected_examples):
            # Fix UUID handling - convert to string first
            uuid_str = str(example['uuid'])
            print(f"  Query {i+1}/{k} for {uuid_str[:8]}...", end=" ... ", flush=True)
            prompt = f"""
PyTorch code: {example['python_code']}
Convert this PyTorch code to a Triton kernel.
"""
            print("✓")
            all_queries[uuid_str] = {
                'text': prompt,
                'type': 'convert',
                'pytorch_code': example['python_code']
            }
        return all_queries

    def create_sft_queries(self):
        n_synthetic = int(0.6 * self.num_examples)
        n_convert = self.num_examples - n_synthetic
        
        synthetic_queries = self.create_synthetic_queries(k=n_synthetic)
        convert_queries = self.create_convert_queries(k=n_convert)
        
        all_queries = {**synthetic_queries, **convert_queries}
        return all_queries

    def get_response(self, query):
        prompt = query['text'] + self.post_prompt + self.xml_format
        return get_completion(prompt, self.responses_system_prompt)

    def get_response_batch(self, queries_batch):
        """Sync wrapper for async method"""
        return asyncio.run(self.get_response_batch_async(queries_batch))

    def generate_sft_dataset(self, output_path="sft_dataset.jsonl"):
        """Generate complete SFT dataset and save to file"""
        queries = self.create_sft_queries()
        dataset = []
        
        # Convert queries dict to list for batch processing
        query_list = [(key, query) for key, query in queries.items()]
        
        print(f"Generating responses for {len(query_list)} examples using async batches of {self.max_concurrent}...")
        
        # Process in batches
        for i in range(0, len(query_list), self.max_concurrent):
            batch = query_list[i:i + self.max_concurrent]
            batch_queries = [query for key, query in batch]
            
            print(f"\nProcessing batch {i//self.max_concurrent + 1}/{(len(query_list) + self.max_concurrent - 1)//self.max_concurrent}")
            print(f"Batch size: {len(batch)} queries")
            
            start_time = time.time()
            results = self.get_response_batch(batch_queries)
            end_time = time.time()
            
            print(f"Batch completed in {end_time - start_time:.2f} seconds")
            
            # Process results
            for j, (query, response, error) in enumerate(results):
                key = batch[j][0]  # Get the original key
                
                if response is not None:
                    example = {
                        "id": key,
                        "instruction": query['text'],
                        "response": response,
                        "type": query['type'],
                        "operation": query.get('operation', ''),
                        "pytorch_code": query.get('pytorch_code', '')
                    }
                    dataset.append(example)
                else:
                    print(f"Skipping {key} due to error: {error}")
            
            # Save checkpoint after each batch
            self._save_dataset(dataset, f"{output_path}.tmp")
            print(f"Checkpoint saved: {len(dataset)} examples completed")
            
            # Rate limiting: ensure we don't exceed 3000 requests per minute
            if i + self.max_concurrent < len(query_list):
                batch_time = end_time - start_time
                min_batch_time = (len(batch) / 3000) * 60  # Minimum time for this batch size
                if batch_time < min_batch_time:
                    sleep_time = min_batch_time - batch_time
                    print(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
        
        # Final save
        self._save_dataset(dataset, output_path)
        print(f"\nFinal dataset saved: {len(dataset)} examples to {output_path}")
        return dataset

    def _save_dataset(self, dataset, path):
        """Save dataset in JSONL format"""
        with open(path, 'w') as f:
            for example in dataset:
                f.write(json.dumps(example) + '\n')

    def load_dataset(self, path):
        """Load dataset from JSONL file"""
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                dataset.append(json.loads(line.strip()))
        return dataset

    def convert_to_hf_format(self, dataset_path, output_path="sft_hf_dataset"):
        """Convert to HuggingFace datasets format for training"""
        dataset = self.load_dataset(dataset_path)
        
        # Format for instruction tuning
        formatted_data = []
        for example in dataset:
            formatted_example = {
                "prompt": example["instruction"],
                "completion": example["response"],
                "metadata": {
                    "type": example["type"],
                    "operation": example.get("operation", ""),
                    "id": example["id"]
                }
            }
            formatted_data.append(formatted_example)
        
        # Create HuggingFace dataset
        hf_dataset = Dataset.from_list(formatted_data)
        hf_dataset.save_to_disk(output_path)
        
        return hf_dataset


if __name__ == "__main__":
    ops = get_ops()
    
    # Generate large dataset
    generator = SFTDatasetGenerator(ops, num_examples=1000)
    dataset = generator.generate_sft_dataset("data/sft_training_data.jsonl")
    
    # Convert to HF format
    hf_dataset = generator.convert_to_hf_format("data/sft_training_data.jsonl")
    print(f"HF dataset created with {len(hf_dataset)} examples")