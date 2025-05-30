import random
from utils import get_ops, get_completion
from datasets import load_dataset

ops = get_ops()

xml_format = """
<kernel>
@triton.jit
...
</kernel>
<launch_fn>
...
</launch_fn>
"""

class SFTDatasetGenerator:
    def __init__(self, ops, num_examples=10, seed=117):
        self.ops = ops
        self.num_examples = num_examples
        self.xml_format = xml_format
        self.post_prompt = """
        Write both the kernel method and the corresponding launch method. 
        Answer in the following format:\n
        """
        random.seed(seed)
        self.queries_system_prompt = """
        You are playing the role of user who is going to ask for a triton kernel for a given pytorch operation. Given an operation, respond with a query a user would ask for a triton kernel for that operation.
        """
        self.responses_system_prompt = """
        You are playing the role of a triton kernel expert. Given a query, respond with a triton kernel for the given operation.
        """

    def create_synthetic_queries(self, k):
        selected_ops = random.sample(self.ops, k)
        
        all_queries = {}
        for op in selected_ops:
            prompt = f"""
            Operation: {op}
            """
            completion = get_completion(prompt, self.queries_system_prompt)
            all_queries[op] = {
                'text': completion,
                'type': 'synthetic'
            }
        return all_queries

    def create_convert_queries(self, k):
        data = load_dataset('GPUMODE/KernelBook')['train']
        selected_examples = random.sample(list(data), k)
        
        all_queries = {}
        for example in selected_examples:
            prompt = f"""
            PyTorch code: {example['python_code']}
            Convert this PyTorch code to a Triton kernel.
            """
            all_queries[example['uuid']] = {
                'text': prompt,
                'type': 'convert'
            }
        return all_queries

    def create_sft_queries(self):
        n_synthetic = int(0.6 * self.num_examples)  # 60% synthetic queries
        n_convert = self.num_examples - n_synthetic  # 40% convert queries
        
        synthetic_queries = self.create_synthetic_queries(k=n_synthetic)
        convert_queries = self.create_convert_queries(k=n_convert)
        
        # Combine both types of queries
        all_queries = {**synthetic_queries, **convert_queries}
        return all_queries

    def get_response(self, query):
        prompt = query['text'] + self.post_prompt + self.xml_format
        return get_completion(prompt, self.responses_system_prompt)

if __name__ == "__main__":
    dataset = SFTDatasetGenerator(ops, num_examples=2)
    queries = dataset.create_sft_queries()
    for query in queries:
        print(query)
        print(dataset.get_response(queries[query]))
        print()
