import modal
import ast
import textwrap

# Setup the same image as in model_triton.py
cuda_version = "12.4.0"
flavor = "devel"
operation_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operation_sys}"

image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.11"
).pip_install("torch", "triton", "numpy")

app = modal.App.lookup("triton-sandbox", create_if_missing=True)

def extract_function_body(file_path, function_name):
    """
    Extract the body of a function from a Python file.
    Returns the function body as executable code.
    """
    with open(file_path, 'r') as file:
        source = file.read()
    
    tree = ast.parse(source)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            # Get the source lines
            lines = source.split('\n')
            start_line = node.lineno - 1
            end_line = node.end_lineno
            
            # Extract function lines
            function_lines = lines[start_line:end_line]
            
            # Find the start of the function body (skip decorators and def line)
            body_start = 0
            for i, line in enumerate(function_lines):
                if line.strip().startswith('def '):
                    body_start = i + 1
                    break
            
            # Get just the body
            body_lines = function_lines[body_start:]
            body_text = '\n'.join(body_lines)
            
            # Remove common indentation
            return textwrap.dedent(body_text)
    
    raise ValueError(f"Function '{function_name}' not found")

def run_function_in_sandbox():
    """
    Parse the f() function and execute it in a Modal sandbox.
    """
    print("Extracting function body from model_triton.py...")
    
    # Extract the function body
    function_body = extract_function_body('sft_grpo/model_triton.py', 'f')
    
    print("Function body extracted:")
    print("=" * 50)
    print(function_body)
    print("=" * 50)
    
    # Create the complete code to execute
    complete_code = f"""
import torch
import triton
import triton.language as tl

# Extracted function body from f()
{function_body}
"""
    
    print("Creating sandbox...")
    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            app=app, 
            image=image,
            gpu="A100"
        )
    
    print(f"Sandbox ID: {sandbox.object_id}")
    print("Executing code in sandbox...")
    
    # Execute the code
    result = sandbox.exec("python", "-c", complete_code)
    
    print("Output:")
    print(result.stdout.read())
    
    if result.stderr.read():
        print("Errors:")
        print(result.stderr.read())
    
    print("Terminating sandbox...")
    sandbox.terminate()

if __name__ == "__main__":
    run_function_in_sandbox() 