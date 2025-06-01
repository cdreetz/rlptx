import modal
import ast

# Same setup as your existing sandbox
image = modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11").pip_install("torch", "triton", "numpy")
app = modal.App.lookup("triton-sandbox", create_if_missing=True)

def get_function_body(file_path, func_name):
    """Extract function body from a Python file"""
    with open(file_path, 'r') as f:
        source = f.read()
    
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            lines = source.split('\n')
            # Skip the 'def' line, get everything else
            body_lines = lines[node.lineno:node.end_lineno]
            return '\n'.join(body_lines)

# Extract the f() function body
function_body = get_function_body('sft_grpo/model_triton.py', 'f')

# Run it in sandbox
with modal.enable_output():
    sandbox = modal.Sandbox.create(app=app, image=image, gpu="A100")
    
    code = f"""
import torch
import triton
import triton.language as tl
{function_body}
"""
    
    result = sandbox.exec("python", "-c", code)
    print(result.stdout.read())
    sandbox.terminate() 