import modal
import ast
import inspect
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

def extract_function_body_from_file(file_path, function_name):
    """
    Extract the body of a function from a Python file using AST parsing.
    Returns the function body as a string without the function definition line.
    """
    with open(file_path, 'r') as file:
        source = file.read()
    
    tree = ast.parse(source)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            # Get the source lines
            lines = source.split('\n')
            start_line = node.lineno - 1  # AST uses 1-based indexing
            end_line = node.end_lineno
            
            # Extract function body (skip the def line and decorators)
            function_lines = lines[start_line:end_line]
            
            # Find the actual start of the function body (after def line)
            body_start = 0
            for i, line in enumerate(function_lines):
                if line.strip().startswith('def '):
                    body_start = i + 1
                    break
            
            # Get the function body
            body_lines = function_lines[body_start:]
            
            # Remove common indentation
            body_text = '\n'.join(body_lines)
            return textwrap.dedent(body_text)
    
    raise ValueError(f"Function '{function_name}' not found in {file_path}")

def create_standalone_script(function_body):
    """
    Create a standalone Python script that can be executed in the sandbox.
    """
    script = f"""
import torch
import triton
import triton.language as tl

def main():
{textwrap.indent(function_body, '    ')}

if __name__ == "__main__":
    main()
"""
    return script

def execute_in_sandbox_method1():
    """
    Method 1: Extract function body and execute as a string
    """
    print("=== Method 1: Execute function body as string ===")
    
    # Extract the function body
    function_body = extract_function_body_from_file('sft_grpo/model_triton.py', 'f')
    
    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            app=app, 
            image=image,
            gpu="A100"
        )
    
    print(f"Sandbox ID: {sandbox.object_id}")
    
    # Execute the function body directly
    python_code = f"""
import torch
import triton
import triton.language as tl

# Function body from f()
{function_body}
"""
    
    # Write the code to a file in the sandbox and execute it
    sandbox.exec("python", "-c", f"exec('''{python_code}''')")
    
    # Get the output
    result = sandbox.exec("python", "-c", f"exec('''{python_code}''')")
    print("Output:")
    print(result.stdout.read())
    
    sandbox.terminate()

def execute_in_sandbox_method2():
    """
    Method 2: Create a complete script file and upload it
    """
    print("=== Method 2: Create and upload script file ===")
    
    # Extract the function body
    function_body = extract_function_body_from_file('sft_grpo/model_triton.py', 'f')
    
    # Create standalone script
    script_content = create_standalone_script(function_body)
    
    # Write script to local file
    with open('/tmp/triton_script.py', 'w') as f:
        f.write(script_content)
    
    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            app=app, 
            image=image,
            gpu="A100"
        )
    
    print(f"Sandbox ID: {sandbox.object_id}")
    
    # Upload the script to the sandbox
    sandbox.exec("mkdir", "-p", "/tmp")
    
    # Copy file content (since we can't directly upload files, we'll write it)
    escaped_content = script_content.replace("'", "\\'").replace('"', '\\"')
    sandbox.exec("bash", "-c", f"cat > /tmp/triton_script.py << 'EOF'\n{script_content}\nEOF")
    
    # Execute the script
    result = sandbox.exec("python", "/tmp/triton_script.py")
    print("Output:")
    print(result.stdout.read())
    
    sandbox.terminate()

def execute_in_sandbox_method3():
    """
    Method 3: Direct execution with inline code
    """
    print("=== Method 3: Direct inline execution ===")
    
    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            app=app, 
            image=image,
            gpu="A100"
        )
    
    print(f"Sandbox ID: {sandbox.object_id}")
    
    # Execute the triton code directly
    triton_code = '''
import torch
import triton
import triton.language as tl
print(torch.cuda.is_available())

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
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
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
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
print(f"The maximum difference between torch and triton is {torch.max(torch.abs(output_torch - output_triton))}")
'''
    
    result = sandbox.exec("python", "-c", triton_code)
    print("Output:")
    print(result.stdout.read())
    
    sandbox.terminate()

if __name__ == "__main__":
    # Try different methods
    try:
        execute_in_sandbox_method1()
    except Exception as e:
        print(f"Method 1 failed: {e}")
    
    try:
        execute_in_sandbox_method2()
    except Exception as e:
        print(f"Method 2 failed: {e}")
    
    try:
        execute_in_sandbox_method3()
    except Exception as e:
        print(f"Method 3 failed: {e}") 