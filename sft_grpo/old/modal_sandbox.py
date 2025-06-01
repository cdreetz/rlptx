import ast
import modal
import sys

cuda_version = "12.4.0"
flavor = "devel"
operation_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operation_sys}"

image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.11"
).pip_install("torch", "triton", "numpy")
app = modal.App.lookup("triton-sandbox", create_if_missing=True)

with modal.enable_output():
    sandbox = modal.Sandbox.create(
        app=app, 
        image=image,
        gpu="A100"
    )

print(f"Sandbox ID: {sandbox.object_id}")


bash_ps = sandbox.exec("echo", "hello from bash")
python_ps = sandbox.exec("python", "-c", "print('hello from python')")
torch_ps = sandbox.exec("python", "-c", "import torch; print(torch.__version__); print(torch.cuda.is_available())")
nvidia_smi_ps = sandbox.exec("nvidia-smi")

print(bash_ps.stdout.read(), end="")
print(python_ps.stdout.read(), end="")
print(torch_ps.stdout.read(), end="")
print(nvidia_smi_ps.stdout.read(), end="")
print()

def extract_triton_code(file_path):
    """Extract Triton kernel code including kernel, launch, and run functions"""
    with open(file_path, 'r') as f:
        source = f.read()
    
    lines = source.split('\n')
    tree = ast.parse(source)
    
    # Find imports and global variables
    imports = []
    globals_code = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Only get top-level imports (no indentation)
        if (stripped.startswith('import ') or stripped.startswith('from ')) and len(line) - len(line.lstrip()) == 0:
            imports.append(line.strip())  # Ensure no leading whitespace
        elif 'DEVICE = ' in line and 'triton.runtime.driver.active.get_active_torch_device()' in line:
            globals_code.append(line.strip())  # Ensure no leading whitespace
        elif stripped.startswith('print(') and any(check in stripped for check in ['torch.cuda.is_available()', 'CUDA', 'cuda']) and len(line) - len(line.lstrip()) == 0:
            globals_code.append(line.strip())  # Ensure no leading whitespace
    
    # Find all functions with Triton decorators and key functions
    functions = {}
    decorators = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            
            # Check for triton.jit decorator
            has_triton_jit = False
            for decorator in node.decorator_list:
                if (isinstance(decorator, ast.Attribute) and 
                    isinstance(decorator.value, ast.Name) and 
                    decorator.value.id == 'triton' and 
                    decorator.attr == 'jit'):
                    has_triton_jit = True
                    break
                elif (isinstance(decorator, ast.Name) and 
                      decorator.id == 'triton'):
                    has_triton_jit = True
                    break
            
            # Extract function with its decorator if it has one
            start_line = node.lineno - 1
            end_line = node.end_lineno
            
            # Check for decorator above the function
            if has_triton_jit and start_line > 0:
                prev_line = lines[start_line - 1].strip()
                if prev_line.startswith('@'):
                    start_line -= 1
            
            # Extract and clean up indentation
            func_lines = lines[start_line:end_line]
            
            # Remove common leading whitespace while preserving relative indentation
            if func_lines:
                # Find the minimum indentation (excluding empty lines)
                non_empty_lines = [line for line in func_lines if line.strip()]
                if non_empty_lines:
                    min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
                    # Remove the common indentation
                    cleaned_lines = []
                    for line in func_lines:
                        if line.strip():  # Non-empty line
                            cleaned_lines.append(line[min_indent:] if len(line) >= min_indent else line)
                        else:  # Empty line
                            cleaned_lines.append('')
                    func_lines = cleaned_lines
            
            functions[func_name] = func_lines
            decorators[func_name] = has_triton_jit
    
    return imports, globals_code, functions, decorators

def build_kernel_test(file_path):
    """Build a complete test script from extracted Triton code"""
    imports, globals_code, functions, decorators = extract_triton_code(file_path)
    
    # Find kernel, launch, and run functions
    kernel_funcs = [name for name, has_decorator in decorators.items() if has_decorator]
    launch_funcs = [name for name in functions.keys() if 'launch' in name.lower() and not decorators[name]]
    run_funcs = [name for name in functions.keys() if name == 'run']
    
    print(f"Found kernel functions: {kernel_funcs}")
    print(f"Found launch functions: {launch_funcs}")
    print(f"Found run functions: {run_funcs}")
    
    # Build the complete code
    code_parts = []
    
    # Add imports
    code_parts.extend(imports)
    code_parts.append("")
    
    # Add debug info
    code_parts.extend([
        'print("CUDA available:", torch.cuda.is_available())',
        'print("Triton version:", triton.__version__)',
        ""
    ])
    
    # Add globals
    code_parts.extend(globals_code)
    if globals_code:
        code_parts.append('print("Device:", DEVICE)')
        code_parts.append("")
    
    # Add all kernel functions first
    for func_name in kernel_funcs:
        code_parts.extend(functions[func_name])
        code_parts.append("")
    
    # Add launch functions
    for func_name in launch_funcs:
        code_parts.extend(functions[func_name])
        code_parts.append("")
    
    # Add other functions (like run)
    for func_name in functions.keys():
        if func_name not in kernel_funcs and func_name not in launch_funcs:
            code_parts.extend(functions[func_name])
            code_parts.append("")
    
    # Add run call if run function exists
    if run_funcs:
        code_parts.append("# Execute the test")
        code_parts.append("run()")
    
    return '\n'.join(code_parts)

# Get file path from command line or use default
file_path = sys.argv[1] if len(sys.argv) > 1 else 'kernel_example.py'
print(f"Processing file: {file_path}")

code_template = build_kernel_test(file_path)

print("Generated code preview (first 10 lines):")
print("-" * 40)
for i, line in enumerate(code_template.split('\n')[:10]):
    print(f"{i+1:2d}: '{line}'")
print("-" * 40)

print("Executing kernel test in sandbox...")
print("=" * 50)

# Clean up any existing temporary file first
cleanup_cmd = "rm -f /tmp/test_kernel.py"
sandbox.exec("bash", "-c", cleanup_cmd)

# Write the code to a temporary file in the sandbox
write_file_cmd = f"""cat > /tmp/test_kernel.py << 'EOF'
{code_template}
EOF"""

# Write the file
sandbox.exec("bash", "-c", write_file_cmd)

# Execute the Python file
result = sandbox.exec("python", "/tmp/test_kernel.py")
stdout_content = result.stdout.read()
stderr_content = result.stderr.read()

print("STDOUT:")
print(stdout_content, end="")
if stderr_content:
    print("\nSTDERR:")
    print(stderr_content, end="")

# Clean up the temporary file after execution
sandbox.exec("bash", "-c", "rm -f /tmp/test_kernel.py")

sandbox.terminate()