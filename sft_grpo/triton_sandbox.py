import ast
import modal
import sys
import os
import tempfile
from typing import Dict, List, Tuple, Optional

class TritonSandbox:
    """A sandbox environment for testing Triton kernels and launchers"""
    
    def __init__(self, gpu_type: str = "A100"):
        self.cuda_version = "12.4.0"
        self.flavor = "devel"
        self.operation_sys = "ubuntu22.04"
        self.tag = f"{self.cuda_version}-{self.flavor}-{self.operation_sys}"
        self.gpu_type = gpu_type
        
        # Create Modal image with Triton dependencies
        self.image = modal.Image.from_registry(
            f"nvidia/cuda:{self.tag}", add_python="3.11"
        ).pip_install(
            "torch", 
            "triton", 
            "numpy",
            "pytest",  # For potential testing
            "matplotlib",  # For potential visualization
        )
        
        self.app = modal.App.lookup("triton-sandbox", create_if_missing=True)
        self.sandbox = None
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
    
    def start(self):
        """Start the sandbox"""
        with modal.enable_output():
            self.sandbox = modal.Sandbox.create(
                app=self.app,
                image=self.image,
                gpu=self.gpu_type
            )
        print(f"Sandbox started with ID: {self.sandbox.object_id}")
        
        # Test basic functionality
        self._test_environment()
    
    def stop(self):
        """Stop the sandbox"""
        if self.sandbox:
            self.sandbox.terminate()
            print("Sandbox terminated")
    
    def _test_environment(self):
        """Test that the sandbox environment is working"""
        print("Testing sandbox environment...")
        
        # Test CUDA availability
        cuda_test = self.sandbox.exec("python", "-c", 
            "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')")
        print(cuda_test.stdout.read(), end="")
        
        # Test Triton import
        triton_test = self.sandbox.exec("python", "-c", 
            "import triton; print(f'Triton version: {triton.__version__}')")
        print(triton_test.stdout.read(), end="")
        print()
    
    def extract_triton_code(self, code_content: str) -> Tuple[List[str], List[str], Dict[str, List[str]], Dict[str, bool]]:
        """Extract Triton kernel code including imports, globals, and functions"""
        lines = code_content.split('\n')
        
        try:
            tree = ast.parse(code_content)
        except SyntaxError as e:
            print(f"Syntax error in code: {e}")
            return [], [], {}, {}
        
        # Find imports and global variables
        imports = []
        globals_code = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Only get top-level imports (no indentation)
            if (stripped.startswith('import ') or stripped.startswith('from ')) and len(line) - len(line.lstrip()) == 0:
                imports.append(line.strip())
            elif any(pattern in line for pattern in ['DEVICE =', 'device =', 'torch.cuda.is_available()', 'torch.manual_seed']) and len(line) - len(line.lstrip()) == 0:
                globals_code.append(line.strip())
        
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
                          decorator.id in ['triton', 'jit']):
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
                    non_empty_lines = [line for line in func_lines if line.strip()]
                    if non_empty_lines:
                        min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
                        cleaned_lines = []
                        for line in func_lines:
                            if line.strip():
                                cleaned_lines.append(line[min_indent:] if len(line) >= min_indent else line)
                            else:
                                cleaned_lines.append('')
                        func_lines = cleaned_lines
                
                functions[func_name] = func_lines
                decorators[func_name] = has_triton_jit
        
        return imports, globals_code, functions, decorators
    
    def build_test_script(self, code_content: str, test_function_name: str = "run") -> str:
        """Build a complete test script from Triton code"""
        imports, globals_code, functions, decorators = self.extract_triton_code(code_content)
        
        # Categorize functions
        kernel_funcs = [name for name, has_decorator in decorators.items() if has_decorator]
        launch_funcs = [name for name in functions.keys() if 'launch' in name.lower() and not decorators[name]]
        test_funcs = [name for name in functions.keys() if name in [test_function_name, 'test', 'main']]
        other_funcs = [name for name in functions.keys() if name not in kernel_funcs + launch_funcs + test_funcs]
        
        print(f"Found kernel functions: {kernel_funcs}")
        print(f"Found launch functions: {launch_funcs}")
        print(f"Found test functions: {test_funcs}")
        print(f"Found other functions: {other_funcs}")
        
        # Build the complete code
        code_parts = []
        
        # Add standard imports if not present
        standard_imports = [
            'import torch',
            'import triton',
            'import triton.language as tl'
        ]
        
        for imp in standard_imports:
            if not any(imp in existing for existing in imports):
                code_parts.append(imp)
        
        # Add user imports
        code_parts.extend(imports)
        code_parts.append("")
        
        # Add debug info
        code_parts.extend([
            'print("CUDA available:", torch.cuda.is_available())',
            'print("Triton version:", triton.__version__)',
            'print("PyTorch version:", torch.__version__)',
            ""
        ])
        
        # Add globals with fallback device setup
        if not any('DEVICE' in g for g in globals_code):
            code_parts.append('DEVICE = "cuda" if torch.cuda.is_available() else "cpu"')
        code_parts.extend(globals_code)
        code_parts.append('print("Device:", DEVICE)')
        code_parts.append("")
        
        # Add all functions in order: kernels, launches, others (but not user-provided tests)
        for func_group in [kernel_funcs, launch_funcs, other_funcs]:
            for func_name in func_group:
                code_parts.extend(functions[func_name])
                code_parts.append("")
        
        # Add user-provided test functions if they exist
        if test_funcs:
            for func_name in test_funcs:
                code_parts.extend(functions[func_name])
                code_parts.append("")
        
        # Generate dynamic run function if no test function exists or if we want to override
        if not test_funcs or test_function_name == "run":
            code_parts.extend(self._generate_dynamic_run_function(launch_funcs))
            code_parts.append("")
        
        # Add execution logic
        code_parts.append("# Execute the test")
        code_parts.append(f"if __name__ == '__main__':")
        if test_funcs and test_function_name != "run":
            code_parts.append(f"    {test_funcs[0]}()")
        else:
            code_parts.append("    run()")
        
        return '\n'.join(code_parts)
    
    def _generate_dynamic_run_function(self, launch_funcs: List[str]) -> List[str]:
        """Generate a dynamic run function that inspects launch function signatures"""
        if not launch_funcs:
            return [
                "def run():",
                "    print('No launch function found. Please provide a launch function.')",
                "    return"
            ]
        
        # Use the first launch function found
        launch_func_name = launch_funcs[0]
        
        run_function = [
            "def run():",
            "    import inspect",
            "    import torch",
            "    ",
            f"    launch_func_name = '{launch_func_name}'",
            f"    # Get the signature of the {launch_func_name} function",
            f"    sig = inspect.signature({launch_func_name})",
            f"    print('Testing {launch_func_name} function:')",
            "    print('Function signature:', sig)",
            "    ",
            "    # Set random seed for reproducibility",
            "    torch.manual_seed(42)",
            "    ",
            "    # Generate example arguments based on type annotations",
            "    args = {}",
            "    default_size = 1024  # Default tensor size",
            "    ",
            "    for param_name, param in sig.parameters.items():",
            "        annotation = param.annotation",
            "        ",
            "        if annotation == torch.Tensor or str(annotation) == \"<class 'torch.Tensor'>\":",
            "            # Generate a random tensor on the correct device",
            "            tensor = torch.rand(default_size, device=DEVICE, dtype=torch.float32)",
            "            args[param_name] = tensor",
            "            print(f'Generated {param_name}: torch.Tensor of shape {tensor.shape} on {tensor.device}')",
            "        ",
            "        elif hasattr(annotation, '__origin__') and annotation.__origin__ is torch.Tensor:",
            "            # Handle more complex tensor type hints if any",
            "            tensor = torch.rand(default_size, device=DEVICE, dtype=torch.float32)",
            "            args[param_name] = tensor",
            "            print(f'Generated {param_name}: torch.Tensor of shape {tensor.shape} on {tensor.device}')",
            "        ",
            "        elif annotation == int or annotation == 'int':",
            "            # Generate a reasonable integer value",
            "            value = default_size",
            "            args[param_name] = value",
            "            print(f'Generated {param_name}: {value}')",
            "        ",
            "        elif annotation == float or annotation == 'float':",
            "            # Generate a reasonable float value",
            "            value = 1.0",
            "            args[param_name] = value",
            "            print(f'Generated {param_name}: {value}')",
            "        ",
            "        elif 'Tensor' in str(annotation) or annotation == inspect.Parameter.empty:",
            "            # For unrecognized types or no annotation, default to tensor",
            "            print(f'Unknown/missing type annotation for {param_name}, defaulting to tensor')",
            "            tensor = torch.rand(default_size, device=DEVICE, dtype=torch.float32)",
            "            args[param_name] = tensor",
            "            print(f'Generated {param_name}: torch.Tensor of shape {tensor.shape} on {tensor.device}')",
            "        ",
            "        else:",
            "            print(f'Warning: Unsupported type {annotation} for {param_name}, skipping')",
            "    ",
            "    if not args:",
            "        print('No arguments generated. The launch function may have no parameters or unsupported types.')",
            "        return",
            "    ",
            "    print(f'\\nCalling {launch_func_name} with generated arguments...')",
            "    ",
            "    try:",
            f"        # Call the {launch_func_name} function with generated arguments",
            f"        result = {launch_func_name}(**args)",
            "        ",
            "        print(f'\\nKernel execution completed successfully!')",
            "        print(f'Result type: {type(result)}')",
            "        if isinstance(result, torch.Tensor):",
            "            print(f'Result shape: {result.shape}')",
            "            print(f'Result device: {result.device}')",
            "            print(f'Result dtype: {result.dtype}')",
            "            print(f'Result sample values: {result.flatten()[:5]}')",
            "            ",
            "            # Try to validate against a simple operation if we have 2 tensor inputs",
            "            tensor_args = [v for v in args.values() if isinstance(v, torch.Tensor)]",
            "            if len(tensor_args) >= 2:",
            "                # Try common operations",
            "                operations = [",
            "                    ('addition', lambda a, b: a + b),",
            "                    ('multiplication', lambda a, b: a * b),",
            "                    ('subtraction', lambda a, b: a - b),",
            "                ]",
            "                ",
            "                for op_name, op_func in operations:",
            "                    try:",
            "                        expected = op_func(tensor_args[0], tensor_args[1])",
            "                        if result.shape == expected.shape:",
            "                            max_diff = torch.max(torch.abs(result - expected)).item()",
            "                            if max_diff < 1e-5:",
            "                                print(f'✓ Kernel appears to implement {op_name} (max diff: {max_diff:.2e})')",
            "                                break",
            "                            else:",
            "                                print(f'  Checked {op_name}: max diff {max_diff:.2e} (not a match)')",
            "                    except:",
            "                        pass",
            "                else:",
            "                    print('  Could not automatically identify the operation')",
            "        else:",
            "            print(f'Result: {result}')",
            "        ",
            "        print('\\n✓ Test completed successfully!')",
            "        ",
            "    except Exception as e:",
            "        print(f'\\n✗ Error during kernel execution: {e}')",
            "        import traceback",
            "        traceback.print_exc()",
            "        raise"
        ]
        
        return run_function
    
    def test_kernel(self, code_content: str, test_function_name: str = "run", verbose: bool = True) -> Dict:
        """Test a Triton kernel in the sandbox"""
        if not self.sandbox:
            raise RuntimeError("Sandbox not started. Call start() first or use as context manager.")
        
        # Build test script
        test_script = self.build_test_script(code_content, test_function_name)
        
        if verbose:
            print("Generated test script preview (first 15 lines):")
            print("-" * 50)
            for i, line in enumerate(test_script.split('\n')[:15]):
                print(f"{i+1:2d}: {line}")
            print("-" * 50)
        
        print("Executing kernel test in sandbox...")
        print("=" * 60)
        
        # Clean up any existing temporary file
        cleanup_cmd = "rm -f /tmp/test_kernel.py"
        self.sandbox.exec("bash", "-c", cleanup_cmd)
        
        # Write the code to a temporary file in the sandbox
        write_file_cmd = f"""cat > /tmp/test_kernel.py << 'EOF'
{test_script}
EOF"""
        
        # Write and execute
        self.sandbox.exec("bash", "-c", write_file_cmd)
        result = self.sandbox.exec("python", "/tmp/test_kernel.py")
        
        # Capture output
        stdout_content = result.stdout.read()
        stderr_content = result.stderr.read()
        
        # Clean up
        self.sandbox.exec("bash", "-c", "rm -f /tmp/test_kernel.py")
        
        # Display results
        print("STDOUT:")
        print(stdout_content, end="")
        if stderr_content:
            print("\nSTDERR:")
            print(stderr_content, end="")
        
        # Return structured results
        return {
            'success': len(stderr_content.strip()) == 0,
            'stdout': stdout_content,
            'stderr': stderr_content,
            'test_script': test_script
        }
    
    def test_kernel_from_file(self, file_path: str, test_function_name: str = "run", verbose: bool = True) -> Dict:
        """Test a Triton kernel from a file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r') as f:
            code_content = f.read()
        
        print(f"Testing kernel from file: {file_path}")
        return self.test_kernel(code_content, test_function_name, verbose)
    
    def benchmark_kernel(self, code_content: str, iterations: int = 100) -> Dict:
        """Benchmark a Triton kernel (placeholder for future implementation)"""
        # This could be extended to include performance benchmarking
        print(f"Benchmarking not yet implemented. Would run {iterations} iterations.")
        return self.test_kernel(code_content)


def main():
    """Main function for command-line usage"""
    if len(sys.argv) < 2:
        print("Usage: python triton_sandbox.py <kernel_file.py> [test_function_name]")
        print("   or: python triton_sandbox.py --interactive")
        sys.exit(1)
    
    if sys.argv[1] == "--interactive":
        print("Interactive mode - enter your Triton kernel code:")
        print("(Type 'END' on a new line to finish)")
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == 'END':
                    break
                lines.append(line)
            except EOFError:
                break
        
        code_content = '\n'.join(lines)
        test_function_name = "run"
    else:
        file_path = sys.argv[1]
        test_function_name = sys.argv[2] if len(sys.argv) > 2 else "run"
        
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found")
            sys.exit(1)
        
        with open(file_path, 'r') as f:
            code_content = f.read()
    
    # Test the kernel
    with TritonSandbox() as sandbox:
        if len(sys.argv) > 1 and sys.argv[1] != "--interactive":
            result = sandbox.test_kernel_from_file(sys.argv[1], test_function_name)
        else:
            result = sandbox.test_kernel(code_content, test_function_name)
        
        print("\n" + "=" * 60)
        print(f"Test {'PASSED' if result['success'] else 'FAILED'}")
        if not result['success']:
            print("Check the STDERR output above for error details.")


if __name__ == "__main__":
    main() 