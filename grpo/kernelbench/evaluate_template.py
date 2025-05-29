import torch
import torch.nn as nn
import triton
import triton.language as tl

def evaluate(triton_kernel_code=None, triton_wrapper_code=None, torch_baseline_code=None):
    """
    Evaluate the performance and correctness of generated triton kernel and wrapper.
    
    Args:
        triton_kernel_code: String containing the generated triton kernel code
        triton_wrapper_code: String containing the generated triton wrapper code
        torch_baseline_code: String containing the torch baseline implementation with Model class
        
    Returns:
        dict: Results containing correctness, performance metrics, and any errors
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = {
        'compiles': False,
        'correct': False,
        'torch_time': None,
        'triton_time': None,
        'speedup': None,
        'error': None
    }
    
    try:
        # torch_baseline_code is required
        if not torch_baseline_code:
            results['error'] = "torch_baseline_code is required"
            return results
            
        torch_namespace = {
            'torch': torch,
            'nn': nn,
            'device': device
        }
        exec(torch_baseline_code, torch_namespace)
        
        # Get the Model class and helper functions
        Model = torch_namespace.get('Model')
        get_inputs = torch_namespace.get('get_inputs')
        get_init_inputs = torch_namespace.get('get_init_inputs')
        
        if Model is None or get_inputs is None:
            raise ValueError("Model class or get_inputs function not found in baseline code")
        
        # Initialize the model
        model = Model()
        if hasattr(model, 'to'):
            model = model.to(device)
        
        # Get inputs from the baseline code
        inputs = get_inputs()
        # Move all inputs to device
        inputs = [inp.to(device) for inp in inputs]
        
        # Define torch function
        def torch_func():
            return model(*inputs)
        
        # Get torch baseline result and benchmark
        torch_result = torch_func()
        torch_benchmark = triton.testing.do_bench(torch_func)
        results['torch_time'] = torch_benchmark
        
        # Only proceed if both kernel and wrapper code are provided
        if not triton_kernel_code or not triton_wrapper_code:
            results['error'] = "Both triton_kernel_code and triton_wrapper_code must be provided"
            return results
        
        # Create a namespace for executing the generated code
        exec_namespace = {
            'torch': torch,
            'triton': triton,
            'tl': tl,
            'device': device
        }
        
        # Add all inputs to the namespace with generic names
        for i, inp in enumerate(inputs):
            exec_namespace[f'input_{i}'] = inp
        
        # Also add A, B for backward compatibility (most kernels are 2-input)
        if len(inputs) >= 2:
            exec_namespace['A'] = inputs[0]
            exec_namespace['B'] = inputs[1]
        elif len(inputs) == 1:
            exec_namespace['A'] = inputs[0]
        
        try:
            # Execute the generated kernel and wrapper code
            exec(triton_kernel_code, exec_namespace)
            exec(triton_wrapper_code, exec_namespace)
            
            # Get the triton_wrapper function from the executed namespace
            triton_wrapper_func = exec_namespace.get('triton_wrapper')
            if triton_wrapper_func is None:
                raise ValueError("triton_wrapper function not found in generated code")
            
            results['compiles'] = True
            
            # Test correctness - call triton wrapper with same inputs as torch
            triton_result = triton_wrapper_func(*inputs)
            
            # Check if results are close
            try:
                triton.testing.assert_close(torch_result, triton_result, atol=1e-2, rtol=1e-2)
                results['correct'] = True
            except Exception as e:
                results['correct'] = False
                results['error'] = f"Correctness check failed: {str(e)}"
            
            # Benchmark triton implementation
            triton_benchmark = triton.testing.do_bench(lambda: triton_wrapper_func(*inputs))
            results['triton_time'] = triton_benchmark
            
            # Calculate speedup
            if torch_benchmark > 0:
                results['speedup'] = torch_benchmark / triton_benchmark
            
        except Exception as e:
            results['compiles'] = False
            results['error'] = f"Compilation/execution error: {str(e)}"
    
    except Exception as e:
        results['error'] = f"Evaluation error: {str(e)}"
    
    return results

def evaluate_kernel_performance(kernel_code, wrapper_code, torch_baseline_code):
    """
    Convenience function to evaluate kernel performance with generated code.
    
    Args:
        kernel_code: String containing the triton kernel implementation
        wrapper_code: String containing the triton wrapper implementation
        torch_baseline_code: String containing the torch baseline implementation with Model class
        
    Returns:
        dict: Performance and correctness results
    """
    return evaluate(kernel_code, wrapper_code, torch_baseline_code)



