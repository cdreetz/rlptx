# SFT Warmup + GRPO


- utilizing \<kernel> <launch_fn> tags for ez extract and test
- dynamic run() method that attempts to parse launch arguements and randomly create tensor/vector sizes


### Intro

- small models (qwen 1.5b) struggle to even write the basics of kernels which makes it hard to even begin to get it to learn
- so plan is to use sft during 'cold start' to get small model to at the very least generate stuff that resembles just enough structure we can parse and verify 
- once it learns to just use @triton.jit, followed by a kernel, followed by a launch method we can move to grpo
- with grpo and warmed up model, start generating kernels and launch methods with a reward system that starts rewarding code that actually compiles, while transitioning from just compiles reward for performance  

### SFT Data

- "Can you implement a {op} triton kernel? Write both the kernel method and the corresponding launch method."
- "Convert this pytorch to a triton kernl. Write both the kernel method and the corresponding launch method."


### Kernel and Launch Testing


- modal sandbox 
- after generating, parse the xml for kernel and launch method, parse launch method for expected arguements, execute in sandbox for pass/fail


### Examples

- test_kernel_2.py and test_kernel_3.py are real examples of generated code and the structure i have landed on for the easiest to generate and validate kernels 