# SFT Warmup + GRPO

- utilizing '<kernel></kernel>' '<launch_fn></launch_fn>' tags for ez extract and test
- dynamic run() method that attempts to parse launch arguements and randomly create tensor/vector sizes

### SFT Data

- "Can you implement a {op} triton kernel? Write both the kernel method and the corresponding launch method."
- "Convert this pytorch to a triton kernl. Write both the kernel method and the corresponding launch method."


### Kernel and Launch Testing


- modal sandbox 
- after generating, parse the xml for kernel and launch method, parse launch method for expected arguements, execute in sandbox for pass/fail