import triton
import triton.language as tl

@triton.jit
def max_pool_stride1_kernel(
    x_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    batch_size,  # Batch size of the input tensor
    channels,  # Number of channels in the input tensor
    height,  # Height of the input tensor
    width,  # Width of the input tensor
    kernel_size,  # Size of the max pooling kernel
    stride: tl.constexpr = 1,  # Stride of the max pooling operation
    BLOCK_SIZE: tl.constexpr = 16  # Block size for Triton kernel
):
    """
    Triton kernel for max pooling with stride 1.

    Args:
    - x_ptr: Pointer to the input tensor of shape (batch_size, channels, height, width)
    - output_ptr: Pointer to the output tensor of shape (batch_size, channels, height, width)
    - batch_size: Batch size of the input tensor
    - channels: Number of channels in the input tensor
    - height: Height of the input tensor
    - width: Width of the input tensor
    - kernel_size: Size of the max pooling kernel
    - stride: Stride of the max pooling operation (default=1)
    - BLOCK_SIZE: Block size for Triton kernel (default=16)

    Expected size, shape, and number of dimensions:
    - Input tensor: (batch_size, channels, height, width)
    - Output tensor: (batch_size, channels, height, width)
    """
    pid = tl.program_id(axis=0)
    batch_idx = pid // (height * channels)
    channel_height_idx = pid % (height * channels)
    channel_idx = channel_height_idx // height
    height_idx = channel_height_idx % height

    # Calculate the output width
    output_width = width

    # Iterate over the output width
    for width_idx in range(0, output_width, BLOCK_SIZE):
        # Calculate the block width
        block_width = min(BLOCK_SIZE, output_width - width_idx)

        # Initialize the max value
        max_val = tl.load(x_ptr + batch_idx * channels * height * width + channel_idx * height * width + height_idx * width + width_idx, mask=(width_idx + tl.arange(0, BLOCK_SIZE) < width), other=-float('inf'))

        # Iterate over the kernel size
        for k_h in range(kernel_size):
            for k_w in range(kernel_size):
                if k_h == 0 and k_w == 0:
                    continue  # Skip the same element as it's already loaded
                # Calculate the neighbor index
                neighbor_height_idx = height_idx + k_h
                neighbor_width_idx = width_idx + k_w

                # Check if the neighbor index is within bounds
                mask = (neighbor_height_idx < height) & (neighbor_width_idx < width) & (neighbor_width_idx >= 0) & (neighbor_height_idx >= 0)
                # Load the neighbor value
                neighbor_val = tl.load(x_ptr + batch_idx * channels * height * width + channel_idx * height * width + neighbor_height_idx * width + neighbor_width_idx, mask=mask, other=-float('inf'))

                # Update the max value
                max_val = tl.maximum(max_val, neighbor_val)

        # Store the max value
        tl.store(output_ptr + batch_idx * channels * height * output_width + channel_idx * height * output_width + height_idx * output_width + width_idx, max_val, mask=(width_idx + tl.arange(0, BLOCK_SIZE) < output_width))

def launch_max_pool_stride1_kernel(x: torch.Tensor, kernel_size: int):
    """
    Launch the Triton kernel for max pooling with stride 1.

    Args:
    - x: Input tensor of shape (batch_size, channels, height, width)
    - kernel_size: Size of the max pooling kernel

    Returns:
    - output: Output tensor of shape (batch_size, channels, height, width)
    """
    batch_size, channels, height, width = x.shape
    output = torch.empty_like(x)

    # Calculate the grid size
    grid_size = batch_size * channels * height

    # Launch the kernel
    max_pool_stride1_kernel[(grid_size,)](
        x, output, batch_size, channels, height, width, kernel_size
    )

    return output