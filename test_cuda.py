import torch
import time

def check_cuda():
    """
    检查 CUDA 是否可用，并打印相关信息。
    """
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"CUDA version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA device(s).")
        for i in range(device_count):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device: {current_device} ({torch.cuda.get_device_name(current_device)})")
        return True
    else:
        print("CUDA is not available. PyTorch is using CPU.")
        return False

def performance_test():
    """
    执行一个简单的性能测试来比较 CPU 和 GPU 上的矩阵乘法。
    """
    print("\n--- Starting Performance Test ---")
    # 定义一个较大的矩阵尺寸
    matrix_size = 4096
    x = torch.randn(matrix_size, matrix_size)
    y = torch.randn(matrix_size, matrix_size)

    # --- CPU 性能测试 ---
    print(f"Performing matrix multiplication ({matrix_size}x{matrix_size}) on CPU...")
    start_time_cpu = time.time()
    _ = torch.matmul(x, y)
    end_time_cpu = time.time()
    cpu_time = end_time_cpu - start_time_cpu
    print(f"CPU time: {cpu_time:.4f} seconds")

    # --- GPU 性能测试 ---
    device = torch.device("cuda")
    x_gpu = x.to(device)
    y_gpu = y.to(device)
    
    # 预热 GPU，确保计时准确
    print("Warming up GPU...")
    for _ in range(5):
        _ = torch.matmul(x_gpu, y_gpu)
    torch.cuda.synchronize() # 等待 GPU 操作完成

    print(f"Performing matrix multiplication ({matrix_size}x{matrix_size}) on GPU...")
    start_time_gpu = time.time()
    _ = torch.matmul(x_gpu, y_gpu)
    torch.cuda.synchronize() # 确保 matmul 操作完成再计时
    end_time_gpu = time.time()
    gpu_time = end_time_gpu - start_time_gpu
    print(f"GPU time: {gpu_time:.4f} seconds")
    
    print("\n--- Performance Test Summary ---")
    if gpu_time > 0:
        print(f"GPU is approximately {cpu_time / gpu_time:.2f} times faster than CPU for this task.")
    else:
        print("GPU execution was too fast to measure accurately.")


if __name__ == "__main__":
    # if check_cuda():
    #     performance_test()


    # 测试多卡 + 大张量 + 混合精度
    device = torch.device("cuda")
    dtype = torch.float64  # Use float64 to avoid overflow

    for i in range(torch.cuda.device_count()):
        x = torch.randn(8192, 8192, device=device, dtype=dtype)
        y = torch.randn(8192, 8192, device=device, dtype=dtype)
        z = torch.matmul(x, y)
        print(f"GPU {i} matmul OK, result sum: {z.sum().item()}")