import time

import numpy as np

# Graceful import to verify installation
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def check_cupy_installation() -> None:
    """
    Validates CuPy installation, queries GPU hardware properties via CUDA API,
    and performs a benchmark to ensure PCIe data transfer and CUDA cores are fully operational.
    """
    if not GPU_AVAILABLE:
        print("[-] CuPy is NOT installed or cannot find the CUDA runtime.")
        print("    Please check your virtual environment and drivers.")
        return

    print("[+] CuPy imported successfully.")
    print(f"[+] CuPy Version: {cp.__version__}\n")

    # 1. Query GPU Hardware Information via CUDA Runtime API
    try:
        device_id = cp.cuda.runtime.getDevice()
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        
        # Decode the byte string returned by the C++ API
        name = props['name'].decode('utf-8')
        total_vram = props['totalGlobalMem'] / (1024 ** 3)  # Convert bytes to Gigabytes
        
        print("--- GPU Hardware Details ---")
        print(f"[*] Default Device ID : {device_id}")
        print(f"[*] GPU Name          : {name}")
        print(f"[*] Total VRAM        : {total_vram:.2f} GB\n")
        
    except cp.cuda.runtime.CUDARuntimeError as e:
        print(f"[-] Error querying CUDA runtime: {e}")
        return

    # 2. Perform a PCIe transfer and CUDA compute test
    print("--- Memory & Compute Diagnostic ---")
    matrix_size = 5000
    print(f"[*] Generating {matrix_size}x{matrix_size} test matrices...")
    
    try:
        # Allocate on Host (CPU RAM)
        cpu_matrix = np.random.rand(matrix_size, matrix_size).astype(np.float32)
        
        # Transfer to Device (GPU VRAM via PCIe bus)
        start_transfer = time.perf_counter()
        gpu_matrix = cp.asarray(cpu_matrix)
        cp.cuda.Stream.null.synchronize()  # Force CPU to wait for GPU memory allocation
        transfer_time = time.perf_counter() - start_transfer
        print(f"    [v] PCIe Transfer to VRAM successful   ({transfer_time:.4f} seconds)")

        # Warm-up run (GPU needs to initialize the CUDA context on the first operation)
        _ = cp.dot(gpu_matrix, gpu_matrix)
        cp.cuda.Stream.null.synchronize()

        # Execute parallel math on CUDA cores
        start_compute = time.perf_counter()
        gpu_result = cp.dot(gpu_matrix, gpu_matrix)
        cp.cuda.Stream.null.synchronize()  # Force CPU to wait for GPU compute
        compute_time = time.perf_counter() - start_compute
        print(f"    [v] CUDA Compute pipeline successful   ({compute_time:.4f} seconds)")

        # Transfer back to Host (VRAM -> CPU RAM)
        start_pull = time.perf_counter()
        _ = cp.asnumpy(gpu_result)
        pull_time = time.perf_counter() - start_pull
        print(f"    [v] PCIe Transfer to CPU RAM successful({pull_time:.4f} seconds)")
        
        print("\n[+] SUCCESS: CuPy is perfectly configured and actively utilizing the GPU.")

    except cp.cuda.memory.OutOfMemoryError:
        print("\n[-] VRAM Out of Memory Error. The GPU is detected but lacks free memory.")
    except Exception as e:
        print(f"\n[-] An unexpected error occurred during execution: {e}")


if __name__ == "__main__":
    check_cupy_installation()
