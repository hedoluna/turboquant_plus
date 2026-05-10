"""Simulating TurboQuant+ on Qwen 3.5 9B architecture with memory optimizations."""

import time
import numpy as np
from turboquant import KVCacheCompressor

# Qwen 3.5 9B dimensions
QWEN_9B = {
    "name": "Qwen 3.5 9B (Dense)",
    "num_layers": 28,
    "num_heads": 16,
    "num_kv_heads": 16,  # GQA
    "head_dim": 128,
}

def compute_mse_memory_efficient(a, b):
    """Compute MSE layer by layer to save memory."""
    total_mse = 0.0
    num_layers = a.shape[0]
    for i in range(num_layers):
        diff = (a[i].astype(np.float32) - b[i].astype(np.float32))
        total_mse += np.mean(diff**2)
    return total_mse / num_layers

def simulate_context(config: dict, seq_len: int, k_bits: int, v_bits: int):
    print(f"\n[SIM] Simulation: {config['name']} | Context: {seq_len} | K={k_bits}b, V={v_bits}b")
    
    # Generate synthetic cache (using float32 to save memory)
    rng = np.random.default_rng(42)
    shape = (config["num_layers"], config["num_kv_heads"], seq_len, config["head_dim"])
    scale = np.float32(1.0 / np.sqrt(config["head_dim"]))
    
    k_cache = (rng.standard_normal(shape) * scale).astype(np.float32)
    v_cache = (rng.standard_normal(shape) * scale).astype(np.float32)
    
    compressor = KVCacheCompressor(head_dim=config["head_dim"], k_bits=k_bits, v_bits=v_bits)
    
    # Compress
    t0 = time.perf_counter()
    compressed = compressor.compress(k_cache, v_cache)
    t_compress = time.perf_counter() - t0
    
    # Decompress
    t0 = time.perf_counter()
    k_hat, v_hat = compressor.decompress(compressed)
    t_decompress = time.perf_counter() - t0
    
    # Metrics (Memory efficient)
    k_mse = compute_mse_memory_efficient(k_cache, k_hat)
    v_mse = compute_mse_memory_efficient(v_cache, v_hat)
    stats = compressor.memory_stats(seq_len, config["num_layers"], config["num_kv_heads"])
    
    print(f"    - Original:        {stats['original_mb']:.1f} MB")
    print(f"    - Compressed:      {stats['compressed_mb']:.1f} MB")
    print(f"    - Ratio:           {stats['compression_ratio']:.2f}x")
    print(f"    - Precision (MSE): K={k_mse:.8f}, V={v_mse:.8f}")
    print(f"    - Total Latency:   {(t_compress + t_decompress)*1000:.2f} ms")

def main():
    print("=" * 70)
    print("TURBOQUANT+ REAL-WORLD ARCHITECTURE SIMULATION (QWEN 3.5 9B)")
    print("=" * 70)
    
    # Test context lengths
    for seq_len in [8192, 16384]:
        try:
            simulate_context(QWEN_9B, seq_len, k_bits=4, v_bits=3)
        except MemoryError:
            print(f"    [!] Skipping seq_len={seq_len} due to system memory limits.")

if __name__ == "__main__":
    main()
