"""Small model simulation test for TurboQuant."""

import time
import numpy as np
from turboquant import KVCacheCompressor

# Small model architecture (roughly Smoll-LLM 135M)
SMALL_LLM = {
    "name": "Smoll-LLM (135M)",
    "num_layers": 12,
    "num_heads": 16,
    "num_kv_heads": 16,
    "head_dim": 32,
    "hidden_dim": 512,
}

def simulate_kv_cache(config: dict, seq_len: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shape = (config["num_layers"], config["num_kv_heads"], seq_len, config["head_dim"])
    scale = 1.0 / np.sqrt(config["head_dim"])
    k_cache = rng.standard_normal(shape) * scale
    v_cache = rng.standard_normal(shape) * scale
    return k_cache, v_cache

def test_compression(config: dict, seq_len: int, k_bits: int, v_bits: int):
    print(f"\n  Testing {config['name']} | seq={seq_len} | K={k_bits}b | V={v_bits}b")
    
    k_cache, v_cache = simulate_kv_cache(config, seq_len)
    head_dim = config["head_dim"]
    
    compressor = KVCacheCompressor(head_dim=head_dim, k_bits=k_bits, v_bits=v_bits)
    
    # Compress
    t0 = time.perf_counter()
    compressed = compressor.compress(k_cache, v_cache)
    t_compress = time.perf_counter() - t0
    
    # Decompress
    t0 = time.perf_counter()
    k_hat, v_hat = compressor.decompress(compressed)
    t_decompress = time.perf_counter() - t0
    
    # Metrics
    k_mse = np.mean((k_cache - k_hat) ** 2)
    v_mse = np.mean((v_cache - v_hat) ** 2)
    
    stats = compressor.memory_stats(seq_len, config["num_layers"], config["num_kv_heads"])
    
    print(f"    K MSE:             {k_mse:.8f}")
    print(f"    V MSE:             {v_mse:.8f}")
    print(f"    Original:          {stats['original_mb']:.3f} MB")
    print(f"    Compressed:        {stats['compressed_mb']:.3f} MB")
    print(f"    Compression Ratio: {stats['compression_ratio']:.1f}x")
    print(f"    Total Time:        {(t_compress + t_decompress)*1000:.2f} ms")

def main():
    print("=" * 60)
    print("TURBOQUANT SMALL MODEL TEST")
    print("=" * 60)
    
    for seq_len in [1024, 4096]:
        for k_bits, v_bits in [(3, 3), (4, 4)]:
            test_compression(SMALL_LLM, seq_len, k_bits, v_bits)
            
    print("\nTest completed successfully.")

if __name__ == "__main__":
    main()
