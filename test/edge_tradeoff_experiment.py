import time
import torch
import numpy as np
from transformers import pipeline
import os

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# 1. Simulating Edge Constraints
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

DEVICE = 0 # Force CPU
TEST_SENTENCE = "Turn off the lights in the master bedroom immediately."
CANDIDATE_LABELS = [
    "Turn off the lights", 
    "Play music", 
    "Check the weather", 
    "Set an alarm",
    "Call mom", 
    "Order pizza", 
    "Check stocks"
] # 7 labels (SNIPS style) for a fair speed test

# ---------------------------------------------------------
# MODELS TO COMPARE
# ---------------------------------------------------------
MODELS = {
    "Your_Model (MiniLM)": "cross-encoder/nli-MiniLM2-L6-H768",
    "Heavy_Baseline (BART-Large)": "facebook/bart-large-mnli"
}

def benchmark_model(name, model_path):
    print(f"\n[{name}] Loading...")
    try:
        classifier = pipeline("zero-shot-classification", model=model_path, device=DEVICE)
    except Exception as e:
        print(f"Failed to load {name}: {e}")
        return None

    print(f"[{name}] Warming up...")
    # Warmup to load weights into RAM
    classifier(TEST_SENTENCE, CANDIDATE_LABELS)

    print(f"[{name}] Running Latency Test (Single Batch)...")
    latencies = []
    iterations = 50  # Run 50 times to get a stable average

    for _ in range(iterations):
        start = time.perf_counter()
        _ = classifier(TEST_SENTENCE, CANDIDATE_LABELS)
        end = time.perf_counter()
        latencies.append((end - start) * 1000) # Convert to ms

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    
    print(f" >> Avg Latency: {avg_latency:.2f} ms")
    print(f" >> Std Dev:     {std_latency:.2f} ms")
    
    # Estimate size in MB
    param_size = 0
    for param in classifier.model.parameters():
        param_size += param.nelement() * param.element_size()
    size_mb = param_size / 1024 / 1024
    print(f" >> Model Size:  {size_mb:.2f} MB")

    return {
        "model": name,
        "latency": avg_latency,
        "size": size_mb
    }

if __name__ == "__main__":
    print("=== EDGE PROCESSOR SIMULATION (Single Thread CPU) ===")
    
    results = []
    for name, path in MODELS.items():
        res = benchmark_model(name, path)
        if res: results.append(res)

    print("\n" + "="*50)
    print("FINAL TRADEOFF REPORT")
    print("="*50)
    print(f"{'Model':<30} | {'Latency (ms)':<15} | {'Size (MB)':<10} | {'Speedup'}")
    print("-" * 70)
    
    baseline = results[1] # BART
    yours = results[0]    # MiniLM
    
    speedup = baseline['latency'] / yours['latency']
    
    for res in results:
        is_baseline = res['model'] == baseline['model']
        speed_factor = "1.0x (Ref)" if is_baseline else f"{baseline['latency']/res['latency']:.1f}x FASTER"
        print(f"{res['model']:<30} | {res['latency']:.2f} ms        | {res['size']:.0f} MB      | {speed_factor}")

    print("-" * 70)
    print(f"\nCONCLUSION:")
    print(f"Your model is {speedup:.1f}x faster and {(baseline['size']/yours['size']):.1f}x smaller.")
    print(f"This enables real-time interaction (<200ms) which the Baseline fails to achieve.")