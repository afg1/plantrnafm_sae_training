import os
import argparse
import itertools
import torch.multiprocessing as mp
from train_sae import train_sae_for_config  # Assuming your main.py is saved as train_sae.py
import torch
import time

def worker(gpu_id, configs):
    """Worker function to run training on a specific GPU."""
    time.sleep(gpu_id * 30)
    for config in configs:
        layer, sparsity, expansion, lr, base_save_dir = config
        
        train_sae_for_config(
            layer=layer,
            k_sparsity=sparsity,
            device=f"cuda:{gpu_id}",
            sae_expansion_factor=expansion,
            training_steps=500000,
            lr=lr,
            base_save_dir=base_save_dir,
        )

def main():
    learning_rates = [1e-6, 1e-5, 1e-4]
    expansions = [16, 32, 64]
    layers = list(range(12))  # 0 to 11
    sparsities = [32, 64, 128, 256, 512]
    base_save_dir = ["trained_saes"]

    # Generate all possible configurations
    configs = list(itertools.product(layers, sparsities, expansions, learning_rates, base_save_dir))
    print(f"Total configurations to run: {len(configs)}")
    
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs. Starting parallel training...")

    # Split configs into chunks for each GPU
    chunks = [[] for _ in range(num_gpus)]
    for i, config in enumerate(configs):
        chunks[i % num_gpus].append(config)
    print(f"Distributed configurations into {len(chunks)} chunks for GPUs.")
    print(f"Chunk sizes: {[len(chunk) for chunk in chunks]}")
    # Start a process for each GPU
    processes = []
    for gpu_id, chunk in enumerate(chunks):
        p = mp.Process(target=worker, args=(gpu_id, chunk))
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
