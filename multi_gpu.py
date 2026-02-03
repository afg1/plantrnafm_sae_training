import os
import argparse
import itertools
import torch.multiprocessing as mp
from train_sae import train_sae_for_config  # Assuming your main.py is saved as train_sae.py
import torch
import time

import click
import yaml
from pathlib import Path

def load_config(config_path):
    """Safely loads the yaml config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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
            training_steps=50000,
            lr=lr,
            base_save_dir=base_save_dir,
            warmup_steps=50_000 // 20,
        )

@click.command()
@click.option(
    '--config', '-c',
    type=click.Path(exists=True, path_type=Path),
    default="config.yaml",
    help="Path to the YAML configuration file."
)
def main(config):
    # Load the dictionary from the file path provided by click
    params = load_config(config)

    # unpacking parameters for clarity
    learning_rates = params.get('learning_rates', [])
    expansions = params.get('expansions', [])
    layers = params.get('layers', [])
    sparsities = params.get('sparsities', [])
    base_save_dir = params.get('base_save_dir', "trained_saes")

    # Example: Print to verify loading
    print(f"Loaded configuration from {config}:")
    print(f"  • Learning Rates: {learning_rates}")
    print(f"  • Expansions: {expansions}")
    print(f"  • Layers: {layers}")
    print(f"  • Sparsities: {sparsities}")
    print(f"  • Save Directory: {base_save_dir}")

    

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
