import argparse
import os
import polars as pl
import torch

import dictionary_learning.training as dl_training
import trackio
from queue import Empty
dl_training.wandb = trackio

from train_sae import train_sae_for_config

def new_new_wandb_process(config, log_queue, entity, project):
    trackio.init(project=project, config=config, name=config["wandb_name"])
    while True:
        try:
            log = log_queue.get(timeout=1)
            if log == "DONE":
                break
            trackio.log(log)
        except Empty:
            continue
    trackio.finish()

dl_training.new_wandb_process = new_new_wandb_process









def main():
    parser = argparse.ArgumentParser(description="Train a suite of Sparse Autoencoders for plantRNAFM.")
    parser.add_argument(
        "--layer",
        type=int,
        default=6,
        help="Layer to train SAE for. Default: 6"
    )
    parser.add_argument(
        "--sparsity",
        type=int,
        default=32,
        help="k-sparsity to train SAE for. Default: 32"
    )
    parser.add_argument(
        "--expansion_factor",
        type=int,
        default=64,
        help="Expansion factor for the SAE dictionary size. Default: 64"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (e.g., 'cuda', 'cpu', 'mps'). Default: cuda"
    )
    parser.add_argument(
        "--training_steps",
        type=int,
        default=10_000,
        help="Number of training steps. Default: 10000"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for training. Default: 1e-5"
    )
    parser.add_argument(
        "--base_save_dir",
        type=str,
        default="trained_saes",
        help="Base directory to save trained SAEs. Default: trained_saes"
    )

    args = parser.parse_args()

    train_sae_for_config(
        layer=args.layer,
        k_sparsity=args.sparsity,
        sae_expansion_factor=args.expansion_factor,
        device=args.device,
        training_steps=args.training_steps,
        base_save_dir=args.base_save_dir,
        warmup_steps=args.training_steps // 20,
        lr=args.learning_rate
    )

if __name__ == '__main__':
    main()