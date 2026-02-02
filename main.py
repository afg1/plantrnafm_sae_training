import argparse
import os
import polars as pl
import torch

import dictionary_learning.training as dl_training
import trackio
from queue import Empty
dl_training.wandb = trackio


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

from nnsight import LanguageModel
from dictionary_learning import ActivationBuffer
from dictionary_learning.trainers.top_k import TopKTrainer, AutoEncoderTopK
from dictionary_learning.training import trainSAE
from transformers import AutoModelForMaskedLM, AutoTokenizer


def sequence_generator(data: pl.DataFrame):
    for row in data.iter_rows(named=True):
        yield row['sequence']

def train_sae_for_config(
    layer: int,
    k_sparsity: int,
    device: str,
    model_name: str = "yangheng/PlantRNA-FM",
    activation_dim: int = 480,
    llm_batch_size: int = 64,
    sae_batch_size: int = 1024,
    sae_expansion_factor: int = 64,
    training_steps: int = 10_000,
    lr: float = 1e-5,
    warmup_steps: int = 500,
    wandb_project: str = "plantrna_fm_sae",
    log_steps: int = 500,
    base_save_dir: str = "trained_saes"
):
    print(f"Starting training for Layer: {layer}, k_sparsity: {k_sparsity}, Device: {device}")

    model = LanguageModel(
        model_name,
        device_map=device,
        automodel=AutoModelForMaskedLM
    )

    submodule = model.esm.encoder.layer[layer].output
    dictionary_size = sae_expansion_factor * activation_dim 

    sequences = pl.read_parquet("10M_sequences.parquet")
    data = sequence_generator(sequences)

    buffer = ActivationBuffer(
        data=data,
        model=model,
        submodule=submodule,
        d_submodule=activation_dim,
        n_ctxs=int(1e2),
        device=device,
        refresh_batch_size=llm_batch_size,
        out_batch_size=sae_batch_size,
    )

    trainer_cfg = {
        "trainer": TopKTrainer,
        "dict_class": AutoEncoderTopK,
        "activation_dim": activation_dim,
        "dict_size": dictionary_size,
        "lr": lr,
        "device": device,
        "steps": training_steps,
        "layer": layer,
        "lm_name": model_name,
        "warmup_steps": warmup_steps,
        "k": k_sparsity,
        "wandb_name": f"plantrnafm-layer-{layer}-k-{k_sparsity}"
    }

    save_dir = os.path.join(base_save_dir, f"dictionary_{dictionary_size}" , f"layer_{layer}", f"k_{k_sparsity}")
    os.makedirs(save_dir, exist_ok=True)

    ae = trainSAE(
        data=buffer,
        trainer_configs=[trainer_cfg],
        steps=training_steps,
        use_wandb=True,
        wandb_project=wandb_project,
        log_steps=log_steps,
        save_dir=save_dir,
        autocast_type= torch.bfloat16
    )

    print(f"Finished training for Layer: {layer}, k_sparsity: {k_sparsity} with dictionary {dictionary_size}. SAE saved to {save_dir}")
    # The trainSAE function returns the trained autoencoder, but it also saves it.
    # We can return it here if needed for further processing, or just rely on the saving.
    return ae


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
        base_save_dir=args.base_save_dir
    )

if __name__ == '__main__':
    main()