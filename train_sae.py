from nnsight import LanguageModel
from dictionary_learning import ActivationBuffer
from dictionary_learning.trainers.top_k import TopKTrainer, AutoEncoderTopK
from dictionary_learning.training import trainSAE
from transformers import AutoModelForMaskedLM, AutoTokenizer
import polars as pl
import torch
import os


def sequence_generator(data: pl.DataFrame):
    for row in data.iter_rows(named=True):
        yield row['sequence']


def train_sae_for_config(
    layer: int,
    k_sparsity: int,
    device: str,
    model_name: str = "yangheng/PlantRNA-FM",
    activation_dim: int = 480,
    llm_batch_size: int = 1024,
    sae_batch_size: int = 16_384,
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
        n_ctxs=int(3e4),
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
        "wandb_name": f"plantrnafm-lr{lr}-dict{dictionary_size}-layer-{layer}-k-{k_sparsity}",
        "decay_start": 40_000,
    }

    save_dir = os.path.join(base_save_dir, f"dictionary_{dictionary_size}" , f"layer_{layer}", f"k_{k_sparsity}", f"lr_{lr}")
    os.makedirs(save_dir, exist_ok=True)

    ae = trainSAE(
        data=buffer,
        trainer_configs=[trainer_cfg],
        steps=training_steps,
        use_wandb=True,
        wandb_project=wandb_project,
        log_steps=log_steps,
        save_dir=save_dir,
        autocast_dtype= torch.bfloat16,
        normalize_activations=True,
    )

    print(f"Finished training for Layer: {layer}, k_sparsity: {k_sparsity} with dictionary {dictionary_size}. SAE saved to {save_dir}")
    # The trainSAE function returns the trained autoencoder, but it also saves it.
    # We can return it here if needed for further processing, or just rely on the saving.
    return ae