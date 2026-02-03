from nnsight import LanguageModel
from dictionary_learning import ActivationBuffer
from dictionary_learning.trainers.top_k import TopKTrainer, AutoEncoderTopK
from dictionary_learning.training import trainSAE
from transformers import AutoModelForMaskedLM, AutoTokenizer
import polars as pl
import torch as t
import os
import gc

tracer_kwargs = {'scan' : False, 'validate' : False}

class EsmActivationBuffer(ActivationBuffer):
    def __init__(self, *args, use_nns_output=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_nns_output = use_nns_output
    
    def refresh(self):
        gc.collect()
        t.cuda.empty_cache()
        self.activations = self.activations[~self.read]

        current_idx = len(self.activations)
        new_activations = t.empty(self.activation_buffer_size, self.d_submodule, device=self.device, dtype=self.model.dtype)

        new_activations[: len(self.activations)] = self.activations
        self.activations = new_activations

        while current_idx < self.activation_buffer_size:
            with t.no_grad():
                tokens = self.tokenized_batch()
                with self.model.trace(
                    tokens,
                    **tracer_kwargs,
                    invoker_args={"truncation": True, "max_length": self.ctx_len},
                ):
                    if self.io == "in":
                        hidden_states = self.submodule.inputs[0].save()
                    elif self.use_nns_output:
                        hidden_states = self.submodule.nns_output.save()  
                    else:
                        hidden_states = self.submodule.output.save()
                    input = self.model.inputs.save()

                    # Also need to handle the stop() call appropriately
                    if self.use_nns_output:
                        self.submodule.nns_output.stop()
                    else:
                        self.submodule.output.stop()
            mask = (input.value[1]["attention_mask"] != 0)
            hidden_states = hidden_states.value
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
                
            if self.remove_bos:
                if self.model.tokenizer.bos_token_id is not None:
                    bos_mask = input.value[1]["input_ids"] == self.model.tokenizer.bos_token_id
                    mask = mask & ~bos_mask
                else:
                    # some models (like Qwen) don't have a bos token, so we need to remove the first non-pad token
                    assert mask.dim() == 2, "expected shape (batch_size, seq_len)"
                    first_one = (mask.to(t.int64).cumsum(dim=1) == 1) & mask
                    mask = mask & ~first_one

            if self.remove_high_norm is not None:
                # some models (like Qwen) have random high norm activation sinks which reduce training effectiveness
                norms_BL = hidden_states.norm(dim=-1)
                median_norm = norms_BL.median()
                norm_mask = norms_BL > median_norm * self.remove_high_norm
                mask = mask & ~norm_mask

            hidden_states = hidden_states[mask]

            remaining_space = self.activation_buffer_size - current_idx
            assert remaining_space > 0
            hidden_states = hidden_states[:remaining_space]

            self.activations[current_idx : current_idx + len(hidden_states)] = hidden_states.to(
                self.device
            )
            current_idx += len(hidden_states)

            # pbar.update(len(hidden_states))

        # pbar.close()
        self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)

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
    sae_batch_size: int = 4096,
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
    submodule = model.esm.encoder.layer[layer]
    dictionary_size = sae_expansion_factor * activation_dim 

    sequences = pl.read_parquet("all_transcripts_1024.parquet")
    data = sequence_generator(sequences)

    buffer = EsmActivationBuffer(
        data=data,
        model=model,
        submodule=submodule,
        d_submodule=activation_dim,
        use_nns_output=True,
        n_ctxs=int(3e2),
        ctx_len=1024,
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
        "wandb_name": f"fullrun-lr{lr}-dict{dictionary_size}-layer-{layer}-k-{k_sparsity}",
        "decay_start": warmup_steps+1,
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
        autocast_dtype= t.bfloat16,
        normalize_activations=True,
    )

    print(f"Finished training for Layer: {layer}, k_sparsity: {k_sparsity} with dictionary {dictionary_size}. SAE saved to {save_dir}")
    # The trainSAE function returns the trained autoencoder, but it also saves it.
    # We can return it here if needed for further processing, or just rely on the saving.
    return ae