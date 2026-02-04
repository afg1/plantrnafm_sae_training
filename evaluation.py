import gc
import os
import torch
import polars as pl
import numpy as np
from pathlib import Path
from tqdm import tqdm
from nnsight import LanguageModel
from transformers import AutoModelForMaskedLM
from dictionary_learning.trainers.top_k import AutoEncoderTopK
from typing import Optional
import json


def load_sae(save_dir: str, device: str) -> AutoEncoderTopK:
    """Load a trained SAE from a directory."""
    # dictionary_learning typically saves as ae.pt
    ae_path = os.path.join(save_dir, "ae.pt")
    if not os.path.exists(ae_path):
        # Try alternative naming
        for fname in os.listdir(save_dir):
            if fname.endswith(".pt"):
                ae_path = os.path.join(save_dir, fname)
                break
    
    ae = torch.load(ae_path, map_location=device)
    ae.eval()
    return ae


def get_activations_batch(
    model: LanguageModel,
    sequences: list[str],
    layer: int,
    device: str,
    ctx_len: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract activations for a batch of sequences.
    Returns activations and attention mask.
    """
    with torch.no_grad():
        tokens = model.tokenizer(
            sequences,
            return_tensors='pt',
            max_length=ctx_len,
            padding=True,
            truncation=True,
        )
        
        with model.trace(tokens, scan=False, validate=False):
            hidden_states = model.esm.encoder.layer[layer].nns_output.save()
        
        attention_mask = tokens['attention_mask'].to(device)
        activations = hidden_states.value  # [batch, seq_len, d_model]
        
    return activations, attention_mask


def compute_reconstruction_metrics(
    sae: AutoEncoderTopK,
    activations: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict:
    """
    Compute reconstruction metrics for a batch of activations.
    
    Returns dict with:
        - l2_loss: mean squared error
        - fve: fraction of variance explained
        - l0: average number of active features (should equal k for TopK)
    """
    with torch.no_grad():
        # Flatten to [num_valid_tokens, d_model]
        mask = attention_mask.bool()
        flat_activations = activations[mask]  # [n_tokens, d_model]
        
        # Get SAE reconstructions
        # AutoEncoderTopK forward returns (reconstructions, feature_activations, ...)
        output = sae(flat_activations)
        if isinstance(output, tuple):
            reconstructions = output[0]
            feature_acts = output[1]  # [n_tokens, dict_size] sparse
        else:
            reconstructions = output
            # Encode separately to get feature activations
            feature_acts = sae.encode(flat_activations)
        
        # L2 reconstruction loss
        l2_loss = ((flat_activations - reconstructions) ** 2).mean().item()
        
        # Fraction of variance explained
        total_variance = ((flat_activations - flat_activations.mean(dim=0)) ** 2).sum().item()
        residual_variance = ((flat_activations - reconstructions) ** 2).sum().item()
        fve = 1 - (residual_variance / total_variance) if total_variance > 0 else 0.0
        
        # L0 - number of active features per token
        l0 = (feature_acts != 0).float().sum(dim=-1).mean().item()
        
        # Feature activation statistics for dead feature detection
        feature_fired = (feature_acts != 0).any(dim=0)  # [dict_size] - did each feature fire at all?
        
    return {
        "l2_loss": l2_loss,
        "fve": fve,
        "l0": l0,
        "features_fired": feature_fired.cpu(),
    }


def evaluate_sae(
    sae: AutoEncoderTopK,
    model: LanguageModel,
    sequences: pl.DataFrame,
    layer: int,
    device: str,
    batch_size: int = 16,  # Reduced from 32
    max_sequences: Optional[int] = None,
) -> dict:
    """Memory-efficient SAE evaluation."""
    
    if max_sequences:
        sequences = sequences.head(max_sequences)
    print(sequences)
    all_l2 = []
    all_fve_num = []
    all_fve_denom = []
    total_tokens = 0
    
    # Track which features fired (on CPU, as numpy)
    n_features = sae.encoder.weight.shape[0]  # or however you access dict_size
    features_ever_fired = np.zeros(n_features, dtype=bool)
    
    seq_list = sequences['sequence'].to_list()
    
    for i in tqdm(range(0, len(seq_list), batch_size), desc="Evaluating"):
        batch_seqs = seq_list[i:i + batch_size]
        
        with torch.no_grad():
            activations, attention_mask = get_activations_batch(
                model, batch_seqs, layer, device
            )
            
            mask = attention_mask.bool()
            flat_acts = activations[mask]
            n_tokens = flat_acts.shape[0]
            
            # Get reconstructions
            output = sae(flat_acts)
            reconstructions = output[0]
            feature_acts = output[1]
            
            # Compute metrics immediately, store as Python floats
            l2 = ((flat_acts - reconstructions) ** 2).sum().item()
            fve_num = l2  # residual variance
            fve_denom = ((flat_acts - flat_acts.mean(dim=0)) ** 2).sum().item()
            
            # Track which features fired (move to CPU immediately)
            batch_fired = (feature_acts != 0).any(dim=0).cpu().numpy()
            features_ever_fired |= batch_fired
            
            all_l2.append(l2)
            all_fve_num.append(fve_num)
            all_fve_denom.append(fve_denom)
            total_tokens += n_tokens
            
            # Explicit cleanup
            del activations, attention_mask, flat_acts, reconstructions, feature_acts, output
        
        # Clear CUDA cache periodically
        if i % (batch_size * 10) == 0:
            torch.cuda.empty_cache()
    
    # Final metrics
    mean_l2 = sum(all_l2) / total_tokens
    fve = 1 - (sum(all_fve_num) / sum(all_fve_denom))
    n_dead = (~features_ever_fired).sum()
    
    return {
        "l2_loss": mean_l2,
        "fve": fve,
        "dead_feature_pct": (n_dead / n_features) * 100,
        "n_dead_features": int(n_dead),
        "n_features": n_features,
        "total_tokens": total_tokens,
    }


def discover_trained_saes(base_dir: str) -> list[dict]:
    """
    Discover all trained SAEs in the directory structure.
    Returns list of dicts with hyperparams and path.
    """
    saes = []
    base_path = Path(base_dir)
    
    # Structure: base_dir/dictionary_{size}/layer_{layer}/k_{k}/lr_{lr}/
    for dict_dir in base_path.glob("dictionary_*"):
        dict_size = int(dict_dir.name.split("_")[1])
        
        for layer_dir in dict_dir.glob("layer_*"):
            layer = int(layer_dir.name.split("_")[1])
            
            for k_dir in layer_dir.glob("k_*"):
                k = int(k_dir.name.split("_")[1])
                
                for lr_dir in k_dir.glob("lr_*"):
                    lr = float(lr_dir.name.split("_")[1])
                    trainer_dir = lr_dir / "trainer_0"
                    # Check if SAE file exists
                    if any(trainer_dir.glob("*.pt")):
                        saes.append({
                            "path": str(lr_dir),
                            "dict_size": dict_size,
                            "layer": layer,
                            "k": k,
                            "lr": lr,
                            "expansion": dict_size // 480,  # assuming d_model=480
                        })
    
    return saes


def main():
    device = "cuda:0"
    base_sae_dir = "plantrnafm_full_saes"
    eval_output_path = "sae_evaluation_results.parquet"
    
    # Load held-out sequences (tail of training data)
    print("Loading evaluation sequences...")
    all_sequences = pl.read_parquet("all_transcripts_1024.parquet")
    n_eval = 50_000  # Number of sequences for evaluation
    eval_sequences = all_sequences.tail(n_eval)
    print(f"Evaluating on {len(eval_sequences)} sequences")
    
    # Load model once
    print("Loading PlantRNA-FM model...")
    model = LanguageModel(
        "yangheng/PlantRNA-FM",
        device_map=device,
        automodel=AutoModelForMaskedLM,
    )
    
    # Discover trained SAEs
    print("Discovering trained SAEs...")
    sae_configs = discover_trained_saes(base_sae_dir)
    print(f"Found {len(sae_configs)} trained SAEs")
    
    # Evaluate each SAE
    results = []
    for config in tqdm(sae_configs, desc="Evaluating SAEs"):
        print(f"\nEvaluating: layer={config['layer']}, k={config['k']}, "
              f"lr={config['lr']}, expansion={config['expansion']}")
        
        try:
            sae = load_sae(config["path"], device)
            
            metrics = evaluate_sae(
                sae=sae,
                model=model,
                sequences=eval_sequences,
                layer=config["layer"],
                device=device,
                batch_size=32,
                max_sequences=5000,  # Use subset for faster eval, increase as needed
            )
            
            results.append({
                **config,
                **metrics,
            })
            
            print(f"  FVE: {metrics['fve']:.4f}, L2: {metrics['l2_loss']:.4f}, "
                  f"Dead: {metrics['dead_feature_pct']:.1f}%")
            del sae
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"  Failed: {e}")
            results.append({
                **config,
                "error": str(e),
            })
        
    
    # Save results
    results_df = pl.DataFrame(results)
    results_df.write_parquet(eval_output_path)
    print(f"\nResults saved to {eval_output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    if "fve" in results_df.columns:
        summary = (
            results_df
            .filter(pl.col("fve").is_not_null())
            .group_by(["k", "expansion", "lr"])
            .agg([
                pl.col("fve").mean().alias("mean_fve"),
                pl.col("dead_feature_pct").mean().alias("mean_dead_pct"),
                pl.col("l2_loss").mean().alias("mean_l2"),
            ])
            .sort("mean_fve", descending=True)
        )
        print("\nTop configurations by FVE (averaged across layers):")
        print(summary.head(10))
        
        # Best config per k
        print("\nBest configuration per sparsity level (k):")
        best_per_k = (
            results_df
            .filter(pl.col("fve").is_not_null())
            .sort("fve", descending=True)
            .group_by("k")
            .first()
            .select(["k", "expansion", "lr", "fve", "dead_feature_pct", "l2_loss"])
            .sort("k")
        )
        print(best_per_k)
        best_per_k.write_parquet("best_sae_configs_per_k.parquet")


if __name__ == "__main__":
    main()