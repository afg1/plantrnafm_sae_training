import os
import torch
import polars as pl
import numpy as np
from pathlib import Path
from tqdm import tqdm
from nnsight import LanguageModel
from transformers import AutoModelForMaskedLM
from dictionary_learning.dictionary import AutoEncoderTopK
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
    batch_size: int = 32,
    max_sequences: Optional[int] = None,
) -> dict:
    """
    Full evaluation of an SAE on held-out sequences.
    """
    if max_sequences:
        sequences = sequences.head(max_sequences)
    
    all_l2 = []
    all_fve_num = []  # residual variance
    all_fve_denom = []  # total variance
    all_l0 = []
    all_features_fired = []
    total_tokens = 0
    
    seq_list = sequences['sequence'].to_list()
    
    for i in tqdm(range(0, len(seq_list), batch_size), desc="Evaluating"):
        batch_seqs = seq_list[i:i + batch_size]
        
        activations, attention_mask = get_activations_batch(
            model, batch_seqs, layer, device
        )
        
        n_tokens = attention_mask.sum().item()
        metrics = compute_reconstruction_metrics(sae, activations, attention_mask)
        
        all_l2.append(metrics["l2_loss"] * n_tokens)
        all_l0.append(metrics["l0"] * n_tokens)
        all_features_fired.append(metrics["features_fired"])
        
        # For FVE, we need to accumulate properly
        mask = attention_mask.bool()
        flat_acts = activations[mask]
        with torch.no_grad():
            recon = sae(flat_acts)[0]
            all_fve_num.append(((flat_acts - recon) ** 2).sum().item())
            all_fve_denom.append(((flat_acts - flat_acts.mean(dim=0)) ** 2).sum().item())
        
        total_tokens += n_tokens
    
    # Aggregate metrics
    mean_l2 = sum(all_l2) / total_tokens
    mean_l0 = sum(all_l0) / total_tokens
    fve = 1 - (sum(all_fve_num) / sum(all_fve_denom))
    
    # Dead features: features that never fired across entire eval set
    all_fired = torch.stack(all_features_fired).any(dim=0)
    dead_feature_pct = (1 - all_fired.float().mean()).item() * 100
    n_dead_features = (~all_fired).sum().item()
    
    return {
        "l2_loss": mean_l2,
        "fve": fve,
        "l0": mean_l0,
        "dead_feature_pct": dead_feature_pct,
        "n_dead_features": n_dead_features,
        "n_features": all_fired.shape[0],
        "total_tokens_evaluated": total_tokens,
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
                    
                    # Check if SAE file exists
                    if any(lr_dir.glob("*.pt")):
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
                max_sequences=10_000,  # Use subset for faster eval, increase as needed
            )
            
            results.append({
                **config,
                **metrics,
            })
            
            print(f"  FVE: {metrics['fve']:.4f}, L2: {metrics['l2_loss']:.4f}, "
                  f"Dead: {metrics['dead_feature_pct']:.1f}%")
            
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