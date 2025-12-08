#!/usr/bin/env python3
"""
Example script demonstrating how to use the three baseline configurations for FiLM experiments.

Based on Part 2 of tuningplan.md:
- Baseline 1: Pretrained checkpoint only (no FiLM)
- Baseline 2: Identity-initialized FiLM (no-op verification)
- Baseline 3: Random-initialized FiLM (negative control)
"""

import torch
from lightning_modules import LigandPocketDDPM

# Path to your pretrained checkpoint
CHECKPOINT_PATH = "checkpoints/crossdocked_fullatom_cond.ckpt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_baseline_1():
    """
    Baseline 1: Pretrained checkpoint without FiLM

    Purpose: Establish ground truth for what "good" performance looks like
    Expected: Connectivity >95%, Validity >95%, QED 0.4-0.6

    This is the original pretrained model without any FiLM modifications.
    """
    print("\n" + "="*60)
    print("BASELINE 1: Pretrained without FiLM")
    print("="*60)

    model = LigandPocketDDPM.load_pretrained_with_esmc(
        CHECKPOINT_PATH,
        device=DEVICE,
        use_film=False,  # Disable FiLM completely
        film_mode="identity"  # Ignored since use_film=False
    )

    print(f"✓ Model loaded on {DEVICE}")
    print(f"✓ FiLM disabled: {not model.ddpm.dynamics.use_film}")

    return model


def load_baseline_2():
    """
    Baseline 2: Identity-initialized FiLM (no-op verification)

    Purpose: Verify that FiLM with γ=1, β=0 produces identical results to baseline 1
    Expected: Should match baseline 1 exactly (within numerical precision)

    FiLM transformation: h' = γ*h + β = 1*h + 0 = h (identity)
    """
    print("\n" + "="*60)
    print("BASELINE 2: Identity-initialized FiLM")
    print("="*60)

    model = LigandPocketDDPM.load_pretrained_with_esmc(
        CHECKPOINT_PATH,
        device=DEVICE,
        use_film=True,  # Enable FiLM
        film_mode="identity"  # Initialize to identity (γ=1, β=0)
    )

    print(f"✓ Model loaded on {DEVICE}")
    print(f"✓ FiLM enabled: {model.ddpm.dynamics.use_film}")

    # Verify identity initialization
    film = model.ddpm.dynamics.film_network
    final_layer = film[-1]
    joint_nf = final_layer.out_features // 2
    gamma = final_layer.bias.data[:joint_nf]
    beta = final_layer.bias.data[joint_nf:]

    print(f"✓ Gamma stats: mean={gamma.mean():.6f}, std={gamma.std():.6f}")
    print(f"✓ Beta stats:  mean={beta.mean():.6f}, std={beta.std():.6f}")

    # Check if identity
    gamma_is_one = torch.allclose(gamma, torch.ones_like(gamma), atol=1e-6)
    beta_is_zero = torch.allclose(beta, torch.zeros_like(beta), atol=1e-6)

    if gamma_is_one and beta_is_zero:
        print("✓ Identity initialization verified (γ≈1, β≈0)")
    else:
        print("⚠ Warning: Identity initialization may not be correct!")

    return model


def load_baseline_3():
    """
    Baseline 3: Random-initialized FiLM (negative control)

    Purpose: Verify that FiLM actually affects the model
    Expected: Much worse than baselines 1&2 (Connectivity 0-30%, Loss 1.0-2.0+)

    If random initialization performs similarly to identity, this indicates
    a bug where FiLM is not being used in the forward pass.
    """
    print("\n" + "="*60)
    print("BASELINE 3: Random-initialized FiLM")
    print("="*60)

    model = LigandPocketDDPM.load_pretrained_with_esmc(
        CHECKPOINT_PATH,
        device=DEVICE,
        use_film=True,  # Enable FiLM
        film_mode="random"  # Random initialization (Kaiming uniform)
    )

    print(f"✓ Model loaded on {DEVICE}")
    print(f"✓ FiLM enabled: {model.ddpm.dynamics.use_film}")

    # Verify random initialization (should NOT be identity)
    film = model.ddpm.dynamics.film_network
    final_layer = film[-1]
    joint_nf = final_layer.out_features // 2
    gamma = final_layer.bias.data[:joint_nf]
    beta = final_layer.bias.data[joint_nf:]

    print(f"✓ Gamma stats: mean={gamma.mean():.6f}, std={gamma.std():.6f}")
    print(f"✓ Beta stats:  mean={beta.mean():.6f}, std={beta.std():.6f}")

    # Check that it's NOT identity (randomness check)
    gamma_is_one = torch.allclose(gamma, torch.ones_like(gamma), atol=0.1)
    beta_is_zero = torch.allclose(beta, torch.zeros_like(beta), atol=0.1)

    if not gamma_is_one or not beta_is_zero:
        print("✓ Random initialization verified (not identity)")
    else:
        print("⚠ Warning: Random initialization looks like identity!")

    return model


def compare_film_parameters(model1, model2, name1, name2):
    """Compare FiLM parameters between two models"""
    print(f"\n{'='*60}")
    print(f"Comparing FiLM parameters: {name1} vs {name2}")
    print('='*60)

    def get_film_stats(model):
        film = model.ddpm.dynamics.film_network
        final_layer = film[-1]
        joint_nf = final_layer.out_features // 2
        gamma = final_layer.bias.data[:joint_nf]
        beta = final_layer.bias.data[joint_nf:]
        return gamma, beta

    gamma1, beta1 = get_film_stats(model1)
    gamma2, beta2 = get_film_stats(model2)

    gamma_diff = (gamma1 - gamma2).abs().mean()
    beta_diff = (beta1 - beta2).abs().mean()

    print(f"Mean absolute difference:")
    print(f"  Gamma: {gamma_diff:.6f}")
    print(f"  Beta:  {beta_diff:.6f}")

    if gamma_diff < 1e-6 and beta_diff < 1e-6:
        print("⚠ Models are nearly identical!")
    else:
        print("✓ Models have different FiLM parameters")


def main():
    """
    Demonstrate loading all three baseline configurations.

    After loading, you can use these models for:
    - Evaluation: model.test_step()
    - Generation: model.generate_ligands()
    - Training: trainer.fit(model)
    """
    print("\n" + "="*60)
    print("FiLM Baseline Configuration Examples")
    print("="*60)
    print("\nThis script demonstrates how to load three baseline configurations:")
    print("  1. Pretrained without FiLM (ground truth)")
    print("  2. Identity FiLM (no-op verification)")
    print("  3. Random FiLM (negative control)")

    # Load all three baselines
    baseline1 = load_baseline_1()
    baseline2 = load_baseline_2()
    baseline3 = load_baseline_3()

    # Compare them
    compare_film_parameters(baseline2, baseline3, "Identity", "Random")

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("\nTo use these models in your experiments:")
    print("")
    print("# Baseline 1: Pretrained only")
    print("model = LigandPocketDDPM.load_pretrained_with_esmc(")
    print('    "checkpoints/crossdocked_fullatom_cond.ckpt",')
    print("    use_film=False")
    print(")")
    print("")
    print("# Baseline 2: Identity FiLM")
    print("model = LigandPocketDDPM.load_pretrained_with_esmc(")
    print('    "checkpoints/crossdocked_fullatom_cond.ckpt",')
    print('    film_mode="identity"')
    print(")")
    print("")
    print("# Baseline 3: Random FiLM")
    print("model = LigandPocketDDPM.load_pretrained_with_esmc(")
    print('    "checkpoints/crossdocked_fullatom_cond.ckpt",')
    print('    film_mode="random"')
    print(")")
    print("")
    print("Expected Results:")
    print("  Baseline 1 ≈ Baseline 2 (both should have >95% connectivity)")
    print("  Baseline 3 << Baseline 1 (random should be much worse)")
    print("")


if __name__ == "__main__":
    main()
