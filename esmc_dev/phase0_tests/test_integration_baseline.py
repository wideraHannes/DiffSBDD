#!/usr/bin/env python3
"""
Test that baseline model still works without ESM-C (backward compatibility test).

This ensures our modifications don't break existing functionality.
"""

import sys
sys.path.insert(0, '/Users/hanneswidera/Uni/Master/thesis/DiffSBDD')

import torch
from dataset import ProcessedLigandPocketDataset

def test_dataset_without_esmc():
    """Test dataset loading without ESM-C (baseline mode)."""
    print("Testing dataset without ESM-C...")

    # Load without esmc_path (baseline)
    dataset = ProcessedLigandPocketDataset(
        npz_path='data/processed_crossdock_noH_full_temp/test.npz',
        center=True,
        esmc_path=None  # Baseline: no ESM-C
    )

    print(f"✓ Dataset loaded: {len(dataset)} samples")

    # Get a sample
    sample = dataset[0]
    print(f"✓ Sample keys: {list(sample.keys())}")

    # Verify pocket_emb is NOT in sample (baseline mode)
    assert 'pocket_emb' not in sample, "pocket_emb should not be present in baseline mode!"
    print("✓ Baseline mode: No pocket_emb (as expected)")

    # Test collate_fn
    batch = dataset.collate_fn([dataset[i] for i in range(2)])
    print(f"✓ Batch keys: {list(batch.keys())}")
    assert 'pocket_emb' not in batch
    print("✓ Baseline collate: No pocket_emb in batch")

    return True

def test_dataset_with_esmc():
    """Test dataset loading WITH ESM-C."""
    print("\nTesting dataset with ESM-C...")

    # Load with esmc_path
    dataset = ProcessedLigandPocketDataset(
        npz_path='data/processed_crossdock_noH_full_temp/test.npz',
        center=True,
        esmc_path='data/processed_crossdock_noH_full_temp/esmc_embeddings/test_esmc_embeddings.npz'
    )

    print(f"✓ Dataset loaded with ESM-C: {len(dataset)} samples")

    # Get a sample
    sample = dataset[0]
    print(f"✓ Sample keys: {list(sample.keys())}")

    # Verify pocket_emb IS in sample
    assert 'pocket_emb' in sample, "pocket_emb should be present with ESM-C!"
    print(f"✓ ESM-C mode: pocket_emb shape = {sample['pocket_emb'].shape}")
    assert sample['pocket_emb'].shape == torch.Size([960]), "pocket_emb should be 960-dim!"

    # Test collate_fn
    batch = dataset.collate_fn([dataset[i] for i in range(2)])
    print(f"✓ Batch keys: {list(batch.keys())}")
    assert 'pocket_emb' in batch
    print(f"✓ ESM-C collate: pocket_emb batch shape = {batch['pocket_emb'].shape}")
    assert batch['pocket_emb'].shape == torch.Size([2, 960])

    return True

if __name__ == "__main__":
    print("="*60)
    print("INTEGRATION TEST: Dataset Layer")
    print("="*60)

    try:
        # Test 1: Baseline mode (no ESM-C)
        test_dataset_without_esmc()

        # Test 2: ESM-C mode
        test_dataset_with_esmc()

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
