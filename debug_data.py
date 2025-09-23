#!/usr/bin/env python3

import yaml
import os
from dataset import ProcessedLigandPocketDataset

def debug_data_dimensions():
    # Load config
    config_path = "configs/crossdock_fullatom_cond_playground.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Loading dataset...")
    
    # Build path to the train dataset
    datadir = config['datadir']
    train_npz_path = os.path.join(datadir, 'train.npz')
    
    print(f"Loading from: {train_npz_path}")
    
    dataset = ProcessedLigandPocketDataset(
        npz_path=train_npz_path,
        center=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get first few samples to debug
    for i in range(min(3, len(dataset))):
        print(f"\n--- Sample {i} ---")
        try:
            data = dataset[i]
            
            # Print the structure of the data
            print(f"Data keys: {data.keys()}")
            
            for key, value in data.items():
                if hasattr(value, 'shape'):
                    print(f"{key} shape: {value.shape}")
                elif hasattr(value, '__len__'):
                    print(f"{key} length: {len(value)}")
                else:
                    print(f"{key}: {value}")
                    
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    debug_data_dimensions()
