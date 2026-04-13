#!/usr/bin/env python3
"""
Comprehensive Config Audit Script for BeatHeritage
Checks all config files against their corresponding dataclass definitions
"""

import os
import yaml
from pathlib import Path
from dataclasses import fields
from typing import Dict, List, Set, Any

# Import all config classes
from config import InferenceConfig, FidConfig, MaiModConfig
from osuT5.osuT5.config import TrainConfig, DataConfig, DataloaderConfig, OptimizerConfig
from osu_diffusion.config import DiffusionTrainConfig

def get_config_fields(config_class) -> Set[str]:
    """Get all field names from a dataclass"""
    return {field.name for field in fields(config_class)}

def get_yaml_keys(yaml_path: str, prefix: str = "") -> Set[str]:
    """Get all keys from a YAML file, including nested keys with dot notation"""
    keys = set()
    
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        def extract_keys(obj, parent_key=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == 'defaults':  # Skip Hydra defaults
                        continue
                    
                    full_key = f"{parent_key}.{key}" if parent_key else key
                    keys.add(full_key)
                    
                    if isinstance(value, dict):
                        extract_keys(value, full_key)
                    elif isinstance(value, list) and value and isinstance(value[0], dict):
                        # Handle list of dicts
                        extract_keys(value[0], full_key)
        
        extract_keys(data)
        
    except Exception as e:
        print(f"Error reading {yaml_path}: {e}")
    
    return keys

def audit_config_mapping(config_path: str, config_class, config_name: str):
    """Audit a specific config file against its dataclass"""
    print(f"\n[AUDIT] {config_name}: {config_path}")
    
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        return
    
    # Get fields from dataclass
    class_fields = get_config_fields(config_class)
    
    # Get keys from YAML
    yaml_keys = get_yaml_keys(config_path)
    
    # Find mismatches
    missing_in_class = yaml_keys - class_fields
    missing_in_config = class_fields - yaml_keys
    
    # Filter out nested keys for top-level check
    top_level_yaml = {key.split('.')[0] for key in yaml_keys}
    top_level_missing = top_level_yaml - class_fields
    
    print(f"[SUMMARY]:")
    print(f"   - Dataclass fields: {len(class_fields)}")
    print(f"   - YAML keys (all): {len(yaml_keys)}")
    print(f"   - YAML keys (top-level): {len(top_level_yaml)}")
    
    if top_level_missing:
        print(f"[MISSING] Keys in YAML but missing in dataclass:")
        for key in sorted(top_level_missing):
            related_keys = [k for k in yaml_keys if k.startswith(key)]
            print(f"   - {key} (related: {len(related_keys)} keys)")
            if len(related_keys) <= 5:  # Show details for small sections
                for rkey in sorted(related_keys)[:5]:
                    print(f"     * {rkey}")
            else:
                print(f"     * ... and {len(related_keys)-3} more keys")
    
    if missing_in_config:
        optional_missing = missing_in_config & {'hydra', 'train', 'diffusion'}  # Usually optional
        real_missing = missing_in_config - optional_missing
        if real_missing:
            print(f"[WARNING] Fields in dataclass but missing in YAML:")
            for key in sorted(real_missing):
                print(f"   - {key}")
    
    return {
        'missing_in_class': top_level_missing,
        'missing_in_config': missing_in_config,
        'all_yaml_keys': yaml_keys,
        'class_fields': class_fields
    }

def main():
    """Run comprehensive config audit"""
    print("BeatHeritage Config Audit - Finding ALL Mismatches")
    print("=" * 60)
    
    # Define config mappings
    config_mappings = [
        # Inference configs
        ("configs/inference/beatheritage_v1.yaml", InferenceConfig, "Inference (BeatHeritage V1)"),
        ("configs/inference/default.yaml", InferenceConfig, "Inference (Default)"),
        
        # Training configs  
        ("configs/train/beatheritage_v1.yaml", TrainConfig, "Training (BeatHeritage V1)"),
        ("configs/train/default.yaml", TrainConfig, "Training (Default)"),
        
        # Diffusion configs
        ("configs/diffusion/v1.yaml", DiffusionTrainConfig, "Diffusion (V1)"),
    ]
    
    all_issues = {}
    
    for config_path, config_class, name in config_mappings:
        issues = audit_config_mapping(config_path, config_class, name)
        if issues and issues['missing_in_class']:
            all_issues[name] = issues
    
    # Summary report
    print(f"\nAUDIT SUMMARY")
    print("=" * 60)
    
    if not all_issues:
        print("All configs are aligned with their dataclasses!")
        return
    
    print(f"Found issues in {len(all_issues)} config(s):")
    
    for config_name, issues in all_issues.items():
        print(f"\n{config_name}:")
        for key in sorted(issues['missing_in_class']):
            print(f"   - Missing field: {key}")
    
    # Generate fix suggestions
    print(f"\nSUGGESTED FIXES")
    print("=" * 60)
    
    for config_name, issues in all_issues.items():
        if 'Inference' in config_name:
            print(f"\nFor InferenceConfig class:")
            for key in sorted(issues['missing_in_class']):
                print(f"   + {key}: <appropriate_type> = <default_value>")

if __name__ == "__main__":
    main()
