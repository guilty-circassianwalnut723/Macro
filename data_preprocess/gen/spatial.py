#!/usr/bin/env python3
"""
Spatial data generation unified script

Features:
1. Unified control of generation config and count for outdoor, indoor, object subtypes
2. Read train/eval data from split/spatial directory
3. Save to final/spatial/{train/eval}/{image_count_category}/data and json directories
4. Support unique IDs to avoid duplicate generation
"""

import json
import sys
from pathlib import Path

# Add utils path
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

# ====== Configuration parameters ======
SPLIT_DIR = (Path(__file__).resolve().parent.parent.parent / "data" / "split" / "spatial")
FINAL_DIR = (Path(__file__).resolve().parent.parent.parent / "data" / "final" / "spatial")

# Subtype selection: outdoor, indoor, or object
# Set to None or [] to process all subtypes, or a specific subtype string to process only that subtype
SUB_TYPE = None  # Modify this value to select the subtype to process; None processes all subtypes

# Generation config: {sub_type: {image_count_category: {train: count, eval: count}}}
GEN_CONFIG = {
    "object": {
        "1-3": {"train": 12000, "eval": 90},
        "4-5": {"train": 12000, "eval": 90},
        "6-7": {"train": 12000, "eval": 90},
        ">=8": {"train": 12000, "eval": 90},
    },
    "outdoor": {
        "1-3": {"train": 9000, "eval": 80},
        "4-5": {"train": 9000, "eval": 80},
        "6-7": {"train": 9000, "eval": 80},
        ">=8": {"train": 9000, "eval": 80},
    },
    "indoor": {
        "1-3": {"train": 9000, "eval": 80},
        "4-5": {"train": 9000, "eval": 80},
        "6-7": {"train": 9000, "eval": 80},
        ">=8": {"train": 9000, "eval": 80},
    },
}

# Sampling parameter config: {sub_type: {param_name: value}}
# See script parameters in runner/spatial/gen directory
SAMPLING_CONFIG = {
    "outdoor": {
        "min_overlap": 0.3,      # Minimum overlap ratio
        "max_overlap": 0.8,      # Maximum overlap ratio
        "min_fov": 90.0,         # Minimum field of view (degrees)
        "max_fov": 90.0,         # Maximum field of view (degrees)
        "image_size": [1024, 1024],  # Output image size [height, width]
        "add_noise": False,      # Whether to add viewpoint perturbation
        "noise_scale": 10.0,     # Viewpoint perturbation magnitude (degrees)
        "base_pitch_range": [-10, 10],  # Random range for base_pitch (degrees), using random.uniform(base_pitch_range[0], base_pitch_range[1])
    },
    "indoor": {
        "min_overlap": 0.3,      # Minimum overlap ratio
        "max_overlap": 0.8,      # Maximum overlap ratio
        "min_fov": 90.0,         # Minimum field of view (degrees)
        "max_fov": 90.0,         # Maximum field of view (degrees)
        "image_size": [1024, 1024],  # Output image size [height, width]
        "add_noise": False,      # Whether to add viewpoint perturbation
        "noise_scale": 10.0,     # Viewpoint perturbation magnitude (degrees)
        "base_pitch_range": [-10, 10],  # Random range for base_pitch (degrees), using random.uniform(base_pitch_range[0], base_pitch_range[1])
    },
    "object": {
        "front_frame_range": list(range(24)),  # List of valid front_frame values, all selectable front frame values (0-23)
        "view_constraint_mode": 1,  # Viewpoint constraint mode: 1=current constraint (if output_view is front/back/left/right etc., must include adjacent views), 2=must include one of front/back/left/right etc., 3=no constraint
    },
}
# ======================


def main():
    """Main function"""
    # Determine the list of subtypes to process
    if SUB_TYPE is None:
        # Process all subtypes
        sub_types_to_process = list(GEN_CONFIG.keys())
        print("=" * 80)
        print("Spatial data generation script - processing all subtypes")
        print("=" * 80)
    else:
        # Process only the specified subtype
        if SUB_TYPE not in GEN_CONFIG:
            raise ValueError(f"Unsupported subtype: {SUB_TYPE}, supported types: {list(GEN_CONFIG.keys())}")
        sub_types_to_process = [SUB_TYPE]
        print("=" * 80)
        print(f"Spatial {SUB_TYPE.upper()} data generation script")
        print("=" * 80)
    
    print(f"Split directory: {SPLIT_DIR}")
    print(f"Final directory: {FINAL_DIR}")
    print(f"Subtypes to process: {sub_types_to_process}")
    print("=" * 80)
    
    # Create final directory
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Iterate over and process each subtype
    for current_sub_type in sub_types_to_process:
        print(f"\n{'=' * 80}")
        print(f"Starting to process subtype: {current_sub_type.upper()}")
        print(f"{'=' * 80}")
        
        gen_config = GEN_CONFIG[current_sub_type]
        sampling_config = SAMPLING_CONFIG.get(current_sub_type, {})
        
        print(f"Generation config: {gen_config}")
        print(f"Sampling config: {sampling_config}")
        
        # Import the corresponding processing module
        # Note: spatial.py is in the gen directory; outdoor/indoor/object.py are in gen/spatial
        if current_sub_type == "outdoor":
            from spatial.outdoor import process_split_data
            from utils.common import load_generated_ids
        elif current_sub_type == "indoor":
            from spatial.indoor import process_split_data
            from utils.common import load_generated_ids
        elif current_sub_type == "object":
            from spatial.object import process_split_data
            from utils.common import load_generated_ids
        else:
            raise ValueError(f"Unsupported subtype: {current_sub_type}")
        
        # Process train and eval data
        for split_type in ["train", "eval"]:
            print(f"\nProcessing {current_sub_type}/{split_type} data...")
            
            for image_count_category, config in gen_config.items():
                target_count = config.get(split_type, 0)
                
                if target_count <= 0:
                    print(f"Skipping {current_sub_type}/{split_type}/{image_count_category} data generation (target count is 0)")
                    continue
                
                # Load already-generated unique IDs (for spatial, sub_type must be passed)
                generated_ids = load_generated_ids(FINAL_DIR, split_type, image_count_category, sub_type=current_sub_type)
                print(f"Loaded {len(generated_ids)} already-generated sample IDs")
                
                # Process data
                process_split_data(
                    split_dir=SPLIT_DIR,
                    final_dir=FINAL_DIR,
                    split_type=split_type,
                    image_count_category=image_count_category,
                    target_count=target_count,
                    generated_ids=generated_ids,
                    sub_type=current_sub_type,
                    sampling_config=sampling_config
                )
        
        print(f"\nSubtype {current_sub_type.upper()} processing complete!")
    
    print(f"\n{'=' * 80}")
    print("All subtypes processing complete!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

