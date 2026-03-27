#!/usr/bin/env python3
"""
Spatial Object data generation script

Features:
1. Read train/eval data from split/spatial directory (object subtype)
2. Sample data from multi-viewpoint object images
3. Save to final/spatial/{train/eval}/{image_count_category}/data and json directories
4. Support unique IDs to avoid duplicate generation
"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from tqdm import tqdm
from PIL import Image

# Add utils path
CURRENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CURRENT_DIR))

from utils.common import (
    get_image_count_category,
    load_generated_ids,
    save_sample_data,
    generate_unique_id
)

# ====== Configuration parameters ======
SPLIT_DIR = (Path(__file__).resolve().parent.parent.parent.parent / "data" / "split" / "spatial")
FINAL_DIR = (Path(__file__).resolve().parent.parent.parent.parent / "data" / "final" / "spatial")

# Generation config: {image_count_category: {train: count, eval: count}}
GEN_CONFIG = {
    "1-3": {"train": 10000, "eval": 0},
    "4-5": {"train": 0, "eval": 0},
    "6-7": {"train": 0, "eval": 0},
    ">=8": {"train": 0, "eval": 0},
}

# Viewpoint direction definitions
VIEW_DIRECTIONS = ['front', 'back', 'left', 'right', 'top', 'bottom', 
                   'front_left', 'front_right', 'back_left', 'back_right']

# Viewpoint name mapping (for prompt generation)
VIEW_NAMES = {
    'front': 'front',
    'front_left': 'front-left',
    'left': 'left',
    'back_left': 'back-left',
    'back': 'back',
    'back_right': 'back-right',
    'right': 'right',
    'front_right': 'front-right',
    'top': 'top',
    'bottom': 'bottom',
}

# High-layer viewpoint offsets (relative to front, mod 24)
HIGH_LAYER_OFFSETS = {
    'front': 0,
    'front_right': 3,
    'right': 6,
    'back_right': 9,
    'back': 12,
    'back_left': 15,
    'left': 18,
    'front_left': 21,
}

# Random seed
RANDOM_SEED = 42
# ======================


def get_frame_indices_for_view(front_frame: int, view: str) -> List[int]:
    """
    Given a front frame and a viewpoint, return the list of possible frame indices (may have high-layer and low-layer options)
    
    Args:
        front_frame: front viewpoint frame number (0-23, high layer)
        view: viewpoint name
    
    Returns:
        list of possible frame indices
    """
    if view == 'top':
        return [25]
    elif view == 'bottom':
        return [26]
    
    if view not in HIGH_LAYER_OFFSETS:
        return []
    
    # High-layer frame calculation
    high_offset = HIGH_LAYER_OFFSETS[view]
    high_frame = (front_frame + high_offset) % 24
    
    # Check whether there is a corresponding low-layer frame
    # High-layer +2n corresponds to low-layer +n
    # Low-layer frame range 27-39 (13 frames total, 27 and 39 are the same)
    # Only front, right, back, left have corresponding low-layer frames (when high-layer frame number is even)
    
    candidates = [high_frame]
    
    total_high_offset = high_frame
    if total_high_offset % 2 == 0:  # There is a corresponding low-layer frame
        total_low_offset = total_high_offset // 2
        low_frame = 27 + (total_low_offset % 12)
        assert low_frame <= 39, f"Low-layer frame number out of range: {low_frame}"
        candidates.append(low_frame)
    
    return candidates


def sample_frame_for_view(front_frame: int, view: str) -> int:
    """
    Given a front frame and a viewpoint, randomly sample one frame
    
    Args:
        front_frame: front viewpoint frame number (0-23, high layer)
        view: viewpoint name
    
    Returns:
        frame index
    """
    candidates = get_frame_indices_for_view(front_frame, view)
    if not candidates:
        return front_frame  # Default: return front frame
    
    return random.choice(candidates)


def get_adjacent_views(view: str) -> List[str]:
    """
    Get the list of adjacent viewpoints for the specified viewpoint
    
    Args:
        view: viewpoint name
    
    Returns:
        list of adjacent viewpoints
    """
    adjacent_map = {
        'front': ['front_left', 'front_right'],
        'back': ['back_left', 'back_right'],
        'left': ['front_left', 'back_left'],
        'right': ['front_right', 'back_right'],
        'front_left': ['front', 'left'],
        'front_right': ['front', 'right'],
        'back_left': ['back', 'left'],
        'back_right': ['back', 'right'],
    }
    return adjacent_map.get(view, [])


def generate_prompt(input_views: List[str], output_view: str) -> str:
    """
    Generate a prompt describing input and output viewpoints
    """
    view_descriptions = []
    for i, view in enumerate(input_views, 1):
        view_name = VIEW_NAMES[view]
        view_descriptions.append(f"<image {i}> is the {view_name} view")
    
    output_name = VIEW_NAMES[output_view]
    prompt = f"For a fixed object, {', '.join(view_descriptions)}. From above, the sequence Front -> Left -> Back -> Right follows a clockwise order. Generate the {output_name} view."
    
    return prompt


def get_all_objects(source_dir: Path) -> List[Path]:
    """
    Get all object directories
    
    Args:
        source_dir: source directory
    
    Returns:
        list of object directories
    """
    objects = []
    if not source_dir.exists():
        return objects
    
    for category_dir in source_dir.iterdir():
        if category_dir.is_dir():
            for obj_dir in category_dir.iterdir():
                if obj_dir.is_dir() and (obj_dir / 'rgb').exists():
                    objects.append(obj_dir)
    
    return objects


def sample_one(
    obj_dir: Path,
    num_input_views: int,
    sampling_config: Optional[Dict] = None
) -> Optional[Dict]:
    """
    Sample one sample from an object directory
    
    Args:
        obj_dir: object directory path
        num_input_views: number of input viewpoints
    
    Returns:
        sample data or None
    """
    rgb_dir = obj_dir / 'rgb'
    if not rgb_dir.exists():
        return None
    
    # Check whether there are enough image files
    image_files = sorted(list(rgb_dir.glob('*.jpg')) + list(rgb_dir.glob('*.png')))
    if len(image_files) < 24:  # At least 24 high-layer frames required
        return None
    
    # Get configuration parameters
    if sampling_config is None:
        sampling_config = {}
    front_frame_range = sampling_config.get("front_frame_range", list(range(24)))
    view_constraint_mode = sampling_config.get("view_constraint_mode", 1)  # Default: use constraint mode 1
    
    # Randomly select front frame (front_frame_range should be a list of all valid front frame values, e.g. [0,1,...,23] or [0,6,12,18])
    if isinstance(front_frame_range, list) and len(front_frame_range) > 0:
        front_frame = random.choice(front_frame_range)
    else:
        # Default: use 0-23
        front_frame = random.randint(0, 23)
    
    # Available viewpoint directions
    available_views = [v for v in VIEW_DIRECTIONS if v != 'front']
    
    # Viewpoints that require constraints (front/back/left/right and their combinations)
    constrained_views = ['front', 'back', 'left', 'right', 
                         'front_left', 'front_right', 'back_left', 'back_right']
    
    # Randomly select output viewpoint
    if len(available_views) < 1:
        return None
    
    # Try multiple times to ensure constraints are satisfied
    max_sampling_attempts = 100
    input_views = None
    output_view = None
    
    for attempt in range(max_sampling_attempts):
        # Randomly select output viewpoint
        output_view = random.choice(available_views)
        
        # Choose sampling logic based on constraint mode
        if view_constraint_mode == 1:
            # Constraint mode 1: if output_view is front/back/left/right etc., must include adjacent views
            if output_view in constrained_views:
                adjacent_views = get_adjacent_views(output_view)
                # Find available adjacent viewpoints (excluding output_view itself)
                available_adjacent = [v for v in adjacent_views if v in available_views and v != output_view]
                
                # If no adjacent viewpoints available, skip this output_view
                if not available_adjacent:
                    continue
                
                # Remaining available viewpoints (excluding output_view)
                remaining_views = [v for v in available_views if v != output_view]
                
                # If not enough remaining viewpoints, skip
                if len(remaining_views) < num_input_views:
                    continue
                
                # Ensure at least one adjacent viewpoint is selected
                # Select at least one from adjacent viewpoints
                num_adjacent_to_include = random.randint(1, min(len(available_adjacent), num_input_views))
                selected_adjacent = random.sample(available_adjacent, num_adjacent_to_include)
                
                # Select remaining input viewpoints from other available viewpoints
                remaining_needed = num_input_views - num_adjacent_to_include
                if remaining_needed > 0:
                    other_views = [v for v in remaining_views if v not in selected_adjacent]
                    if len(other_views) < remaining_needed:
                        continue
                    selected_other = random.sample(other_views, remaining_needed)
                    input_views = selected_adjacent + selected_other
                else:
                    input_views = selected_adjacent
            else:
                # Output viewpoint needs no constraint, select normally
                remaining_views = [v for v in available_views if v != output_view]
                if len(remaining_views) < num_input_views:
                    continue
                input_views = random.sample(remaining_views, num_input_views)
        
        elif view_constraint_mode == 2:
            # Constraint mode 2: regardless of target viewpoint, must include one of front/back/left/right/front-left/front-right/back-left/back-right
            remaining_views = [v for v in available_views if v != output_view]
            
            # Check whether there are enough viewpoints
            if len(remaining_views) < num_input_views:
                continue
            
            # Find available constrained viewpoints (excluding output_view itself)
            available_constrained = [v for v in constrained_views if v in available_views and v != output_view]
            
            # If no constrained viewpoints available, skip
            if not available_constrained:
                continue
            
            # Ensure at least one constrained viewpoint is selected
            num_constrained_to_include = random.randint(1, min(len(available_constrained), num_input_views))
            selected_constrained = random.sample(available_constrained, num_constrained_to_include)
            
            # Select remaining input viewpoints from other available viewpoints
            remaining_needed = num_input_views - num_constrained_to_include
            if remaining_needed > 0:
                other_views = [v for v in remaining_views if v not in selected_constrained]
                if len(other_views) < remaining_needed:
                    continue
                selected_other = random.sample(other_views, remaining_needed)
                input_views = selected_constrained + selected_other
            else:
                input_views = selected_constrained
        
        elif view_constraint_mode == 3:
            # Constraint mode 3: no constraint, random selection
            remaining_views = [v for v in available_views if v != output_view]
            if len(remaining_views) < num_input_views:
                continue
            input_views = random.sample(remaining_views, num_input_views)
        
        else:
            # Unknown constraint mode, use default (no constraint)
            remaining_views = [v for v in available_views if v != output_view]
            if len(remaining_views) < num_input_views:
                continue
            input_views = random.sample(remaining_views, num_input_views)
        
        # If viewpoints were successfully selected, exit the loop
        if input_views is not None:
            break
    
    # If constraints cannot be satisfied after multiple attempts, return None
    if input_views is None or output_view is None:
        return None
    
    # Load input images
    input_images = []
    
    for view in input_views:
        frame_idx = sample_frame_for_view(front_frame, view)
        # Try to find the corresponding image file (naming format may differ)
        frame_file = None
        for img_file in image_files:
            # Try to extract frame number from filename
            try:
                frame_num = int(img_file.stem.split('_')[-1] or img_file.stem.split('.')[0])
                if frame_num == frame_idx:
                    frame_file = img_file
                    break
            except:
                continue
        
        if frame_file is None:
            # If not found, try using index
            if frame_idx < len(image_files):
                frame_file = image_files[frame_idx]
            else:
                continue
        
        try:
            img = Image.open(frame_file)
            input_images.append(img)
        except Exception as e:
            print(f"Failed to load image {frame_file}: {e}")
            continue
    
    if len(input_images) != num_input_views:
        return None
    
    # Load output image
    output_frame_idx = sample_frame_for_view(front_frame, output_view)
    output_frame_file = None
    for img_file in image_files:
        try:
            frame_num = int(img_file.stem.split('_')[-1] or img_file.stem.split('.')[0])
            if frame_num == output_frame_idx:
                output_frame_file = img_file
                break
        except:
            continue
    
    if output_frame_file is None:
        if output_frame_idx < len(image_files):
            output_frame_file = image_files[output_frame_idx]
        else:
            return None
    
    try:
        output_image = Image.open(output_frame_file)
    except Exception as e:
        print(f"Failed to load output image {output_frame_file}: {e}")
        return None
    
    return {
        'obj_dir': obj_dir,
        'front_frame': front_frame,
        'input_images': input_images,
        'input_views': input_views,  # Return original viewpoint keys (underscore format)
        'output_image': output_image,
        'output_view': output_view  # Return original viewpoint key (underscore format)
    }


def load_split_data(split_dir: Path, split_type: str, sub_type: str) -> List[Dict]:
    """
    Load data from split directory
    
    Args:
        split_dir: split directory
        split_type: "train" or "eval"
        sub_type: subtype ("object")
    
    Returns:
        list of samples
    """
    json_file = split_dir / f"{sub_type}_{split_type}.json"
    
    if not json_file.exists():
        # Compatibility with old format: try to read txt file
        txt_file = split_dir / f"{sub_type}_{split_type}.txt"
        if txt_file.exists():
            samples = []
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append({'obj_dir': Path(line)})
            return samples
        return []
    
    samples = []
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        sources = data.get('sources', [])
        for source in sources:
            samples.append({'obj_dir': Path(source)})
    
    return samples


def process_split_data(
    split_dir: Path,
    final_dir: Path,
    split_type: str,
    image_count_category: str,
    target_count: int,
    generated_ids: Set[str],
    sub_type: str = "object",
    sampling_config: Optional[Dict] = None
) -> None:
    """
    Process split data and generate final data
    
    Args:
        split_dir: split directory
        final_dir: final directory
        split_type: "train" or "eval"
        image_count_category: image count category
        target_count: target generation count
        generated_ids: set of already-generated unique IDs
        sub_type: subtype ("object")
    """
    # Load split data
    samples = load_split_data(split_dir, split_type, sub_type)
    
    if not samples:
        print(f"No data found for {split_type}/{sub_type}")
        return
    
    # Set random seed
    random.seed(RANDOM_SEED)
    
    # Process samples
    completed = len(generated_ids)  # Start count from already-generated count
    current_idx = len(generated_ids)
    max_attempts = target_count * 10  # Maximum number of attempts
    
    # Statistics
    total_attempts = 0
    successful_attempts = 0
    
    with tqdm(
        total=target_count, 
        desc=f"{split_type}/{image_count_category}",
        unit="sample",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    ) as pbar:
        pbar.update(len(generated_ids))
        
        attempt = 0
        while completed < target_count and attempt < max_attempts:
            attempt += 1
            total_attempts += 1
            
            # Randomly determine input viewpoint count per sample based on image_count_category
            if image_count_category == "1-3":
                num_input_views = random.randint(1, 3)
            elif image_count_category == "4-5":
                num_input_views = random.randint(4, 5)
            elif image_count_category == "6-7":
                num_input_views = random.randint(6, 7)
            else:  # >=8
                num_input_views = random.randint(8, 9)  # At most 9 input viewpoints
            
            # Randomly select an object
            sample = random.choice(samples)
            obj_dir = sample['obj_dir']
            
            # Sample one sample
            sampled = sample_one(obj_dir, num_input_views, sampling_config)
            if not sampled:
                continue
            
            # Generate unique ID
            # Build view_config string with viewpoint configuration info (using converted names)
            input_view_names = [VIEW_NAMES[v] for v in sampled['input_views']]
            output_view_name = VIEW_NAMES[sampled['output_view']]
            view_config = f"f{sampled['front_frame']:02d}_i{','.join(sorted(input_view_names))}_o{output_view_name}"
            # If SAVE_ORIGINAL_STRING=True, get the original string for saving
            from utils.common import SAVE_ORIGINAL_STRING
            unique_id_result = generate_unique_id(
                "spatial",
                return_original=SAVE_ORIGINAL_STRING,
                sub_type=sub_type,
                source=str(obj_dir),
                view_config=view_config
            )
            
            # Extract unique ID for checking (use MD5 hash if tuple, or use string directly)
            unique_id = unique_id_result[0] if isinstance(unique_id_result, tuple) else unique_id_result
            
            if unique_id in generated_ids:
                continue
            
            # Generate prompt (using original viewpoint keys)
            prompt = generate_prompt(sampled['input_views'], sampled['output_view'])
            
            # Prepare image files
            image_files = {}
            for i, img in enumerate(sampled['input_images']):
                image_files[f"image_{i+1}.jpg"] = img
            image_files["image_output.jpg"] = sampled['output_image']
            
            # Build expected image paths (save_sample_data will update to actual paths)
            if sub_type:
                data_dir = final_dir / split_type / sub_type / image_count_category / "data" / f"{current_idx:08d}"
            else:
                data_dir = final_dir / split_type / image_count_category / "data" / f"{current_idx:08d}"
            
            input_image_paths = [str(data_dir / f"image_{i+1}.jpg") for i in range(len(sampled['input_images']))]
            output_image_path = str(data_dir / "image_output.jpg")
            
            # Build JSON data (using converted names)
            json_data = {
                'obj_dir': str(obj_dir),
                'front_frame': sampled['front_frame'],
                'num_input_views': num_input_views,
                'input_views': input_view_names,
                'output_view': output_view_name,
                'prompt': prompt,
                'input_images': input_image_paths,
                'output_image': output_image_path
            }
            
            # Save data (for object, save under subtype directory)
            from utils.common import save_sample_data
            # For object, path is final/spatial/{train/eval}/object/{image_count_category}/...
            # Note: sub_type is already "object", so the path will automatically include it
            success = save_sample_data(
                final_dir,
                split_type,
                image_count_category,
                current_idx,
                unique_id_result,  # May be a string or (md5_hash, original_string) tuple
                json_data,
                image_files,
                sub_type=sub_type  # Pass subtype to support categorized saving
            )
            
            if success:
                generated_ids.add(unique_id)
                completed += 1
                successful_attempts += 1
                current_idx += 1
                pbar.update(1)
                
                # Update progress bar description with detailed statistics
                success_rate = (successful_attempts / total_attempts * 100) if total_attempts > 0 else 0
                pbar.set_description(
                    f"{split_type}/{image_count_category} | "
                    f"Done:{completed}/{target_count} | "
                    f"Attempts:{total_attempts} | "
                    f"SuccessRate:{success_rate:.1f}%"
                )
    
    print(f"\n{split_type}/{image_count_category} done: {completed}/{target_count}")


def main():
    """Main function"""
    print("=" * 80)
    print("Spatial Object data generation script")
    print("=" * 80)
    print(f"Split directory: {SPLIT_DIR}")
    print(f"Final directory: {FINAL_DIR}")
    print(f"Generation config: {GEN_CONFIG}")
    print("=" * 80)
    
    # Create final directory
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process train and eval data
    for split_type in ["train", "eval"]:
        print(f"\nProcessing {split_type} data...")
        
        for image_count_category, config in GEN_CONFIG.items():
            target_count = config.get(split_type, 0)
            
            if target_count <= 0:
                print(f"Skipping {split_type}/{image_count_category} data generation (target count is 0)")
                continue
            
            # Load already-generated unique IDs
            generated_ids = load_generated_ids(FINAL_DIR, split_type, image_count_category)
            print(f"Loaded {len(generated_ids)} already-generated sample IDs")
            
            # Process data
            process_split_data(
                split_dir=SPLIT_DIR,
                final_dir=FINAL_DIR,
                split_type=split_type,
                image_count_category=image_count_category,
                target_count=target_count,
                generated_ids=generated_ids,
                sub_type="object"
            )
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()

