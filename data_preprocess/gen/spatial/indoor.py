#!/usr/bin/env python3
"""
Spatial Indoor data generation script

Features:
1. Read train/eval data from split/spatial directory (indoor subtype)
2. Sample multi-viewpoint data from indoor panoramic images
3. Use the equilib library for viewpoint transformation
4. Save to final/spatial/{train/eval}/{image_count_category}/data and json directories
5. Support unique IDs to avoid duplicate generation
"""

import json
import random
import math
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from tqdm import tqdm
from PIL import Image

try:
    from equilib import equi2pers
except ImportError:
    raise ImportError("The equilib library is required: pip install equilib")

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

# Viewpoint direction definitions (angle relative to front, in degrees)
VIEW_DIRECTIONS = {
    'front': 0,
    'front_left': 45,
    'left': 90,
    'back_left': 135,
    'back': 180,
    'back_right': 225,
    'right': 270,
    'front_right': 315,
    'top': None,
    'bottom': None,
}

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

# Random seed
RANDOM_SEED = 42
# ======================


def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians"""
    return degrees * math.pi / 180.0


def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees"""
    return radians * 180.0 / math.pi


def calculate_overlap_ratio_2d(
    view1_yaw: float,
    view1_pitch: float,
    view2_yaw: float,
    view2_pitch: float,
    fov1: float,
    fov2: float
) -> float:
    """Calculate the area overlap ratio between two viewpoints"""
    delta_yaw = abs(view1_yaw - view2_yaw)
    if delta_yaw > math.pi:
        delta_yaw = 2 * math.pi - delta_yaw
    delta_pitch = view1_pitch - view2_pitch
    
    sin_dp = math.sin(delta_pitch / 2.0)
    cos_p1 = math.cos(view1_pitch)
    cos_p2 = math.cos(view2_pitch)
    sin_dy = math.sin(delta_yaw / 2.0)
    
    a = sin_dp * sin_dp + cos_p1 * cos_p2 * sin_dy * sin_dy
    a = min(1.0, max(0.0, a))
    dist = 2.0 * math.asin(math.sqrt(a))
    
    r1 = degrees_to_radians(fov1) / 2.0
    r2 = degrees_to_radians(fov2) / 2.0
    
    if dist >= (r1 + r2):
        return 0.0
    if dist <= abs(r1 - r2):
        return 1.0
    
    try:
        angle1 = 2.0 * math.acos((r1*r1 + dist*dist - r2*r2) / (2.0 * r1 * dist))
        angle2 = 2.0 * math.acos((r2*r2 + dist*dist - r1*r1) / (2.0 * r2 * dist))
        
        inter_area1 = 0.5 * r1*r1 * (angle1 - math.sin(angle1))
        inter_area2 = 0.5 * r2*r2 * (angle2 - math.sin(angle2))
        
        intersection_area = inter_area1 + inter_area2
        min_r = min(r1, r2)
        min_area = math.pi * min_r * min_r
        
        return intersection_area / min_area
    except (ValueError, ZeroDivisionError):
        return 0.0


def get_view_angles(
    direction: str,
    base_yaw: float = 0.0,
    base_pitch: float = 0.0,
    noise: float = 0.0
) -> Dict[str, float]:
    """Get the viewpoint angles for the specified direction"""
    if direction == 'top':
        yaw = base_yaw + (random.uniform(-noise, noise) if noise > 0 else 0)
        return {
            'roll': 0.0,
            'pitch': -math.pi / 2 + base_pitch + (random.uniform(-noise, noise) if noise > 0 else 0),
            'yaw': yaw
        }
    elif direction == 'bottom':
        yaw = base_yaw + (random.uniform(-noise, noise) if noise > 0 else 0)
        return {
            'roll': 0.0,
            'pitch': math.pi / 2 + base_pitch + (random.uniform(-noise, noise) if noise > 0 else 0),
            'yaw': yaw
        }
    else:
        base_angle = degrees_to_radians(VIEW_DIRECTIONS[direction])
        yaw = base_yaw + base_angle + (random.uniform(-noise, noise) if noise > 0 else 0)
        yaw = yaw % (2 * math.pi)
        return {
            'roll': 0.0,
            'pitch': base_pitch + (random.uniform(-noise, noise) if noise > 0 else 0),
            'yaw': yaw
        }


def extract_perspective_view(
    equi_img: np.ndarray,
    rots: Dict[str, float],
    height: int = 512,
    width: int = 512,
    fov: float = 90.0
) -> np.ndarray:
    """Extract a perspective view from an equirectangular image"""
    if len(equi_img.shape) == 3:
        if equi_img.shape[2] == 3 or equi_img.shape[2] == 1:
            equi_img = np.transpose(equi_img, (2, 0, 1))
    
    pers_img = equi2pers(
        equi=equi_img,
        rots=rots,
        height=height,
        width=width,
        fov_x=fov,
        mode='bilinear',
    )
    
    if len(pers_img.shape) == 3:
        if pers_img.shape[0] == 3 or pers_img.shape[0] == 1:
            pers_img = np.transpose(pers_img, (1, 2, 0))
    
    return pers_img


def find_valid_view_combination(
    num_views: int,
    output_view: str,
    min_overlap: float,
    max_overlap: float,
    min_fov: float,
    max_fov: float,
    base_yaw: float,
    base_pitch: float,
    noise: float,
    max_attempts: int = 200
) -> Optional[Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, float]]]:
    """Find a valid viewpoint combination"""
    available_views = [v for v in VIEW_DIRECTIONS.keys() if v != output_view]
    num_input = num_views - 1
    
    if num_input > len(available_views):
        raise ValueError(f"Requested input view count ({num_input}) exceeds available view count ({len(available_views)})")
    
    for attempt in range(max_attempts):
        input_views = random.sample(available_views, num_input)
        
        view_fovs = {}
        all_views = input_views + [output_view]
        for view in all_views:
            view_fovs[view] = random.uniform(min_fov, max_fov)
        
        view_angles = {}
        for view in all_views:
            view_angles[view] = get_view_angles(view, base_yaw, base_pitch=base_pitch, noise=noise)
        
        output_angles = view_angles[output_view]
        output_yaw = output_angles['yaw']
        output_pitch = output_angles['pitch']
        output_fov = view_fovs[output_view]
        
        overlaps = []
        for input_view in input_views:
            input_angles = view_angles[input_view]
            input_yaw = input_angles['yaw']
            input_pitch = input_angles['pitch']
            input_fov = view_fovs[input_view]
            
            overlap = calculate_overlap_ratio_2d(
                input_yaw, input_pitch,
                output_yaw, output_pitch,
                input_fov, output_fov
            )
            overlaps.append(overlap)
        
        max_overlap_found = max(overlaps) if overlaps else 0.0
        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
        
        if min_overlap <= max_overlap_found <= max_overlap:
            return input_views, view_angles, view_fovs
        elif min_overlap <= avg_overlap <= max_overlap:
            return input_views, view_angles, view_fovs
    
    return None


def load_split_data(split_dir: Path, split_type: str, sub_type: str) -> List[Dict]:
    """
    Load data from split directory
    
    Args:
        split_dir: split directory
        split_type: "train" or "eval"
        sub_type: subtype ("indoor")
    
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
                        samples.append({'equi_path': Path(line)})
            return samples
        return []
    
    samples = []
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        sources = data.get('sources', [])
        for source in sources:
            samples.append({'equi_path': Path(source)})
    
    return samples


def sample_one(
    equi_path: Path,
    num_input_views: int,
    sampling_config: Optional[Dict] = None
) -> Optional[Dict]:
    """
    Sample one sample from a panoramic image
    
    Args:
        equi_path: panoramic image path
        num_input_views: number of input viewpoints
        sampling_config: sampling config dict
    
    Returns:
        sample data or None
    """
    if not equi_path.exists():
        return None
    
    # Load panoramic image
    try:
        equi_img = Image.open(equi_path)
        equi_img = np.asarray(equi_img)
        
        if len(equi_img.shape) == 2:
            equi_img = np.stack([equi_img] * 3, axis=-1)
        elif equi_img.shape[2] == 4:
            equi_img = equi_img[:, :, :3]
        
        # Resize image so the shorter side is 2048
        height, width = equi_img.shape[:2]
        min_dim = min(height, width)
        if min_dim > 2048:
            scale = 2048 / min_dim
            new_height = int(height * scale)
            new_width = int(width * scale)
            equi_img_pil = Image.fromarray(equi_img)
            equi_img_pil = equi_img_pil.resize((new_width, new_height), Image.LANCZOS)
            equi_img = np.asarray(equi_img_pil)
    except Exception as e:
        print(f"Failed to load panoramic image {equi_path}: {e}")
        return None
    
    # Get configuration parameters
    if sampling_config is None:
        sampling_config = {}
    base_pitch_range = sampling_config.get("base_pitch_range", [-10, 10])
    add_noise = sampling_config.get("add_noise", False)
    noise_scale = sampling_config.get("noise_scale", 10.0)
    min_fov = sampling_config.get("min_fov", 90.0)
    max_fov = sampling_config.get("max_fov", 90.0)
    min_overlap = sampling_config.get("min_overlap", 0.3)
    max_overlap = sampling_config.get("max_overlap", 0.8)
    image_size = tuple(sampling_config.get("image_size", [512, 512]))
    
    noise = degrees_to_radians(noise_scale) if add_noise else 0.0
    
    # Total viewpoint count (input viewpoints + 1 output viewpoint)
    num_views = num_input_views + 1
    
    # Try to find a valid viewpoint combination
    max_tries = 5
    result = None
    final_base_yaw = None
    final_base_pitch = None
    final_output_view = None
    final_view_fovs = None
    
    for try_idx in range(max_tries):
        # Re-generate base_yaw and base_pitch each iteration to increase the chance of finding a valid combination
        base_yaw = random.uniform(0, 2 * math.pi)
        base_pitch = degrees_to_radians(random.uniform(base_pitch_range[0], base_pitch_range[1]))
        
        available_output_views = list(VIEW_DIRECTIONS.keys())
        output_view = random.choice(available_output_views)
        
        result = find_valid_view_combination(
            num_views=num_views,
            output_view=output_view,
            min_overlap=min_overlap,
            max_overlap=max_overlap,
            min_fov=min_fov,
            max_fov=max_fov,
            base_yaw=base_yaw,
            base_pitch=base_pitch,
            noise=noise
        )
        
        if result is not None:
            final_base_yaw = base_yaw
            final_base_pitch = base_pitch
            final_output_view = output_view
            input_views, view_angles, view_fovs = result
            final_view_fovs = view_fovs
            break
    
    if result is None:
        return None
    
    # Extract viewpoint images
    input_images = []
    input_view_names = []
    height, width = image_size
    
    for view in input_views:
        rots = view_angles[view]
        view_fov = final_view_fovs[view]
        pers_img = extract_perspective_view(
            equi_img, rots, height=height, width=width, fov=view_fov
        )
        input_images.append(pers_img)
        input_view_names.append(VIEW_NAMES[view])
    
    # Extract output viewpoint
    output_rots = view_angles[final_output_view]
    output_fov = final_view_fovs[final_output_view]
    output_image = extract_perspective_view(
        equi_img, output_rots, height=height, width=width, fov=output_fov
    )
    output_view_name = VIEW_NAMES[final_output_view]
    
    return {
        'equi_path': equi_path,
        'input_images': input_images,
        'input_views': input_view_names,
        'output_image': output_image,
        'output_view': output_view_name,
        'base_yaw': radians_to_degrees(final_base_yaw),
        'base_pitch': radians_to_degrees(final_base_pitch)
    }


def generate_prompt(input_views: List[str], output_view: str) -> str:
    """Generate a prompt describing input and output viewpoints"""
    # Note: input_views and output_view are already the converted display names (e.g. 'back-left'),
    # not the original view keys (e.g. 'back_left'), so they can be used directly
    view_descriptions = []
    for i, view_name in enumerate(input_views, 1):
        view_descriptions.append(f"<image {i}> is the {view_name} view")
    
    prompt = f"From a fixed camera position, {', '.join(view_descriptions)}. From above, the sequence Front -> Left -> Back -> Right follows a counter-clockwise order. Generate the {output_view} view."
    
    return prompt


def process_split_data(
    split_dir: Path,
    final_dir: Path,
    split_type: str,
    image_count_category: str,
    target_count: int,
    generated_ids: Set[str],
    sub_type: str = "indoor",
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
        sub_type: subtype ("indoor")
        sampling_config: sampling config dict
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
                num_input_views = random.randint(8, 9)
            
            # Randomly select a panoramic image
            sample = random.choice(samples)
            equi_path = sample['equi_path']
            
            # Sample one sample
            sampled = sample_one(equi_path, num_input_views, sampling_config)
            if not sampled:
                continue
            
            # Generate unique ID
            # Build view_config string containing viewpoint configuration info
            view_config = f"y{sampled['base_yaw']:.4f}_p{sampled['base_pitch']:.4f}_i{','.join(sorted(sampled['input_views']))}_o{sampled['output_view']}"
            # If SAVE_ORIGINAL_STRING=True, get the original string for saving
            from utils.common import SAVE_ORIGINAL_STRING
            unique_id_result = generate_unique_id(
                "spatial",
                return_original=SAVE_ORIGINAL_STRING,
                sub_type=sub_type,
                source=str(equi_path),
                view_config=view_config
            )
            
            # Extract unique ID for checking (use MD5 hash if tuple, or use string directly)
            unique_id = unique_id_result[0] if isinstance(unique_id_result, tuple) else unique_id_result
            
            if unique_id in generated_ids:
                continue
            
            # Generate prompt
            prompt = generate_prompt(sampled['input_views'], sampled['output_view'])
            
            # Prepare image files
            image_files = {}
            for i, img in enumerate(sampled['input_images']):
                image_files[f"image_{i+1}.jpg"] = Image.fromarray(img)
            image_files["image_output.jpg"] = Image.fromarray(sampled['output_image'])
            
            # Build expected image paths (save_sample_data will update to actual paths)
            if sub_type:
                data_dir = final_dir / split_type / sub_type / image_count_category / "data" / f"{current_idx:08d}"
            else:
                data_dir = final_dir / split_type / image_count_category / "data" / f"{current_idx:08d}"
            
            input_image_paths = [str(data_dir / f"image_{i+1}.jpg") for i in range(len(sampled['input_images']))]
            output_image_path = str(data_dir / "image_output.jpg")
            
            # Build JSON data
            json_data = {
                'equi_path': str(equi_path),
                'num_input_views': num_input_views,
                'input_views': sampled['input_views'],
                'output_view': sampled['output_view'],
                'prompt': prompt,
                'base_yaw': sampled['base_yaw'],
                'base_pitch': sampled['base_pitch'],
                'input_images': input_image_paths,
                'output_image': output_image_path
            }
            
            # Save data (for indoor, sub_type must be passed)
            from utils.common import save_sample_data
            success = save_sample_data(
                final_dir,
                split_type,
                image_count_category,
                current_idx,
                unique_id_result,  # May be a string or (md5_hash, original_string) tuple
                json_data,
                image_files,
                sub_type=sub_type
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
    print("Spatial Indoor data generation script")
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
                sub_type="indoor"
            )
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()

