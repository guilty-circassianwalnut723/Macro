#!/usr/bin/env python3
"""
Temporal data split script: sample from processed/temporal and split into train/eval data

Features:
1. Read all jsonl files from processed/temporal directory
2. Split frame sequences based on dino_threshold
3. Categorize as 1-3/4-5/6-7/>=8 based on frame count (note: actual category is segment frame count - 1)
4. Users only specify eval count; all others are automatically classified as train
5. Save to data_hl02/split/temporal directory, format is json (using list)
6. Record image num category statistics
7. If data volume is large, split into multiple sub-files
"""

import json
import random
import time
from pathlib import Path

MACRO_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = MACRO_DIR / "data"
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

# ====== Configuration parameters ======
DATA_DIR = DATA_DIR / "processed" / "temporal"
OUTPUT_DIR = DATA_DIR / "split" / "temporal"

# DINO threshold
DINO_THRESHOLD = 0.7

# Eval count config: {image_num_category: eval_count}
# Note: image_num_category is the actual image count (segment frame count - 1)
# Users only specify eval count; train count is automatically (total - eval count)
EVAL_COUNTS = {
    "1-3": 500,
    "4-5": 500,
    "6-7": 500,
    ">=8": 500,
}

# Maximum samples per json file (for splitting large files)
MAX_SAMPLES_PER_FILE = 10000

# Random seed
RANDOM_SEED = 42
# ======================


def load_jsonl(jsonl_path: Path) -> List[dict]:
    """Load jsonl file"""
    debug = False  # Can be set to True to enable detailed debug output
    samples = []
    if not jsonl_path.exists():
        if debug:
            print(f"[DEBUG] File not found: {jsonl_path}")
        return samples
    
    start_time = time.time()
    if debug:
        print(f"[DEBUG] Starting to read file: {jsonl_path}")
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            line_count = 0
            for line in f:
                line_count += 1
                if line_count % 10000 == 0 and debug:
                    elapsed = time.time() - start_time
                    print(f"[DEBUG] Read {line_count} lines, elapsed {elapsed:.2f}s: {jsonl_path.name}")
                
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if 'error' in data:
                        continue
                    samples.append(data)
                except json.JSONDecodeError as e:
                    if debug:
                        print(f"[DEBUG] Warning: cannot parse JSON line {line_count}: {e}")
                    continue
    except Exception as e:
        print(f"[ERROR] Failed to read file {jsonl_path}: {e}")
        return samples
    
    elapsed = time.time() - start_time
    if debug or elapsed > 5.0:  # Auto output debug info if reading takes > 5s
        print(f"[DEBUG] Read complete: {jsonl_path.name}, {len(samples)} samples, elapsed {elapsed:.2f}s")
    
    return samples


def split_segments_by_threshold(
    timestamps: List[int],
    dino_scores: List[Optional[float]],
    threshold: float,
) -> List[List[int]]:
    """Split a timestamp sequence into multiple segments based on adjacent DINO scores and threshold"""
    if not timestamps or len(timestamps) < 2:
        return []
    
    if len(dino_scores) != len(timestamps):
        if len(dino_scores) == len(timestamps) - 1:
            dino_scores = [None] + dino_scores
        else:
            print(f"Warning: timestamps length ({len(timestamps)}) does not match dino_scores length ({len(dino_scores)})")
            return []
    
    parts: List[List[int]] = []
    current: List[int] = [timestamps[0]]
    
    for i in range(1, len(timestamps)):
        prev_idx = i - 1
        cur_idx = i
        
        score = dino_scores[cur_idx]
        
        if score is not None and score < threshold:
            parts.append(current)
            current = [timestamps[cur_idx]]
        else:
            current.append(timestamps[cur_idx])
    
    if current:
        parts.append(current)
    
    return parts


def get_image_num_category(image_num: int) -> str:
    """Return category name based on image count"""
    if image_num <= 3:
        return "1-3"
    elif image_num <= 5:
        return "4-5"
    elif image_num <= 7:
        return "6-7"
    else:
        return ">=8"


def map_viewfs_path(viewfs_path: str, hdfs_mount: str = "/mnt/hdfs") -> str:
    """
    Map a viewfs:// path to a local mount path.
    
    If the source data contains viewfs:// paths, this function strips the
    scheme and cluster prefix so that files can be accessed via a local HDFS
    mount point.  Set the ``hdfs_mount`` argument (or the ``HDFS_MOUNT``
    environment variable) to point to your local HDFS mount.

    Args:
        viewfs_path: The original path, which may start with viewfs://.
        hdfs_mount: Local HDFS mount point (default: /mnt/hdfs).

    Returns:
        Local path string.
    """
    import os, re
    hdfs_mount = os.environ.get("HDFS_MOUNT", hdfs_mount).rstrip("/")
    if viewfs_path.startswith("viewfs://"):
        # Strip "viewfs://<cluster-name>" prefix
        local_path = re.sub(r"^viewfs://[^/]+", hdfs_mount, viewfs_path)
        return local_path
    return viewfs_path


def load_source_file(yt_file: str) -> Optional[Union[List, List[str]]]:
    """
    Load source_file (yt_file) at once, supports JSON array or line-by-line reading
    
    Args:
        yt_file: YouTube file path
    
    Returns:
        JSON array (List) or line list (List[str]), returns None on failure
    """
    debug = False  # Can be set to True to enable detailed debug output
    max_file_size_mb = 10  # Output warning if file exceeds 10MB
    
    if not yt_file:
        return None
    
    # Map viewfs path
    local_yt_file = map_viewfs_path(yt_file)
    yt_path = Path(local_yt_file)
    
    if not yt_path.exists():
        if debug:
            print(f"[DEBUG] yt_file not found: {local_yt_file}")
        return None
    
    # Check file size
    file_size_mb = 0
    try:
        file_size_mb = yt_path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_file_size_mb:
            print(f"[WARNING] Large file ({file_size_mb:.1f}MB), may read slowly: {yt_path.name}")
    except Exception:
        pass
    
    # Try to read as JSON file
    start_time = time.time()
    try:
        with open(yt_path, 'r', encoding='utf-8') as f:
            # Try to parse as JSON
            try:
                if debug:
                    print(f"[DEBUG] Attempting to parse JSON: {yt_path.name}")
                
                data = json.load(f)
                elapsed = time.time() - start_time
                if elapsed > 2.0:  # Output warning if parsing takes > 2s
                    size_info = f"Size: {file_size_mb:.1f}MB" if file_size_mb > 0 else "Size: unknown"
                    print(f"[WARNING] JSON parsing elapsed {elapsed:.2f}s: {yt_path.name} ({size_info})")
                
                if isinstance(data, list):
                    print(f"[DEBUG] Successfully loaded JSON array: {yt_path.name}, length: {len(data)}")
                    return data
                else:
                    # If not an array, fall back to line-by-line reading
                    raise ValueError("Not a JSON array")
            except (json.JSONDecodeError, ValueError) as e:
                # Not JSON format, read line by line
                if debug:
                    print(f"[DEBUG] Not JSON format, reading line by line: {yt_path.name}, error: {e}")
                
                f.seek(0)  # Reset file pointer
                read_start = time.time()
                lines = f.readlines()
                read_elapsed = time.time() - read_start
                
                if read_elapsed > 2.0:
                    print(f"[WARNING] Line-by-line read elapsed {read_elapsed:.2f}s: {yt_path.name} (lines: {len(lines)})")
                
                print(f"[DEBUG] Successfully loaded line by line: {yt_path.name}, lines: {len(lines)}")
                return lines
    except Exception as e:
        print(f"[ERROR] Failed to read yt_file {yt_path}: {e}")
        import traceback
        if debug:
            traceback.print_exc()
    
    return None


def get_video_path_from_loaded_data(loaded_data: Union[List, List[str]], yt_line: int) -> Optional[str]:
    """
    Get video path from already-loaded data (to avoid re-reading the file)
    
    Args:
        loaded_data: already-loaded data (JSON array or line list)
        yt_line: line number or JSON array index
    
    Returns:
        video path string or None
    """
    debug = False  # Can be set to True to enable detailed debug output
    
    if loaded_data is None or yt_line < 0:
        return None
    
    try:
        # Determine whether it is a JSON array or a line list
        # Line list from readlines(): all elements are strings, usually ending with newline
        # JSON array: elements may be dict, str, int, etc.; string values usually have no newline
        
        is_line_list = False
        if len(loaded_data) > 0:
            # If the first element is a dict, it must be a JSON array
            if isinstance(loaded_data[0], dict):
                is_line_list = False
            # If all elements are strings, check for newline (readlines characteristic)
            elif all(isinstance(item, str) for item in loaded_data):
                # Check whether any line ends with newline (readlines characteristic, except last line)
                # Or check whether any line contains a newline
                has_newline = any('\n' in item or item.endswith('\n') or item.endswith('\r\n') 
                                 for item in loaded_data[:min(10, len(loaded_data))])
                is_line_list = has_newline
            else:
                # Mixed types, should be JSON array
                is_line_list = False
        
        if is_line_list:
            # Line list (result of readlines, contains newline)
            if yt_line >= len(loaded_data):
                if debug:
                    print(f"[DEBUG] yt_line ({yt_line}) exceeds file line count ({len(loaded_data)})")
                return None
            
            video_path_str = loaded_data[yt_line].strip()
            
            if not video_path_str:
                return None
            
            video_path = Path(map_viewfs_path(video_path_str))
            
            if not video_path.exists():
                if debug:
                    print(f"[DEBUG] Video path not found: {video_path}")
                return None
            
            return str(video_path)
        else:
            # JSON array
            if yt_line >= len(loaded_data):
                if debug:
                    print(f"[DEBUG] yt_line ({yt_line}) exceeds array length ({len(loaded_data)})")
                return None
            
            item = loaded_data[yt_line]
            if isinstance(item, dict):
                # Try to get video path from multiple possible fields
                video_path_str = item.get('videoPath') or item.get('video_path') or item.get('path')
                if not video_path_str:
                    return None
            elif isinstance(item, str):
                video_path_str = item
            else:
                return None
            
            if not video_path_str:
                return None
            
            video_path = Path(map_viewfs_path(video_path_str))
            
            if not video_path.exists():
                if debug:
                    print(f"[DEBUG] Video path not found: {video_path}")
                return None
            
            return str(video_path)
    except Exception as e:
        if debug:
            print(f"[ERROR] Failed to get video path from loaded data: {e}")
        return None


def get_video_path(yt_file: str, yt_line: int) -> Optional[str]:
    """
    Get video path from yt_file and yt_line (compatible with old interface, re-reads file)
    
    Note: for performance, prefer using load_source_file + get_video_path_from_loaded_data
    
    Args:
        yt_file: YouTube file path
        yt_line: line number or JSON array index
    
    Returns:
        video path string or None
    """
    loaded_data = load_source_file(yt_file)
    if loaded_data is None:
        return None
    return get_video_path_from_loaded_data(loaded_data, yt_line)


def collect_all_segments(
    data_dir: Path,
    dino_threshold: float,
) -> Dict[str, List[dict]]:
    """Collect all split segments, grouped by category"""
    print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Starting segment collection, data dir: {data_dir}")
    
    segments_by_category = defaultdict(list)
    
    print(f"[DEBUG] Scanning directory: {data_dir}")
    try:
        video_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        print(f"[DEBUG] Found {len(video_dirs)} video directories")
    except Exception as e:
        print(f"[ERROR] Failed to scan directory: {e}")
        import traceback
        traceback.print_exc()
        return segments_by_category
    
    # Collect all jsonl files grouped by directory (each directory corresponds to the same source_file)
    dir_jsonl_files = defaultdict(list)  # {video_dir: [jsonl_file1, jsonl_file2, ...]}
    print(f"[DEBUG] Collecting all jsonl files and grouping by directory...")
    for idx, video_dir in enumerate(video_dirs):
        if idx % 100 == 0:
            print(f"[DEBUG] Scanned {idx}/{len(video_dirs)} directories...")
        try:
            jsonl_files = sorted(video_dir.glob("*.jsonl"))
            if jsonl_files:
                dir_jsonl_files[video_dir] = jsonl_files
        except Exception as e:
            print(f"[ERROR] Failed to scan directory {video_dir}: {e}")
            continue
    
    total_jsonl_files = sum(len(files) for files in dir_jsonl_files.values())
    total_segments = 0
    total_samples = 0
    
    print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Scan complete: {len(video_dirs)} video directories, {len(dir_jsonl_files)} valid directories, {total_jsonl_files} jsonl files")
    print(f"Processing {len(dir_jsonl_files)} video directories, {total_jsonl_files} jsonl files...")
    
    # Unified progress bar: process by directory
    file_start_time = time.time()
    last_log_time = time.time()
    processed_files = 0
    
    with tqdm(dir_jsonl_files.items(), desc="Collecting segments", unit="dir") as pbar:
        for dir_idx, (video_dir, jsonl_files) in enumerate(pbar):
            current_time = time.time()
            
            # Output detailed progress every 10 seconds
            if current_time - last_log_time >= 10.0:
                elapsed = current_time - file_start_time
                avg_time = elapsed / (dir_idx + 1) if dir_idx > 0 else 0
                remaining = avg_time * (len(dir_jsonl_files) - dir_idx - 1)
                print(f"\n[DEBUG] {datetime.now().strftime('%H:%M:%S')} Progress: {dir_idx + 1}/{len(dir_jsonl_files)} "
                      f"({dir_idx * 100 // len(dir_jsonl_files)}%), "
                      f"Elapsed: {elapsed:.1f}s, ETA: {remaining:.1f}s, "
                      f"CurrentDir: {video_dir.name[:30]}")
                last_log_time = current_time
            
            pbar.set_postfix({
                "Directory": video_dir.name[:20] + "..." if len(video_dir.name) > 20 else video_dir.name,
                "Files": len(jsonl_files),
                "Segments": total_segments
            })
            
            # Step 1: Determine source_file (yt_file) for this directory
            # Determined by reading the first sample from the first jsonl file
            source_file_yt_file = None
            source_file_loaded_data = None
            
            try:
                first_jsonl_file = jsonl_files[0]
                first_samples = load_jsonl(first_jsonl_file)
                if first_samples and len(first_samples) > 0:
                    source_file_yt_file = first_samples[0].get('yt_file', '')
                    if source_file_yt_file:
                        # Load source_file at once
                        source_file_load_start = time.time()
                        source_file_loaded_data = load_source_file(source_file_yt_file)
                        source_file_load_elapsed = time.time() - source_file_load_start
                        
                        if source_file_loaded_data is None:
                            print(f"\n[WARNING] Cannot load source_file: {source_file_yt_file} (dir: {video_dir.name})")
                        elif source_file_load_elapsed > 5.0:
                            data_size = len(source_file_loaded_data) if isinstance(source_file_loaded_data, list) else 0
                            print(f"\n[WARNING] source_file load elapsed {source_file_load_elapsed:.2f}s: {Path(source_file_yt_file).name} "
                                  f"(dir: {video_dir.name}, data size: {data_size})")
                        else:
                            data_size = len(source_file_loaded_data) if isinstance(source_file_loaded_data, list) else 0
                            print(f"\n[DEBUG] Loaded source_file: {Path(source_file_yt_file).name} (dir: {video_dir.name}, data size: {data_size})")
            except Exception as e:
                print(f"\n[ERROR] Failed to determine source_file (dir: {video_dir.name}): {e}")
                import traceback
                traceback.print_exc()
                # If source_file cannot be determined, fall back to the original method
                source_file_yt_file = None
                source_file_loaded_data = None
            
            # Step 2: Process all jsonl files in this directory
            with tqdm(jsonl_files, desc=f"Files", unit="file", leave=False,
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as file_pbar:
                for jsonl_file in file_pbar:
                    processed_files += 1
                    file_pbar.set_description(f"File {jsonl_file.name[:30]}..." if len(jsonl_file.name) > 30 else f"File {jsonl_file.name}")
                    
                    try:
                        file_load_start = time.time()
                        samples = load_jsonl(jsonl_file)
                        file_load_elapsed = time.time() - file_load_start
                        
                        if file_load_elapsed > 5.0:  # Output warning if loading file takes > 5s
                            print(f"\n[WARNING] File load elapsed {file_load_elapsed:.2f}s: {jsonl_file.name} (samples: {len(samples)})")
                        
                        total_samples += len(samples)
                        
                        # Update file progress bar suffix info
                        file_pbar.set_postfix({
                            "Samples": len(samples),
                            "TotalSegments": total_segments,
                            "Processed": f"{processed_files}/{total_jsonl_files}"
                        })
                        
                        # Process each sample - add progress bar
                        process_start = time.time()
                        video_path_count = 0
                        video_path_total_time = 0
                        
                        # Add progress bar for sample processing of each jsonl_file
                        file_name_display = jsonl_file.name[:50] + "..." if len(jsonl_file.name) > 50 else jsonl_file.name
                        with tqdm(samples, desc=f"  Samples", unit="sample", leave=False, 
                                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as sample_pbar:
                            for sample_idx, sample in enumerate(sample_pbar):
                                timestamps = sample.get('timestamp', [])
                                dino_scores = sample.get('dino_score', [])
                                yt_file = sample.get('yt_file', '')
                                yt_line = sample.get('yt_line', -1)
                                
                                # Verify yt_file is consistent with source_file
                                if source_file_yt_file and yt_file != source_file_yt_file:
                                    print(f"\n[WARNING] yt_file mismatch: expected={source_file_yt_file}, actual={yt_file} "
                                          f"(file: {jsonl_file.name}, sample: {sample_idx})")
                                
                                if not timestamps or len(timestamps) < 2:
                                    continue
                                
                                parts = split_segments_by_threshold(timestamps, dino_scores, dino_threshold)
                                
                                for part_idx, part_timestamps in enumerate(parts):
                                    image_num = len(part_timestamps) - 1
                                    
                                    if image_num <= 0:
                                        continue
                                    
                                    category = get_image_num_category(image_num)
                                    
                                    # Get video path - use already-loaded data to avoid re-reading
                                    video_path_start = time.time()
                                    if source_file_loaded_data is not None:
                                        # Use already-loaded data (recommended, avoids re-reading)
                                        video_path = get_video_path_from_loaded_data(source_file_loaded_data, yt_line)
                                    else:
                                        # Fall back to original method (if source_file cannot be determined)
                                        video_path = get_video_path(yt_file, yt_line)
                                    
                                    video_path_elapsed = time.time() - video_path_start
                                    video_path_total_time += video_path_elapsed
                                    video_path_count += 1
                                    
                                    # Output warning if a single get_video_path call takes > 3s
                                    if video_path_elapsed > 3.0:
                                        print(f"\n[WARNING] get_video_path elapsed {video_path_elapsed:.2f}s: "
                                              f"yt_file={yt_file}, yt_line={yt_line}")
                                    
                                    segment_info = {
                                        'yt_file': yt_file,
                                        'yt_line': yt_line,
                                        'jsonl_file': str(jsonl_file),
                                        'part_idx': part_idx,
                                        'timestamps': part_timestamps,
                                        'image_num': image_num,
                                        'image_num_category': category,
                                        'video_path': video_path,  # Save video path
                                    }
                                    
                                    segments_by_category[category].append(segment_info)
                                    total_segments += 1
                                
                                # Update sample progress bar suffix to show current file segment count
                                sample_pbar.set_postfix({
                                    "Segments": total_segments,
                                    "Files": f"{processed_files}/{total_jsonl_files}"
                                })
                        
                        process_elapsed = time.time() - process_start
                        if process_elapsed > 10.0:  # Output detailed info if processing a single file takes > 10s
                            avg_video_path_time = video_path_total_time / video_path_count if video_path_count > 0 else 0
                            print(f"\n[DEBUG] process file elapsed {process_elapsed:.2f}s: {jsonl_file.name}, "
                                  f"samples: {len(samples)}, segments: {total_segments}, "
                                  f"get_video_path calls: {video_path_count}, avg elapsed: {avg_video_path_time:.3f}s")
                    
                    except Exception as e:
                        print(f"\n[ERROR] Failed to process file {jsonl_file}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
            
            # Update progress bar suffix to show segment count per category
            category_info = ", ".join([f"{k}={len(v)}" for k, v in sorted(segments_by_category.items())[:3]])
            if len(segments_by_category) > 3:
                category_info += "..."
            pbar.set_postfix({
                "TotalSegments": total_segments,
                "Categories": category_info
            })
    
    total_elapsed = time.time() - file_start_time
    print(f"\n[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Segment collection complete")
    print(f"Processing complete: total_samples={total_samples}, total_segments={total_segments}, elapsed={total_elapsed:.2f}s")
    
    return segments_by_category


def sample_segments(
    segments: List[dict],
    eval_count: int,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """Sample eval samples from segment list; the rest are train samples; sample from different videos as much as possible"""
    print(f"[DEBUG] {datetime.now().strftime('%H:%M:%S')} Starting sampling, total segments: {len(segments)}, eval count: {eval_count}")
    start_time = time.time()
    
    if eval_count is None or eval_count <= 0:
        print(f"[DEBUG] eval_count is invalid, returning all segments as train")
        return segments, []
    
    if len(segments) <= eval_count:
        print(f"Warning: available segment count ({len(segments)}) <= eval count ({eval_count}), all samples used for eval")
        return [], segments
    
    random.seed(seed)
    
    # Group by video
    print(f"[DEBUG] Grouping by video...")
    group_start = time.time()
    segments_by_video = defaultdict(list)
    for seg in segments:
        video_key = f"{seg['yt_file']}_{seg['yt_line']}"
        segments_by_video[video_key].append(seg)
    
    video_keys = list(segments_by_video.keys())
    group_elapsed = time.time() - group_start
    print(f"[DEBUG] Grouping complete: {len(video_keys)} videos, elapsed {group_elapsed:.2f}s")
    
    if group_elapsed > 5.0:
        print(f"[WARNING] Grouping took too long: {group_elapsed:.2f}s")
    
    print(f"[DEBUG] Shuffling video order...")
    shuffle_start = time.time()
    random.shuffle(video_keys)
    
    for video_key in video_keys:
        random.shuffle(segments_by_video[video_key])
    shuffle_elapsed = time.time() - shuffle_start
    print(f"[DEBUG] Shuffling complete, elapsed {shuffle_elapsed:.2f}s")
    
    selected_eval_segments = []
    selected_seg_keys = set()
    
    print(f"[DEBUG] Starting eval sample sampling...")
    sample_start = time.time()
    
    # Strategy 1: if there are enough videos, sample one from each
    if len(video_keys) >= eval_count:
        print(f"[DEBUG] Strategy 1: video count ({len(video_keys)}) >= eval count ({eval_count})")
        for idx, video_key in enumerate(video_keys[:eval_count]):
            if idx % 100 == 0:
                print(f"[DEBUG] Sampling progress: {idx}/{eval_count}")
            video_segments = segments_by_video[video_key]
            for seg in video_segments:
                seg_key = f"{seg['yt_file']}_{seg['yt_line']}_{seg['part_idx']}"
                if seg_key not in selected_seg_keys:
                    selected_eval_segments.append(seg)
                    selected_seg_keys.add(seg_key)
                    break
    else:
        # Strategy 2: not enough videos, first sample one from each
        print(f"[DEBUG] Strategy 2: video count ({len(video_keys)}) < eval count ({eval_count})")
        for video_key in video_keys:
            if len(selected_eval_segments) >= eval_count:
                break
            video_segments = segments_by_video[video_key]
            for seg in video_segments:
                seg_key = f"{seg['yt_file']}_{seg['yt_line']}_{seg['part_idx']}"
                if seg_key not in selected_seg_keys:
                    selected_eval_segments.append(seg)
                    selected_seg_keys.add(seg_key)
                    break
        
        # If still not enough, continue sampling from videos
        iteration = 0
        while len(selected_eval_segments) < eval_count:
            iteration += 1
            if iteration % 10 == 0:
                print(f"[DEBUG] Sampling iteration {iteration}, sampled: {len(selected_eval_segments)}/{eval_count}")
            
            found_new = False
            for video_key in video_keys:
                if len(selected_eval_segments) >= eval_count:
                    break
                video_segments = segments_by_video[video_key]
                for seg in video_segments:
                    seg_key = f"{seg['yt_file']}_{seg['yt_line']}_{seg['part_idx']}"
                    if seg_key not in selected_seg_keys:
                        selected_eval_segments.append(seg)
                        selected_seg_keys.add(seg_key)
                        found_new = True
                        break
            if not found_new:
                print(f"[WARNING] Cannot find more new samples, stopping sampling")
                break
    
    sample_elapsed = time.time() - sample_start
    print(f"[DEBUG] Eval sampling complete: {len(selected_eval_segments)} samples, elapsed {sample_elapsed:.2f}s")
    
    # All remaining samples go to train
    print(f"[DEBUG] Building train sample list...")
    train_start = time.time()
    train_segments = []
    for idx, seg in enumerate(segments):
        if idx % 100000 == 0 and idx > 0:
            print(f"[DEBUG] Train sample processing progress: {idx}/{len(segments)}")
        seg_key = f"{seg['yt_file']}_{seg['yt_line']}_{seg['part_idx']}"
        if seg_key not in selected_seg_keys:
            train_segments.append(seg)
    
    train_elapsed = time.time() - train_start
    total_elapsed = time.time() - start_time
    print(f"[DEBUG] sampling complete: train={len(train_segments)}, eval={len(selected_eval_segments)}, "
          f"total elapsed={total_elapsed:.2f}s (train build: {train_elapsed:.2f}s)")
    
    return train_segments, selected_eval_segments


def save_json(samples: List[dict], output_path: Path):
    """Save samples to json file (using list format)"""
    start_time = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        elapsed = time.time() - start_time
        file_size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0
        
        if elapsed > 5.0 or file_size_mb > 50:
            print(f"[DEBUG] save file elapsed {elapsed:.2f}s: {output_path.name}, "
                  f"samples: {len(samples)}, file size: {file_size_mb:.1f}MB")
    except Exception as e:
        print(f"[ERROR] Failed to save file {output_path}: {e}")
        import traceback
        traceback.print_exc()
        raise


def save_segments_in_chunks(
    segments: List[dict],
    output_dir: Path,
    split_type: str,
    category: str,
    max_samples_per_file: int = 10000,
):
    """Save segments to json file (using list format); split into multiple files if data volume is large"""
    print(f"[DEBUG] {datetime.now().strftime('%H:%M:%S')} Starting to save {split_type} data, segments: {len(segments)}")
    start_time = time.time()
    
    if not segments:
        print(f"[DEBUG] No {split_type} data to save")
        return
    
    if len(segments) <= max_samples_per_file:
        # Single file, save directly, no progress bar
        output_file = output_dir / f"{split_type}.json"
        print(f"Saving {len(segments)} {split_type} samples to: {output_file.name}...", end="", flush=True)
        save_json(segments, output_file)
        elapsed = time.time() - start_time
        print(f" Done (elapsed {elapsed:.2f}s)")
    else:
        # Multiple files, use progress bar
        num_files = (len(segments) + max_samples_per_file - 1) // max_samples_per_file
        print(f"[DEBUG] Need to split into {num_files} files for saving")
        with tqdm(total=num_files, desc=f"Saving {split_type} data", unit="file") as pbar:
            for file_idx in range(num_files):
                file_start = time.time()
                start_idx = file_idx * max_samples_per_file
                end_idx = min(start_idx + max_samples_per_file, len(segments))
                chunk = segments[start_idx:end_idx]
                
                output_file = output_dir / f"{split_type}_{file_idx:04d}.json"
                save_json(chunk, output_file)
                file_elapsed = time.time() - file_start
                
                if file_elapsed > 10.0:
                    print(f"\n[WARNING] Save file elapsed {file_elapsed:.2f}s: {output_file.name}")
                
                pbar.update(1)
                pbar.set_postfix({
                    "File": f"{file_idx + 1}/{num_files}",
                    "Samples": len(chunk),
                    "Cumulative": end_idx,
                    "Total": len(segments)
                })
        
        total_elapsed = time.time() - start_time
        print(f"[DEBUG] {datetime.now().strftime('%H:%M:%S')} Saving {split_type} data complete, total elapsed: {total_elapsed:.2f}s")


def main():
    """Main function"""
    main_start_time = time.time()
    print("=" * 80)
    print("Temporal data split script")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"DINO threshold: {DINO_THRESHOLD}")
    print(f"Eval count config: {EVAL_COUNTS}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Max samples per file: {MAX_SAMPLES_PER_FILE}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    print(f"[DEBUG] Checking whether data directory exists: {DATA_DIR}")
    if not DATA_DIR.exists():
        print(f"[ERROR] Data directory not found: {DATA_DIR}")
        return
    print(f"[DEBUG] Data directory exists, continuing...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] Output directory created/confirmed: {OUTPUT_DIR}")
    
    # 1. Collect all segments
    print("=" * 80)
    print("Step 1: Collecting all segments")
    print("=" * 80)
    step1_start = time.time()
    segments_by_category = collect_all_segments(DATA_DIR, DINO_THRESHOLD)
    step1_elapsed = time.time() - step1_start
    print(f"\n[DEBUG] Step 1 complete, elapsed: {step1_elapsed:.2f}s ({step1_elapsed/60:.1f}min)")
    
    # Statistics
    print(f"\n[DEBUG] {datetime.now().strftime('%H:%M:%S')} Starting category statistics...")
    statistics = {}
    for category, segments in segments_by_category.items():
        statistics[category] = len(segments)
        print(f"Category {category}: {len(segments)} segments")
    print(f"[DEBUG] Statistics complete, {len(statistics)} categories total")
    
    # 2. Sample train/eval samples
    print("\n" + "=" * 80)
    print("Step 2: Sampling train/eval samples")
    print("=" * 80)
    step2_start = time.time()
    
    all_train_segments = []
    all_eval_segments = []
    category_stats = {}  # {category: {train: count, eval: count}}
    
    categories = list(segments_by_category.items())
    with tqdm(categories, desc="Sampling categories", unit="category") as pbar:
        for category, segments in pbar:
            eval_count = EVAL_COUNTS.get(category, 0)
            
            pbar.set_postfix({
                "Category": category,
                "Total": len(segments),
                "TargetEval": eval_count
            })
            
            print(f"\nProcessing category: {category}")
            print(f"  Available segments: {len(segments)}")
            print(f"  Need eval: {eval_count}")
            
            train_segments, eval_segments = sample_segments(
                segments, eval_count, RANDOM_SEED
            )
            
            print(f"  Actual sampled train: {len(train_segments)}, eval: {len(eval_segments)}")
            
            # Add image_num_category field to each sample
            for seg in train_segments:
                seg['image_num_category'] = category
            for seg in eval_segments:
                seg['image_num_category'] = category
            
            all_train_segments.extend(train_segments)
            all_eval_segments.extend(eval_segments)
            
            category_stats[category] = {
                'train': len(train_segments),
                'eval': len(eval_segments),
                'total': len(segments)
            }
            
            # Update progress bar to show cumulative statistics
            pbar.set_postfix({
                "Category": category,
                "TotalTrain": len(all_train_segments),
                "TotalEval": len(all_eval_segments)
            })
    
    step2_elapsed = time.time() - step2_start
    print(f"\n[DEBUG] Step 2 complete, elapsed: {step2_elapsed:.2f}s ({step2_elapsed/60:.1f}min)")
    
    # 3. Save data
    print("\n" + "=" * 80)
    print("Step 3: Saving data")
    print("=" * 80)
    step3_start = time.time()
    
    # Save train data (split by file)
    if all_train_segments:
        save_segments_in_chunks(
            all_train_segments,
            OUTPUT_DIR,
            "train",
            "all",
            MAX_SAMPLES_PER_FILE
        )
    
    # Save eval data (split by file)
    if all_eval_segments:
        save_segments_in_chunks(
            all_eval_segments,
            OUTPUT_DIR,
            "eval",
            "all",
            MAX_SAMPLES_PER_FILE
        )
    
    # 4. Save statistics (including image_num_category stats)
    stats_data = {
        'total_statistics': statistics,
        'category_statistics': category_stats,
        'summary': {
            'total_train': len(all_train_segments),
            'total_eval': len(all_eval_segments),
            'total_all': len(all_train_segments) + len(all_eval_segments),
        },
        'image_num_category': category_stats,  # Record image num category statistics
    }
    
    step3_elapsed = time.time() - step3_start
    print(f"\n[DEBUG] Step 3 complete, elapsed: {step3_elapsed:.2f}s ({step3_elapsed/60:.1f}min)")
    
    stats_file = OUTPUT_DIR / "statistics.json"
    print(f"[DEBUG] Saving statistics to: {stats_file}")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved statistics to: {stats_file}")
    
    total_elapsed = time.time() - main_start_time
    print("\n" + "=" * 80)
    print("Processing complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total elapsed: {total_elapsed:.2f}s ({total_elapsed/60:.1f}min, {total_elapsed/3600:.2f}h)")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nStatistics:")
    print(f"  Total Train samples: {len(all_train_segments)}")
    print(f"  Total Eval samples: {len(all_eval_segments)}")
    print(f"  Total samples: {len(all_train_segments) + len(all_eval_segments)}")
    print("\nStatistics by image_num_category:")
    for category, stats in sorted(category_stats.items()):
        print(f"  {category}: total={stats['total']}, train={stats['train']}, eval={stats['eval']}")
    print("\nTime per step:")
    print(f"  Step 1 (collect segments): {step1_elapsed:.2f}s ({step1_elapsed/60:.1f}min)")
    print(f"  Step 2 (sampling): {step2_elapsed:.2f}s ({step2_elapsed/60:.1f}min)")
    print(f"  Step 3 (saving): {step3_elapsed:.2f}s ({step3_elapsed/60:.1f}min)")
    print("=" * 80)


if __name__ == "__main__":
    main()
