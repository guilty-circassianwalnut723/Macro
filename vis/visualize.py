"""
Visualization service - supports multi-baseline evaluation result visualization.
"""
import json
import math
from pathlib import Path
from typing import Dict, List, Optional
from flask import Flask, render_template, jsonify, request, send_file
from collections import defaultdict
import re

# Path configuration
SCRIPT_DIR = Path(__file__).parent
MACRO_DIR = SCRIPT_DIR.parent
OUTPUTS_DIR = MACRO_DIR / "outputs"

# Supported baseline models
SUPPORTED_BASELINES = ["bagel", "omnigen", "qwen"]

# Supported tasks and categories (multi-reference image tasks scored by LLM)
SUPPORTED_TASKS = ["customization", "spatial", "illustration", "temporal"]
IMAGE_NUM_CATEGORIES = ["1-3", "4-5", "6-7", ">=8"]

app = Flask(__name__, template_folder=str(SCRIPT_DIR / "templates" / "compare"))
app.static_folder = str(SCRIPT_DIR / "templates" / "compare" / "static")


def get_category_dir(baseline: str, exp_name: str, task: str, category: str = "") -> Path:
    """Get the output directory for the specified baseline/experiment/task/category.

    Directory structure: outputs/{baseline}/{exp_name}/{task}/{category}

    Supports api:{model} format (not a standalone baseline, used as an experiment option for comparison):
    - API experiment path: outputs/api/{model_name}/{task}/{category}
    """
    # api:{model_name} experiment: path is outputs/api/{model_name}/...
    if exp_name.startswith("api:"):
        model_name = exp_name[4:].strip()
        if not model_name:
            return OUTPUTS_DIR / "api" / "_" / task / category  # placeholder, does not actually exist
        base_dir = OUTPUTS_DIR / "api" / model_name / task / category
        return base_dir

    # Regular baseline experiment
    base_dir = OUTPUTS_DIR / baseline / exp_name / task / category
    return base_dir


def get_baselines() -> List[str]:
    """Get all baselines that have output."""
    baselines = []
    for baseline in SUPPORTED_BASELINES:
        baseline_dir = OUTPUTS_DIR / baseline
        if baseline_dir.exists() and any(baseline_dir.iterdir()):
            baselines.append(baseline)
    return baselines


def get_tasks_for_baseline(baseline: str) -> List[str]:
    """Get the list of tasks with output for the specified baseline."""
    tasks = set()
    baseline_dir = OUTPUTS_DIR / baseline
    if not baseline_dir.exists():
        return []

    for exp_dir in baseline_dir.iterdir():
        if exp_dir.is_dir():
            for task_dir in exp_dir.iterdir():
                if task_dir.is_dir() and task_dir.name in SUPPORTED_TASKS:
                    tasks.add(task_dir.name)

    return [task for task in SUPPORTED_TASKS if task in tasks]


def get_exps_for_baseline_task(baseline: str, task: str) -> List[str]:
    """Get the experiment list for the specified baseline and task.

    Includes: 1) experiments under the baseline; 2) API experiments api:{model_name} (from outputs/api/{model}/)
    """
    exps = set()

    # 1) Experiments under the baseline
    baseline_dir = OUTPUTS_DIR / baseline
    if baseline_dir.exists():
        for exp_dir in baseline_dir.iterdir():
            if exp_dir.is_dir():
                task_dir = exp_dir / task
                if task_dir.exists() and task_dir.is_dir():
                    for cat_dir in task_dir.iterdir():
                        if cat_dir.is_dir() and cat_dir.name in IMAGE_NUM_CATEGORIES:
                            exps.add(exp_dir.name)
                            break

    # 2) API experiments: outputs/api/{model_name}/, listed as api:{model_name}
    api_dir = OUTPUTS_DIR / "api"
    if api_dir.exists() and api_dir.is_dir():
        for model_dir in api_dir.iterdir():
            if model_dir.is_dir():
                task_dir = model_dir / task
                if task_dir.exists() and task_dir.is_dir():
                    for cat_dir in task_dir.iterdir():
                        if cat_dir.is_dir() and cat_dir.name in IMAGE_NUM_CATEGORIES:
                            exps.add(f"api:{model_dir.name}")
                            break

    return sorted(exps)


def get_exps_for_ours(baseline: str) -> List[str]:
    """Get the available experiment list for the Ours view (union of customization/illustration/spatial/temporal)."""
    exps = set()
    for task in ["customization", "illustration", "spatial", "temporal"]:
        exps.update(get_exps_for_baseline_task(baseline, task))
    return sorted(exps)


def get_exp_structure_for_ours(baseline: str) -> Dict:
    """Get the experiment structure for the Ours view."""
    exps = get_exps_for_ours(baseline)
    base_names = set()
    steps = set()
    exp_map = defaultdict(dict)

    for exp in exps:
        base_name, step = parse_exp_name(exp)
        base_names.add(base_name)
        if step:
            steps.add(step)
        exp_map[base_name][step] = exp

    sorted_steps = sorted(steps, key=lambda x: int(x) if x and x.isdigit() else 0)

    return {
        "base_names": sorted(base_names),
        "steps": sorted_steps,
        "exp_map": dict(exp_map)
    }


def parse_exp_name(exp_name: str) -> tuple:
    """Parse an experiment name and extract the base name and step.

    Supports two formats:
    1. With _step suffix: "all_no_t2i_0010000_step5000" -> ("all_no_t2i_0010000", "5000")
    2. Trailing number as step: "bagel_with_t2i_nano_max7_0010000" -> ("bagel_with_t2i_nano_max7", "0010000")
    3. No step: "bagel" -> ("bagel", None)

    For the trailing-number format, requirements:
    - Trailing digits (at least 4, typically training steps)
    - Preceded by an underscore or directly connected
    - Base name must not be purely numeric
    """
    # First try to match _step(\d+)$ format
    step_match = re.search(r'_step(\d+)$', exp_name)
    if step_match:
        base_name = exp_name[:step_match.start()]
        step = step_match.group(1)
        return (base_name, step)

    # Try to match trailing number format (at least 4 digits, typically training steps)
    # e.g.: bagel_with_t2i_nano_max7_0010000 -> ("bagel_with_t2i_nano_max7", "0010000")
    # Pattern: trailing at least 4 digits, preceded by underscore or directly connected
    end_num_match = re.search(r'[_-](\d{4,})$', exp_name)
    if end_num_match:
        base_name = exp_name[:end_num_match.start()]
        step = end_num_match.group(1)
        # Verify the base name is not purely numeric (to avoid false matches)
        if base_name and not base_name.replace('_', '').replace('-', '').isdigit():
            return (base_name, step)

    return (exp_name, None)


def get_exp_structure_for_baseline_task(baseline: str, task: str) -> Dict:
    """Get the experiment structure for the specified baseline and task.

    Returns structure: {
        "base_names": ["bagel", "all_no_t2i_0010000", ...],
        "steps": ["5000", "10000", ...],
        "exp_map": {
            "bagel": {None: "bagel"},
            "all_no_t2i_0010000": {"5000": "all_no_t2i_0010000_step5000", ...}
        }
    }
    """
    exps = get_exps_for_baseline_task(baseline, task)

    base_names = set()
    steps = set()
    exp_map = defaultdict(dict)

    for exp in exps:
        base_name, step = parse_exp_name(exp)
        base_names.add(base_name)
        if step:
            steps.add(step)
        exp_map[base_name][step] = exp

    # Sort steps in natural order (numeric sort)
    sorted_steps = sorted(steps, key=lambda x: int(x) if x and x.isdigit() else 0)

    return {
        "base_names": sorted(base_names),
        "steps": sorted_steps,
        "exp_map": dict(exp_map)
    }


def get_categories_for_baseline_task_exp(baseline: str, task: str, exp: str) -> List[str]:
    """Get the category list for the specified baseline/task/experiment.

    For api:{model} experiments, categories are fetched from outputs/api/{model}/{task}.
    """
    if exp.startswith("api:"):
        model_name = exp[4:].strip()
        task_dir = OUTPUTS_DIR / "api" / model_name / task
    else:
        task_dir = OUTPUTS_DIR / baseline / exp / task
    if not task_dir.exists():
        return []

    return [cat for cat in IMAGE_NUM_CATEGORIES if (task_dir / cat).is_dir()]


def load_single_sample(baseline: str, exp: str, task: str, category: str, sample_id: str) -> Optional[Dict]:
    """Load a single sample's data (optimized: only loads the specified sample_id).

    Args:
        baseline: Baseline name.
        exp: Experiment name.
        task: Task name.
        category: Category.
        sample_id: Sample ID (e.g. "00000001").

    Returns:
        Sample data dict, or None if not found.
    """
    category_dir = get_category_dir(baseline, exp, task, category)
    if not category_dir.exists():
        return None

    json_file = category_dir / f"{sample_id}.json"

    if json_file.exists():
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                info = json.load(f)
            info['sample_id'] = sample_id
            info['sample_dir'] = str(category_dir)
            if 'prompt' in info and 'instruction' not in info:
                info['instruction'] = info['prompt']
            return info
        except Exception:
            pass

    return None


def load_samples_for_baseline(baseline: str, exp: str, task: str, category: str) -> List[Dict]:
    """Load all samples for the specified baseline/experiment/task/category.

    Directory structure: {category_dir}/{idx:08d}.json
    """
    category_dir = get_category_dir(baseline, exp, task, category)
    if not category_dir.exists():
        return []

    samples = []
    json_files = sorted(category_dir.glob("[0-9]" * 8 + ".json"))
    for json_file in json_files:
        if json_file.name.startswith("scores"):
            continue
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                info = json.load(f)
            info['sample_id'] = json_file.stem
            info['sample_dir'] = str(category_dir)
            if 'prompt' in info and 'instruction' not in info:
                info['instruction'] = info['prompt']
            samples.append(info)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return samples


def load_score_for_baseline(baseline: str, exp: str, task: str, category: str, evaluator: str = 'gpt') -> Dict:
    """Load score data.

    Supports two formats (by priority):
    1. New format: {category_dir}/{sample_id}.score (one score file per sample, containing gpt_scores and gemini_scores)
    2. Old format: {category_dir}/scores.json or scores_gemini.json (unified score file)
    """
    category_dir = get_category_dir(baseline, exp, task, category)

    # Prefer loading .score files (new format)
    scores_data = {}
    score_files = list(category_dir.glob("*.score"))
    if score_files:
        for score_file in score_files:
            try:
                sample_id = score_file.stem
                with open(score_file, 'r', encoding='utf-8') as f:
                    score_content = json.load(f)

                score_data = score_content.get('gemini_scores' if evaluator == 'gemini' else 'gpt_scores', {})
                if not score_data:
                    continue

                if task == "customization":
                    scores_data[sample_id] = {
                        'consistency': score_data.get('consistency_scores', []),
                        'following_score': score_data.get('following_score'),
                        'reasoning': score_data.get('overall_reasoning', ''),
                        'consistency_scores_list': score_data.get('consistency_scores', [])
                    }
                elif task == "spatial":
                    scores_data[sample_id] = {
                        'viewpoint_transformation_score': score_data.get('viewpoint_transformation_score'),
                        'content_consistency_score': score_data.get('content_consistency_score'),
                        'reasoning': score_data.get('overall_reasoning', ''),
                        'consistency_scores_list': []
                    }
                elif task == "illustration":
                    scores_data[sample_id] = {
                        'text_consistency_score': score_data.get('text_consistency_score'),
                        'image_consistency_score': score_data.get('image_consistency_score'),
                        'reasoning': score_data.get('overall_reasoning', ''),
                        'consistency_scores_list': []
                    }
                elif task == "temporal":
                    scores_data[sample_id] = {
                        'context_consistency_score': score_data.get('context_consistency_score'),
                        'image_sequence_consistency_score': score_data.get('image_sequence_consistency_score'),
                        'reasoning': score_data.get('overall_reasoning', ''),
                        'consistency_scores_list': []
                    }
            except Exception as e:
                print(f"Error loading {score_file}: {e}")
                continue

        if scores_data:
            return scores_data

    # Fall back to old format scores.json
    score_filename = "scores_gemini.json" if evaluator == 'gemini' else "scores.json"
    score_file = category_dir / score_filename
    if not score_file.exists():
        return {}

    try:
        with open(score_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {score_file}: {e}")
        return {}


def harmonic_mean(scores: List[float]) -> float:
    """
    Compute the harmonic mean.

    Args:
        scores: List of scores.

    Returns:
        Harmonic mean; returns 0 if the list is empty or contains zero values.
    """
    if not scores:
        return 0
    # Filter out zeros and negatives to avoid division by zero
    positive_scores = [s if s > 0 else 1 for s in scores]
    if not positive_scores:
        return 0
    n = len(positive_scores)
    return n / sum(1.0 / s for s in positive_scores)


def get_metric_names(task: str) -> tuple:
    """Get the metric names for the specified task.

    The returned names match the actual field names in the score files:
    - customization: consistency_scores, following_score
    - spatial: viewpoint_transformation_score, content_consistency_score
    - illustration: text_consistency_score, image_consistency_score
    - temporal: context_consistency_score, image_sequence_consistency_score
    """
    metric_map = {
        "customization": ("consistency", "following_score"),
        "spatial": ("viewpoint_transformation_score", "content_consistency_score"),
        "illustration": ("text_consistency_score", "image_consistency_score"),
        "temporal": ("context_consistency_score", "image_sequence_consistency_score"),
    }
    return metric_map.get(task, ("metric1", "metric2"))


def calculate_statistics(scores: Dict, task: str) -> Dict:
    """Calculate statistics.

    For customization/illustration/spatial/temporal tasks, additionally compute average = mean(sqrt(m1*m2)).
    """
    metric1_name, metric2_name = get_metric_names(task)

    metric1_scores = []
    metric2_scores = []
    average_scores = []  # list of sqrt(m1*m2), used to compute average

    for sample_id, score_data in scores.items():
        if isinstance(score_data, dict):
            m1 = score_data.get(metric1_name)
            m2 = score_data.get(metric2_name)

            # For customization, consistency may be a list; compute harmonic mean
            if task == "customization" and metric1_name == "consistency":
                if isinstance(m1, list) and len(m1) > 0:
                    m1 = harmonic_mean(m1)

            if m1 is not None and isinstance(m1, (int, float)):
                metric1_scores.append(m1)
            if m2 is not None and isinstance(m2, (int, float)):
                metric2_scores.append(m2)
            # When both m1 and m2 are available, compute sqrt(m1*m2) for average
            if m1 is not None and m2 is not None and isinstance(m1, (int, float)) and isinstance(m2, (int, float)) and m1 >= 0 and m2 >= 0:
                average_scores.append(math.sqrt(m1 * m2))

    def calc_stats(values):
        if not values:
            return {"mean": "N/A", "median": "N/A", "count": 0}
        sorted_vals = sorted(values)
        mean_val = sum(values) / len(values)
        median_val = sorted_vals[len(sorted_vals) // 2]
        return {
            "mean": round(mean_val, 2),
            "median": round(median_val, 2),
            "count": len(values)
        }

    result = {
        metric1_name: calc_stats(metric1_scores),
        metric2_name: calc_stats(metric2_scores),
        "metric1_name": metric1_name,
        "metric2_name": metric2_name
    }
    # For customization/illustration/spatial/temporal, add average
    if task in ["customization", "illustration", "spatial", "temporal"]:
        result["average"] = calc_stats(average_scores)
    return result


def get_average_for_task_category(baseline: str, exp: str, task: str, category: str, evaluator: str = 'gpt') -> Optional[float]:
    """Compute the average (mean of sqrt(score_a*score_b)) for a single task+category.

    Returns the mean of sqrt(m1*m2) for all samples in that task+category, or None if no data.
    """
    scores = load_score_for_baseline(baseline, exp, task, category, evaluator)
    if not scores:
        return None

    metric1_name, metric2_name = get_metric_names(task)
    average_scores = []

    for sample_id, score_data in scores.items():
        if isinstance(score_data, dict):
            m1 = score_data.get(metric1_name)
            m2 = score_data.get(metric2_name)

            if task == "customization" and metric1_name == "consistency":
                if isinstance(m1, list) and len(m1) > 0:
                    m1 = harmonic_mean(m1)

            if m1 is not None and m2 is not None and isinstance(m1, (int, float)) and isinstance(m2, (int, float)) and m1 >= 0 and m2 >= 0:
                average_scores.append(math.sqrt(m1 * m2))

    if not average_scores:
        return None
    return round(sum(average_scores) / len(average_scores), 2)


def get_sample_count_for_task_category(baseline: str, exp: str, task: str, category: str, evaluator: str = 'gpt') -> int:
    """Get the number of evaluated samples for a single task+category."""
    scores = load_score_for_baseline(baseline, exp, task, category, evaluator)
    if not scores:
        return 0

    metric1_name, metric2_name = get_metric_names(task)
    count = 0

    for sample_id, score_data in scores.items():
        if isinstance(score_data, dict):
            m1 = score_data.get(metric1_name)
            m2 = score_data.get(metric2_name)
            if task == "customization" and metric1_name == "consistency":
                if isinstance(m1, list) and len(m1) > 0:
                    m1 = harmonic_mean(m1)
            if m1 is not None and m2 is not None and isinstance(m1, (int, float)) and isinstance(m2, (int, float)) and m1 >= 0 and m2 >= 0:
                count += 1
    return count


def get_ours_scores(baseline: str, exps: List[str], evaluator: str = 'gpt') -> Dict:
    """Get aggregated scores for the Ours view.

    For each experiment, aggregate scores across customization/illustration/spatial/temporal:
    - Each task+category produces an average (mean of sqrt(m1*m2))
    - Task mean = arithmetic mean of averages across all categories for that task
    - Overall mean = arithmetic mean of the four task means

    Return structure:
    {
        "exps": [exp1, exp2, ...],
        "task_rows": [
            {"task": "...", "task_mean": {}, "categories": {}, "categories_count": {}, "task_count": {}},
            ...
        ],
        "total_row": {}
    }
    """
    OURS_TASKS = ["customization", "illustration", "spatial", "temporal"]

    result = {
        "exps": exps,
        "task_rows": [],
        "total_row": {}
    }

    for exp in exps:
        result["total_row"][exp] = None

    for task in OURS_TASKS:
        task_data = {
            "task": task,
            "task_mean": {},
            "categories": {},
            "categories_count": {},  # {exp: {cat: count}}
            "task_count": {}  # {exp: total_count}
        }

        for exp in exps:
            task_data["task_mean"][exp] = None
            task_data["categories"][exp] = {}
            task_data["categories_count"][exp] = {}
            task_data["task_count"][exp] = 0

            categories = get_categories_for_baseline_task_exp(baseline, task, exp)
            if not categories:
                continue

            category_avgs = []
            task_total = 0
            for cat in categories:
                avg_val = get_average_for_task_category(baseline, exp, task, cat, evaluator)
                cnt = get_sample_count_for_task_category(baseline, exp, task, cat, evaluator)
                task_data["categories_count"][exp][cat] = cnt
                task_total += cnt
                if avg_val is not None:
                    task_data["categories"][exp][cat] = avg_val
                    category_avgs.append(avg_val)

            task_data["task_count"][exp] = task_total
            if category_avgs:
                task_data["task_mean"][exp] = round(sum(category_avgs) / len(category_avgs), 2)

        result["task_rows"].append(task_data)

    # Compute overall mean: for each exp, (cust_mean + illus_mean + spatial_mean + temporal_mean) / 4
    # total_count: sum of sample counts across all four tasks (for validation)
    result["total_count"] = {}
    for exp in exps:
        task_means = []
        total_cnt = 0
        for task_row in result["task_rows"]:
            tm = task_row["task_mean"].get(exp)
            if tm is not None:
                task_means.append(tm)
            total_cnt += task_row["task_count"].get(exp, 0)
        if task_means:
            result["total_row"][exp] = round(sum(task_means) / len(task_means), 2)
        result["total_count"][exp] = total_cnt

    return result


# ==================== API Routes ====================

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@app.route('/api/baselines')
def api_baselines():
    """Get all baseline list."""
    return jsonify(get_baselines())


@app.route('/api/tasks/<baseline>')
def api_tasks(baseline: str):
    """Get task list for the specified baseline."""
    return jsonify(get_tasks_for_baseline(baseline))


@app.route('/api/exp-structure/<baseline>/<task>')
def api_exp_structure(baseline: str, task: str):
    """Get experiment structure for the specified baseline and task."""
    return jsonify(get_exp_structure_for_baseline_task(baseline, task))


@app.route('/api/ours-exp-structure/<baseline>')
def api_ours_exp_structure(baseline: str):
    """Get experiment structure for the Ours view."""
    return jsonify(get_exp_structure_for_ours(baseline))


@app.route('/api/categories/<baseline>/<task>/<exp>')
def api_categories(baseline: str, task: str, exp: str):
    """Get category list for the specified baseline/task/experiment."""
    return jsonify(get_categories_for_baseline_task_exp(baseline, task, exp))


@app.route('/api/comparison')
def api_comparison():
    """Get multi-experiment comparison statistics."""
    baseline = request.args.get('baseline', '')
    task = request.args.get('task', '')
    category = request.args.get('category', '')
    exps = request.args.get('exps', '').split(',')
    evaluator = request.args.get('evaluator', 'gpt')

    if not all([baseline, task, category, exps]):
        return jsonify({"error": "Missing parameters"}), 400

    exps = [e.strip() for e in exps if e.strip()]
    if not exps:
        return jsonify({"error": "No experiments specified"}), 400

    # Get metric names
    metric1_name, metric2_name = get_metric_names(task)

    # Collect statistics for each experiment
    comparison_data = {}
    for exp in exps:
        scores = load_score_for_baseline(baseline, exp, task, category, evaluator)
        stats = calculate_statistics(scores, task)
        comparison_data[exp] = stats

    # Build comparison table
    comparison_table = []

    # For customization/illustration/spatial/temporal, add average row first (sqrt(score_a*score_b))
    if task in ["customization", "illustration", "spatial", "temporal"]:
        row_avg = {"metric_name": "average"}
        for exp in exps:
            stats = comparison_data.get(exp, {}).get("average", {})
            row_avg[exp] = stats if isinstance(stats, dict) else {"mean": "N/A", "median": "N/A"}
        comparison_table.append(row_avg)

    # Metric 1 statistics
    row1 = {"metric_name": metric1_name}
    for exp in exps:
        stats = comparison_data.get(exp, {}).get(metric1_name, {})
        row1[exp] = stats if isinstance(stats, dict) else {"mean": "N/A", "median": "N/A"}
    comparison_table.append(row1)

    # Metric 2 statistics
    row2 = {"metric_name": metric2_name}
    for exp in exps:
        stats = comparison_data.get(exp, {}).get(metric2_name, {})
        row2[exp] = stats if isinstance(stats, dict) else {"mean": "N/A", "median": "N/A"}
    comparison_table.append(row2)

    # Sample count
    row3 = {"metric_name": "sample_count"}
    for exp in exps:
        stats = comparison_data.get(exp, {}).get(metric1_name, {})
        row3[exp] = stats.get("count", 0) if isinstance(stats, dict) else 0
    comparison_table.append(row3)

    return jsonify({
        "comparison_table": comparison_table,
        "metric1_name": metric1_name,
        "metric2_name": metric2_name
    })


@app.route('/api/ours-scores')
def api_ours_scores():
    """Get aggregated scores for the Ours view.

    Aggregates average scores across customization/illustration/spatial/temporal.
    """
    baseline = request.args.get('baseline', '')
    exps = request.args.get('exps', '').split(',')
    evaluator = request.args.get('evaluator', 'gpt')

    if not baseline or not exps:
        return jsonify({"error": "Missing parameters"}), 400

    exps = [e.strip() for e in exps if e.strip()]
    if not exps:
        return jsonify({"error": "No experiments specified"}), 400

    return jsonify(get_ours_scores(baseline, exps, evaluator))


def normalize_image_path(path: str) -> str:
    """Normalize an image path to a format accessible by the frontend.

    Supports the following formats:
    - Absolute path (kept as-is, accessed via /api/image?path=)
    - MACRO_DIR-relative path (e.g. outputs/bagel/..., converted to absolute path string)
    - Path relative to OUTPUTS_DIR (kept as-is)

    The return value is always a path string accessible via /api/image?path=.
    """
    if not path:
        return path
    p = Path(path)
    if p.is_absolute():
        return path
    # Relative path: first try relative to MACRO_DIR
    abs_path = MACRO_DIR / p
    if abs_path.exists():
        return str(abs_path)
    # Then try relative to OUTPUTS_DIR
    abs_path2 = OUTPUTS_DIR / p
    if abs_path2.exists():
        return str(abs_path2)
    # Neither exists; return absolute path relative to MACRO_DIR (let frontend handle 404)
    return str(MACRO_DIR / p)


@app.route('/api/page')
def api_page():
    """Get paginated data."""
    baseline = request.args.get('baseline', '')
    task = request.args.get('task', '')
    category = request.args.get('category', '')
    exps = request.args.get('exps', '').split(',')
    page = int(request.args.get('page', 1))
    evaluator = request.args.get('evaluator', 'gpt')
    items_per_page = 10

    if not all([baseline, task, category, exps]):
        return jsonify({"error": "Missing parameters"}), 400

    exps = [e.strip() for e in exps if e.strip()]
    if not exps:
        return jsonify({"error": "No experiments specified"}), 400

    # Get metric names
    metric1_name, metric2_name = get_metric_names(task)

    # Load only the sample ID list of the first experiment for pagination
    first_exp = exps[0]
    category_dir = get_category_dir(baseline, first_exp, task, category)

    # Get all sample IDs
    sample_ids = []
    if category_dir.exists():
        json_files = list(category_dir.glob("[0-9]" * 8 + ".json"))
        for json_file in sorted(json_files):
            if not json_file.name.startswith("scores"):
                sample_ids.append(json_file.stem)

    # Pagination
    total_samples = len(sample_ids)
    total_pages = max(1, (total_samples + items_per_page - 1) // items_per_page)
    page = max(1, min(page, total_pages))

    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_samples)
    page_sample_ids = sample_ids[start_idx:end_idx]

    # Load only the score data needed for the current page
    all_scores = {}
    for exp in exps:
        exp_scores = load_score_for_baseline(baseline, exp, task, category, evaluator)
        all_scores[exp] = {sid: exp_scores.get(sid, {}) for sid in page_sample_ids}

    # Build page data
    samples_data = []
    for sample_id in page_sample_ids:
        base_sample = load_single_sample(baseline, first_exp, task, category, sample_id)
        if not base_sample:
            continue

        instruction = base_sample.get("instruction") or base_sample.get("prompt", "")

        # Normalize input image paths
        input_images = [normalize_image_path(p) for p in base_sample.get("input_images", [])]
        target_image = normalize_image_path(base_sample.get("target_image", ""))

        sample_data = {
            "sample_id": sample_id,
            "instruction": instruction,
            "input_images": input_images,
            "target_image": target_image,
            "experiments": {}
        }

        # Add data for each experiment
        for exp in exps:
            exp_sample = load_single_sample(baseline, exp, task, category, sample_id)
            exp_scores = all_scores.get(exp, {}).get(sample_id, {})

            if exp_sample:
                output_image = normalize_image_path(exp_sample.get("output_image", ""))
                sample_data["experiments"][exp] = {
                    "output_image": output_image,
                    "metric1_score": exp_scores.get(metric1_name),
                    "metric2_score": exp_scores.get(metric2_name),
                    "reasoning": exp_scores.get("reasoning", ""),
                    "consistency_scores_list": exp_scores.get("consistency_scores_list", [])
                }
            else:
                sample_data["experiments"][exp] = {
                    "output_image": "",
                    "metric1_score": None,
                    "metric2_score": None,
                    "reasoning": ""
                }

        samples_data.append(sample_data)

    return jsonify({
        "samples": samples_data,
        "sample_ids": page_sample_ids,
        "current_page": page,
        "total_pages": total_pages,
        "total_samples": total_samples,
        "metric1_name": metric1_name,
        "metric2_name": metric2_name
    })


@app.route('/api/image')
def api_image():
    """Serve image files.

    Supports two path formats:
    1. Absolute path (used directly; input/target images are usually absolute paths)
    2. Relative path (relative to MACRO_DIR or outputs directory)
    """
    path = request.args.get('path', '')
    if not path:
        return "Missing path", 400

    p = Path(path)

    # If relative path, try MACRO_DIR then OUTPUTS_DIR
    if not p.is_absolute():
        candidate = MACRO_DIR / p
        if candidate.exists():
            p = candidate
        else:
            p = OUTPUTS_DIR / p

    # Safety check: file must exist
    if not p.is_file():
        return "Image not found", 404

    return send_file(str(p))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind')
    parser.add_argument('--port', type=int, default=8416, help='Port to bind')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    print(f"Starting visualization server at http://{args.host}:{args.port}")
    print(f"Outputs directory: {OUTPUTS_DIR}")
    app.run(host=args.host, port=args.port, debug=args.debug)
