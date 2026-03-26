"""
可视化服务 - 支持多baseline的评测结果可视化
"""
import json
import math
from pathlib import Path
from typing import Dict, List, Optional
from flask import Flask, render_template, jsonify, request, send_file
from collections import defaultdict
import re

# 路径配置
SCRIPT_DIR = Path(__file__).parent
MACRO_DIR = SCRIPT_DIR.parent
OUTPUTS_DIR = MACRO_DIR / "outputs"

# 支持的baseline模型
SUPPORTED_BASELINES = ["bagel", "omnigen", "qwen"]

# 支持的任务和类别（使用LLM评分的多参考图任务）
SUPPORTED_TASKS = ["customization", "spatial", "illustration", "temporal"]
IMAGE_NUM_CATEGORIES = ["1-3", "4-5", "6-7", ">=8"]

app = Flask(__name__, template_folder=str(SCRIPT_DIR / "templates" / "compare"))
app.static_folder = str(SCRIPT_DIR / "templates" / "compare" / "static")


def get_category_dir(baseline: str, exp_name: str, task: str, category: str = "") -> Path:
    """获取指定baseline/实验/任务/类别的输出目录
    
    目录结构: outputs/{baseline}/{exp_name}/{task}/{category}
    
    支持 api:{model} 格式（不作为单独 baseline，作为实验选项对比）:
    - api 实验路径: outputs/api/{model_name}/{task}/{category}
    """
    # api:{model_name} 实验：路径为 outputs/api/{model_name}/...
    if exp_name.startswith("api:"):
        model_name = exp_name[4:].strip()
        if not model_name:
            return OUTPUTS_DIR / "api" / "_" / task / category  # 占位，实际不存在
        base_dir = OUTPUTS_DIR / "api" / model_name / task / category
        return base_dir
    
    # 常规 baseline 实验
    base_dir = OUTPUTS_DIR / baseline / exp_name / task / category
    return base_dir


def get_baselines() -> List[str]:
    """获取所有有输出的baseline"""
    baselines = []
    for baseline in SUPPORTED_BASELINES:
        baseline_dir = OUTPUTS_DIR / baseline
        if baseline_dir.exists() and any(baseline_dir.iterdir()):
            baselines.append(baseline)
    return baselines


def get_tasks_for_baseline(baseline: str) -> List[str]:
    """获取指定baseline下有输出的任务列表"""
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
    """获取指定baseline和任务下的实验列表
    
    包含：1) baseline 下的实验；2) api 实验 api:{model_name}（来自 outputs/api/{model}/）
    """
    exps = set()
    
    # 1) baseline 下的实验
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
    
    # 2) api 实验：outputs/api/{model_name}/，以 api:{model_name} 作为实验选项
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
    """获取 Ours 视图可用的实验列表（customization/illustration/spatial/temporal 的并集）"""
    exps = set()
    for task in ["customization", "illustration", "spatial", "temporal"]:
        exps.update(get_exps_for_baseline_task(baseline, task))
    return sorted(exps)


def get_exp_structure_for_ours(baseline: str) -> Dict:
    """获取 Ours 视图的实验结构"""
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
    """解析实验名称，提取基础名称和step
    
    支持两种格式:
    1. 带 _step 后缀: "all_no_t2i_0010000_step5000" -> ("all_no_t2i_0010000", "5000")
    2. 末尾数字作为 step: "bagel_with_t2i_nano_max7_0010000" -> ("bagel_with_t2i_nano_max7", "0010000")
    3. 无 step: "bagel" -> ("bagel", None)
    
    对于末尾数字格式，要求：
    - 末尾是数字（至少4位，通常是训练步数）
    - 数字前有下划线或直接连接
    - 基础名称不能是纯数字
    """
    # 首先尝试匹配 _step(\d+)$ 格式
    step_match = re.search(r'_step(\d+)$', exp_name)
    if step_match:
        base_name = exp_name[:step_match.start()]
        step = step_match.group(1)
        return (base_name, step)
    
    # 尝试匹配末尾数字格式（至少4位数字，通常是训练步数）
    # 例如: bagel_with_t2i_nano_max7_0010000 -> ("bagel_with_t2i_nano_max7", "0010000")
    # 匹配模式：末尾是至少4位数字，前面有下划线或直接连接
    end_num_match = re.search(r'[_-](\d{4,})$', exp_name)
    if end_num_match:
        base_name = exp_name[:end_num_match.start()]
        step = end_num_match.group(1)
        # 验证基础名称不是纯数字（避免误匹配）
        if base_name and not base_name.replace('_', '').replace('-', '').isdigit():
            return (base_name, step)
    
    return (exp_name, None)


def get_exp_structure_for_baseline_task(baseline: str, task: str) -> Dict:
    """获取指定baseline和任务下的实验结构
    
    返回结构: {
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
    
    # 对steps进行自然排序（数字排序）
    sorted_steps = sorted(steps, key=lambda x: int(x) if x and x.isdigit() else 0)
    
    return {
        "base_names": sorted(base_names),
        "steps": sorted_steps,
        "exp_map": dict(exp_map)
    }


def get_categories_for_baseline_task_exp(baseline: str, task: str, exp: str) -> List[str]:
    """获取指定baseline/任务/实验下的类别列表
    
    对于 api:{model} 实验，从 outputs/api/{model}/{task} 获取类别
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
    """加载单个样本数据（优化版本，只加载指定sample_id的数据）
    
    Args:
        baseline: baseline名称
        exp: 实验名称
        task: 任务名称
        category: 类别
        sample_id: 样本ID（例如 "00000001"）
    
    Returns:
        样本数据字典，如果不存在则返回None
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
    """加载指定baseline/实验/任务/类别的所有样本
    
    目录结构: {category_dir}/{idx:08d}.json
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
    """加载评分数据
    
    支持两种格式（按优先级）：
    1. 新格式: {category_dir}/{sample_id}.score (每个样本一个评分文件，包含 gpt_scores 和 gemini_scores)
    2. 旧格式: {category_dir}/scores.json 或 scores_gemini.json (统一评分文件)
    """
    category_dir = get_category_dir(baseline, exp, task, category)
    
    # 优先加载 .score 文件（新格式）
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
    
    # 退而求其次：旧格式 scores.json
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
    计算调和平均
    
    Args:
        scores: 得分列表
        
    Returns:
        调和平均，如果列表为空或包含0值则返回0
    """
    if not scores:
        return 0
    # 过滤掉0值和负数，避免除零错误
    positive_scores = [s if s > 0 else 1 for s in scores]
    if not positive_scores:
        return 0
    n = len(positive_scores)
    return n / sum(1.0 / s for s in positive_scores)


def get_metric_names(task: str) -> tuple:
    """获取指定任务的指标名称
    
    返回的名称与评分文件中的实际字段名一致：
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
    """计算统计信息
    
    对于 customization/illustration/spatial/temporal 任务，额外计算 average = mean(sqrt(m1*m2))
    """
    metric1_name, metric2_name = get_metric_names(task)
    
    metric1_scores = []
    metric2_scores = []
    average_scores = []  # sqrt(m1*m2) 的列表，用于计算 average
    
    for sample_id, score_data in scores.items():
        if isinstance(score_data, dict):
            m1 = score_data.get(metric1_name)
            m2 = score_data.get(metric2_name)
            
            # 对于 customization 任务，consistency 可能是列表，需要计算调和平均
            if task == "customization" and metric1_name == "consistency":
                if isinstance(m1, list) and len(m1) > 0:
                    m1 = harmonic_mean(m1)  # 使用调和平均
            
            if m1 is not None and isinstance(m1, (int, float)):
                metric1_scores.append(m1)
            if m2 is not None and isinstance(m2, (int, float)):
                metric2_scores.append(m2)
            # 同时有 m1 和 m2 时，计算 sqrt(m1*m2) 用于 average
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
    # 对于 customization/illustration/spatial/temporal 任务，添加 average
    if task in ["customization", "illustration", "spatial", "temporal"]:
        result["average"] = calc_stats(average_scores)
    return result


def get_average_for_task_category(baseline: str, exp: str, task: str, category: str, evaluator: str = 'gpt') -> Optional[float]:
    """计算单个 task+category 的 average（sqrt(score_a*score_b) 的均值）
    
    返回该 task+category 下所有样本的 sqrt(m1*m2) 的均值，若无可计算数据则返回 None
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
    """获取单个 task+category 的评测样本数"""
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
    """获取 Ours 视图的聚合得分
    
    对于每个实验，聚合 customization/illustration/spatial/temporal 四个任务的得分：
    - 每个 task+category 得到一个 average（sqrt(m1*m2) 的均值）
    - 每个 task 的任务均值 = 该 task 下所有 category 的 average 的算术平均
    - 总均值 = 四个任务均值的算术平均
    
    返回结构:
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
    
    # 计算总均值：对每个 exp，(cust_mean + illus_mean + spatial_mean + temporal_mean) / 4
    # total_count: 四个任务的样本数总和（用于校验）
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


# ==================== API 路由 ====================

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/baselines')
def api_baselines():
    """获取所有baseline列表"""
    return jsonify(get_baselines())


@app.route('/api/tasks/<baseline>')
def api_tasks(baseline: str):
    """获取指定baseline的任务列表"""
    return jsonify(get_tasks_for_baseline(baseline))


@app.route('/api/exp-structure/<baseline>/<task>')
def api_exp_structure(baseline: str, task: str):
    """获取指定baseline和任务的实验结构"""
    return jsonify(get_exp_structure_for_baseline_task(baseline, task))


@app.route('/api/ours-exp-structure/<baseline>')
def api_ours_exp_structure(baseline: str):
    """获取 Ours 视图的实验结构"""
    return jsonify(get_exp_structure_for_ours(baseline))


@app.route('/api/categories/<baseline>/<task>/<exp>')
def api_categories(baseline: str, task: str, exp: str):
    """获取指定baseline/任务/实验的类别列表"""
    return jsonify(get_categories_for_baseline_task_exp(baseline, task, exp))


@app.route('/api/comparison')
def api_comparison():
    """获取多实验对比统计数据"""
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
    
    # 获取指标名称
    metric1_name, metric2_name = get_metric_names(task)
    
    # 收集各实验的统计数据
    comparison_data = {}
    for exp in exps:
        scores = load_score_for_baseline(baseline, exp, task, category, evaluator)
        stats = calculate_statistics(scores, task)
        comparison_data[exp] = stats
    
    # 构建对比表格
    comparison_table = []
    
    # 对于 customization/illustration/spatial/temporal，先添加 average 行（sqrt(score_a*score_b)）
    if task in ["customization", "illustration", "spatial", "temporal"]:
        row_avg = {"metric_name": "average"}
        for exp in exps:
            stats = comparison_data.get(exp, {}).get("average", {})
            row_avg[exp] = stats if isinstance(stats, dict) else {"mean": "N/A", "median": "N/A"}
        comparison_table.append(row_avg)
    
    # 指标1统计
    row1 = {"metric_name": metric1_name}
    for exp in exps:
        stats = comparison_data.get(exp, {}).get(metric1_name, {})
        row1[exp] = stats if isinstance(stats, dict) else {"mean": "N/A", "median": "N/A"}
    comparison_table.append(row1)
    
    # 指标2统计
    row2 = {"metric_name": metric2_name}
    for exp in exps:
        stats = comparison_data.get(exp, {}).get(metric2_name, {})
        row2[exp] = stats if isinstance(stats, dict) else {"mean": "N/A", "median": "N/A"}
    comparison_table.append(row2)
    
    # 样本数量
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
    """获取 Ours 视图的聚合得分
    
    聚合 customization/illustration/spatial/temporal 四个任务的 average 得分
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
    """将图像路径规范化为前端可访问的格式
    
    支持以下格式：
    - 绝对路径（保持不变，通过 /api/image?path= 访问）
    - MACRO_DIR 相对路径（如 outputs/bagel/...，转换为绝对路径字符串）
    - 相对于 OUTPUTS_DIR 的路径（保持不变）
    
    返回值始终是一个可通过 /api/image?path= 访问的路径字符串
    """
    if not path:
        return path
    p = Path(path)
    if p.is_absolute():
        return path
    # 相对路径：先尝试相对于 MACRO_DIR
    abs_path = MACRO_DIR / p
    if abs_path.exists():
        return str(abs_path)
    # 再尝试相对于 OUTPUTS_DIR
    abs_path2 = OUTPUTS_DIR / p
    if abs_path2.exists():
        return str(abs_path2)
    # 都不存在，返回相对于 MACRO_DIR 的绝对路径（让前端处理 404）
    return str(MACRO_DIR / p)


@app.route('/api/page')
def api_page():
    """获取分页数据"""
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
    
    # 获取指标名称
    metric1_name, metric2_name = get_metric_names(task)
    
    # 先只加载第一个实验的样本ID列表，用于分页
    first_exp = exps[0]
    category_dir = get_category_dir(baseline, first_exp, task, category)
    
    # 获取所有样本ID
    sample_ids = []
    if category_dir.exists():
        json_files = list(category_dir.glob("[0-9]" * 8 + ".json"))
        for json_file in sorted(json_files):
            if not json_file.name.startswith("scores"):
                sample_ids.append(json_file.stem)
    
    # 分页
    total_samples = len(sample_ids)
    total_pages = max(1, (total_samples + items_per_page - 1) // items_per_page)
    page = max(1, min(page, total_pages))
    
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_samples)
    page_sample_ids = sample_ids[start_idx:end_idx]
    
    # 只加载当前页需要的评分数据
    all_scores = {}
    for exp in exps:
        exp_scores = load_score_for_baseline(baseline, exp, task, category, evaluator)
        all_scores[exp] = {sid: exp_scores.get(sid, {}) for sid in page_sample_ids}
    
    # 构建页面数据
    samples_data = []
    for sample_id in page_sample_ids:
        base_sample = load_single_sample(baseline, first_exp, task, category, sample_id)
        if not base_sample:
            continue
        
        instruction = base_sample.get("instruction") or base_sample.get("prompt", "")
        
        # 规范化输入图像路径
        input_images = [normalize_image_path(p) for p in base_sample.get("input_images", [])]
        target_image = normalize_image_path(base_sample.get("target_image", ""))
        
        sample_data = {
            "sample_id": sample_id,
            "instruction": instruction,
            "input_images": input_images,
            "target_image": target_image,
            "experiments": {}
        }
        
        # 添加各实验的数据
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
    """提供图片访问
    
    支持两种路径格式：
    1. 绝对路径（直接使用，输入/目标图像通常为绝对路径）
    2. 相对路径（相对于 MACRO_DIR 或 outputs 目录）
    """
    path = request.args.get('path', '')
    if not path:
        return "Missing path", 400
    
    p = Path(path)
    
    # 如果是相对路径，依次尝试 MACRO_DIR 和 OUTPUTS_DIR
    if not p.is_absolute():
        candidate = MACRO_DIR / p
        if candidate.exists():
            p = candidate
        else:
            p = OUTPUTS_DIR / p
    
    # 安全检查：文件必须存在
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
