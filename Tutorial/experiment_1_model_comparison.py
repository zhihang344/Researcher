#!/usr/bin/env python3
"""
实验1: CycleReviewer 8B vs DeepReviewer 7B 模型对比实验

比较两个评审模型的性能差异：
- 评分差异
- 发现问题的能力
- 评审速度
- 评审详细程度
"""

import json
import time
import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from math import pi
import torch

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# ==================== GPU 配置 ====================
GPU_FOR_CYCLE_REVIEWER = "3"  # CycleReviewer使用的GPU编号
GPU_FOR_DEEP_REVIEWER = "4"   # DeepReviewer使用的GPU编号
USE_SEPARATE_GPUS = True       # 是否使用不同GPU

print("=" * 80)
print("实验 1: CycleReviewer 8B vs DeepReviewer 7B 模型对比")
print("=" * 80)

# ==================== 辅助函数 ====================
def extract_metrics(review_result):
    """从评审结果中提取量化指标"""
    if not review_result:
        return None
    
    metrics = {
        'avg_rating': 0.0,
        'num_reviewers': 0,
        'avg_soundness': 0.0,
        'avg_presentation': 0.0,
        'avg_contribution': 0.0,
        'total_weaknesses': 0,
        'total_strengths': 0,
        'total_suggestions': 0,
        'total_questions': 0,
        'review_length': 0
    }
    
    is_cycle_format = 'content' in review_result and 'avg_rating' in review_result
    
    if is_cycle_format:
        # CycleReviewer 格式
        try:
            metrics['avg_rating'] = float(review_result.get('avg_rating', 0))
        except (ValueError, TypeError):
            import re
            rating_str = str(review_result.get('rating', '0'))
            numbers = re.findall(r'\d+', rating_str)
            if numbers:
                metrics['avg_rating'] = float(numbers[0])
        
        for field, key in [('soundness', 'avg_soundness'), 
                           ('presentation', 'avg_presentation'),
                           ('contribution', 'avg_contribution')]:
            try:
                import re
                field_str = str(review_result.get(field, '0'))
                numbers = re.findall(r'\d+', field_str)
                if numbers:
                    metrics[key] = float(numbers[0])
            except:
                pass
        
        # 计数反馈
        for field, key in [('weaknesses', 'total_weaknesses'),
                           ('strength', 'total_strengths'),
                           ('strengths', 'total_strengths')]:
            if field in review_result and isinstance(review_result[field], list):
                for text in review_result[field]:
                    if isinstance(text, str):
                        lines = text.strip().split('\n')
                        count = sum(1 for line in lines if line.strip() and 
                                   not line.strip().startswith('#') and
                                   len(line.strip()) > 10)
                        metrics[key] += max(1, count)
        
        if 'questions' in review_result and isinstance(review_result['questions'], list):
            for text in review_result['questions']:
                if isinstance(text, str):
                    lines = text.strip().split('\n')
                    count = sum(1 for line in lines if line.strip() and 
                               (line.strip()[0].isdigit() or line.strip().startswith('-')))
                    metrics['total_questions'] += count
        
        if 'reviews' in review_result and isinstance(review_result['reviews'], list):
            metrics['num_reviewers'] = len(review_result['reviews'])
        else:
            metrics['num_reviewers'] = 1
        
        if 'content' in review_result:
            metrics['review_length'] = len(review_result['content'])
    
    else:
        # DeepReviewer 格式
        import re
        reviews = review_result.get('reviews', [])
        if isinstance(reviews, list):
            metrics['num_reviewers'] = len(reviews)
            
            ratings, soundness, presentation, contribution = [], [], [], []
            
            for review in reviews:
                if not isinstance(review, dict):
                    continue
                
                review_text = review.get('text', '')
                if not isinstance(review_text, str):
                    continue
                
                # 提取评分
                for pattern, lst in [(r'###?\s*Soundness\s*\n+\s*(\d+)', soundness),
                                     (r'###?\s*Presentation\s*\n+\s*(\d+)', presentation),
                                     (r'###?\s*Contribution\s*\n+\s*(\d+)', contribution),
                                     (r'###?\s*Rating\s*\n+\s*(\d+)', ratings)]:
                    match = re.search(pattern, review_text, re.IGNORECASE)
                    if match:
                        lst.append(float(match.group(1)))
                
                # 计数反馈
                for section_name, metric_key in [('Weaknesses?', 'total_weaknesses'),
                                                  ('Strengths?', 'total_strengths'),
                                                  ('Questions?', 'total_questions')]:
                    section = re.search(f'###?\s*{section_name}\s*\n+(.*?)(?=###|\Z)', 
                                       review_text, re.DOTALL | re.IGNORECASE)
                    if section:
                        text = section.group(1)
                        lines = [l.strip() for l in text.split('\n') 
                                if l.strip() and (l.strip().startswith('-') or 
                                                  (len(l.strip()) > 0 and l.strip()[0].isdigit()))]
                        lines = [l for l in lines if len(l) > 15]
                        metrics[metric_key] += len(lines)
            
            if ratings:
                metrics['avg_rating'] = np.mean(ratings)
            if soundness:
                metrics['avg_soundness'] = np.mean(soundness)
            if presentation:
                metrics['avg_presentation'] = np.mean(presentation)
            if contribution:
                metrics['avg_contribution'] = np.mean(contribution)
        
        if 'raw_text' in review_result:
            metrics['review_length'] = len(review_result['raw_text'])
    
    return metrics


def print_metrics(metrics, title="Metrics"):
    """打印指标摘要"""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:25s}: {value:.2f}")
            else:
                print(f"{key:25s}: {value}")
    else:
        print("No metrics available")
    print(f"{'='*50}\n")


# ==================== 加载测试数据 ====================
print("\n[1/5] 加载测试数据...")
with open('generated_paper.json', 'r', encoding='utf-8') as f:
    all_papers = json.load(f)

test_paper = all_papers[0]
print(f"✓ 测试论文: {test_paper['title']}")
print(f"  LaTeX长度: {len(test_paper['latex'])} 字符")


# ==================== 初始化模型 ====================
print("\n[2/5] 初始化模型...")

if USE_SEPARATE_GPUS:
    # 初始化 CycleReviewer
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_FOR_CYCLE_REVIEWER
    from ai_researcher import CycleReviewer
    print(f"  初始化 CycleReviewer 8B (GPU {GPU_FOR_CYCLE_REVIEWER})...")
    cycle_reviewer = CycleReviewer(
        model_size="8B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75
    )
    print(f"  ✓ CycleReviewer 已就绪")
    
    # 初始化 DeepReviewer
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_FOR_DEEP_REVIEWER
    from ai_researcher import DeepReviewer
    print(f"  初始化 DeepReviewer 7B (GPU {GPU_FOR_DEEP_REVIEWER})...")
    deep_reviewer = DeepReviewer(
        custom_model_name="WestlakeNLP/DeepReviewer-7B",
        device="cuda",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75
    )
    print(f"  ✓ DeepReviewer 已就绪")
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_FOR_CYCLE_REVIEWER
    from ai_researcher import CycleReviewer, DeepReviewer
    
    cycle_reviewer = CycleReviewer(model_size="8B", tensor_parallel_size=1, gpu_memory_utilization=0.6)
    deep_reviewer = DeepReviewer(custom_model_name="WestlakeNLP/DeepReviewer-7B", 
                                 device="cuda", tensor_parallel_size=1, gpu_memory_utilization=0.6)


# ==================== 运行评审 ====================
print("\n[3/5] 运行评审...")

print("  CycleReviewer 评审中...")
start_time = time.time()
cycle_review = cycle_reviewer.evaluate([test_paper['latex']])[0]
cycle_time = time.time() - start_time
print(f"  ✓ CycleReviewer 完成 ({cycle_time:.2f}秒)")

print("  DeepReviewer 评审中...")
start_time = time.time()
deep_review = deep_reviewer.evaluate([test_paper['latex']], mode="Standard Mode", reviewer_num=4)[0]
deep_time = time.time() - start_time
print(f"  ✓ DeepReviewer 完成 ({deep_time:.2f}秒)")


# ==================== 提取指标 ====================
print("\n[4/5] 提取并分析指标...")

cycle_metrics = extract_metrics(cycle_review)
deep_metrics = extract_metrics(deep_review)

cycle_metrics['review_time'] = cycle_time
deep_metrics['review_time'] = deep_time

print_metrics(cycle_metrics, "CycleReviewer 8B Metrics")
print_metrics(deep_metrics, "DeepReviewer 7B Metrics")

# 对比分析
print("\n关键发现：")
print("=" * 60)
rating_diff = cycle_metrics['avg_rating'] - deep_metrics['avg_rating']
print(f"1. 评分差异: {rating_diff:+.2f} (Cycle - Deep)")
print(f"   → {'CycleReviewer更严格' if rating_diff < 0 else 'DeepReviewer更严格'}")

weakness_diff = cycle_metrics['total_weaknesses'] - deep_metrics['total_weaknesses']
print(f"\n2. 发现问题数量: {cycle_metrics['total_weaknesses']} vs {deep_metrics['total_weaknesses']}")
print(f"   → {'CycleReviewer' if weakness_diff > 0 else 'DeepReviewer'}发现更多问题")

time_speedup = deep_time / cycle_time
print(f"\n3. 速度对比: CycleReviewer快{time_speedup:.2f}倍")

detail_ratio = deep_metrics['review_length'] / cycle_metrics['review_length']
print(f"4. 详细程度: DeepReviewer详细{detail_ratio:.2f}倍")


# ==================== 可视化 ====================
print("\n[5/5] 生成可视化...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('实验1: CycleReviewer 8B vs DeepReviewer 7B 对比', fontsize=16, fontweight='bold')

# 1. 评分对比
metrics_to_plot = ['avg_rating', 'avg_soundness', 'avg_presentation', 'avg_contribution']
x_pos = np.arange(len(metrics_to_plot))
width = 0.35

cycle_scores = [cycle_metrics[m] for m in metrics_to_plot]
deep_scores = [deep_metrics[m] for m in metrics_to_plot]

axes[0, 0].bar(x_pos - width/2, cycle_scores, width, label='CycleReviewer 8B', alpha=0.8)
axes[0, 0].bar(x_pos + width/2, deep_scores, width, label='DeepReviewer 7B', alpha=0.8)
axes[0, 0].set_ylabel('Score')
axes[0, 0].set_title('评分对比')
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(['Overall', 'Soundness', 'Presentation', 'Contribution'], rotation=45, ha='right')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. 反馈数量对比
feedback_metrics = ['total_weaknesses', 'total_strengths', 'total_questions']
x_pos2 = np.arange(len(feedback_metrics))
cycle_feedback = [cycle_metrics[m] for m in feedback_metrics]
deep_feedback = [deep_metrics[m] for m in feedback_metrics]

axes[0, 1].bar(x_pos2 - width/2, cycle_feedback, width, label='CycleReviewer', alpha=0.8)
axes[0, 1].bar(x_pos2 + width/2, deep_feedback, width, label='DeepReviewer', alpha=0.8)
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('反馈数量对比')
axes[0, 1].set_xticks(x_pos2)
axes[0, 1].set_xticklabels(['Weaknesses', 'Strengths', 'Questions'], rotation=45, ha='right')
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

# 3. 时间对比
axes[0, 2].bar(['CycleReviewer', 'DeepReviewer'], 
               [cycle_time, deep_time], alpha=0.8, color=['#1f77b4', '#ff7f0e'])
axes[0, 2].set_ylabel('Time (seconds)')
axes[0, 2].set_title('评审时间')
axes[0, 2].grid(axis='y', alpha=0.3)

# 4. 长度对比
axes[1, 0].bar(['CycleReviewer', 'DeepReviewer'], 
               [cycle_metrics['review_length'], deep_metrics['review_length']], 
               alpha=0.8, color=['#1f77b4', '#ff7f0e'])
axes[1, 0].set_ylabel('Characters')
axes[1, 0].set_title('评审长度')
axes[1, 0].grid(axis='y', alpha=0.3)

# 5. 审稿人数
axes[1, 1].bar(['CycleReviewer', 'DeepReviewer'], 
               [cycle_metrics['num_reviewers'], deep_metrics['num_reviewers']], 
               alpha=0.8, color=['#1f77b4', '#ff7f0e'])
axes[1, 1].set_ylabel('Number')
axes[1, 1].set_title('审稿人数量')
axes[1, 1].grid(axis='y', alpha=0.3)

# 6. 雷达图
categories = ['Rating', 'Soundness', 'Presentation', 'Contribution', 'Feedback']
N = len(categories)

cycle_radar = [
    cycle_metrics['avg_rating'] / 10,
    cycle_metrics['avg_soundness'] / 4,
    cycle_metrics['avg_presentation'] / 4,
    cycle_metrics['avg_contribution'] / 4,
    (cycle_metrics['total_weaknesses'] + cycle_metrics['total_strengths']) / 20
]
deep_radar = [
    deep_metrics['avg_rating'] / 10,
    deep_metrics['avg_soundness'] / 4,
    deep_metrics['avg_presentation'] / 4,
    deep_metrics['avg_contribution'] / 4,
    (deep_metrics['total_weaknesses'] + deep_metrics['total_strengths']) / 20
]

angles = [n / float(N) * 2 * pi for n in range(N)]
cycle_radar += cycle_radar[:1]
deep_radar += deep_radar[:1]
angles += angles[:1]

ax = plt.subplot(2, 3, 6, projection='polar')
ax.plot(angles, cycle_radar, 'o-', linewidth=2, label='CycleReviewer', color='#1f77b4')
ax.fill(angles, cycle_radar, alpha=0.25, color='#1f77b4')
ax.plot(angles, deep_radar, 'o-', linewidth=2, label='DeepReviewer', color='#ff7f0e')
ax.fill(angles, deep_radar, alpha=0.25, color='#ff7f0e')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=8)
ax.set_ylim(0, 1)
ax.set_title('综合性能对比', y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)

plt.tight_layout()
plt.savefig('experiment1_model_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ 图表已保存: experiment1_model_comparison.png")

# 保存数据
comparison_df = pd.DataFrame({
    'CycleReviewer_8B': cycle_metrics,
    'DeepReviewer_7B': deep_metrics
})

results = {
    'timestamp': datetime.now().isoformat(),
    'test_paper': test_paper['title'],
    'metrics': {
        'cyclereviewer': cycle_metrics,
        'deepreviewer': deep_metrics
    },
    'comparison': {
        'rating_diff': rating_diff,
        'weakness_diff': weakness_diff,
        'time_speedup': time_speedup,
        'detail_ratio': detail_ratio
    }
}

with open('experiment1_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("  ✓ 结果已保存: experiment1_results.json")

comparison_df.T.to_excel('experiment1_comparison.xlsx')
print("  ✓ Excel已保存: experiment1_comparison.xlsx")


# ==================== 清理显存 ====================
print("\n[清理] 释放显存...")
del cycle_reviewer, deep_reviewer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("  ✓ 显存已释放")

print("\n" + "=" * 80)
print("✅ 实验1完成！")
print("=" * 80)
