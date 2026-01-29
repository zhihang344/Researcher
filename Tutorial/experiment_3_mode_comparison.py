#!/usr/bin/env python3
"""
实验3: 评审模式对比实验 (Fast Mode vs Standard Mode)

对比DeepReviewer的两种评审模式：
- Fast Mode: 快速评审，单个审稿人
- Standard Mode: 标准评审，多个审稿人

评估维度：
- 评审时间
- 评审详细程度
- 发现问题的数量
- 评分差异
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
import torch

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# ==================== 配置 ====================
GPU_FOR_REVIEWER = "4"     # DeepReviewer使用的GPU
NUM_REVIEWERS = 3          # Standard Mode的审稿人数量

print("=" * 80)
print("实验3: Fast Mode vs Standard Mode 对比")
print("=" * 80)


# ==================== 辅助函数 ====================
def extract_metrics(review_result, mode_name=""):
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
    
    import re
    
    # 如果有raw_text，先记录长度
    if 'raw_text' in review_result:
        metrics['review_length'] = len(review_result['raw_text'])
        raw_text = review_result['raw_text']
        
        # Fast Mode可能没有reviews列表，直接从raw_text提取
        if mode_name == "Fast Mode" or not review_result.get('reviews'):
            # 尝试从raw_text直接提取评分
            rating_match = re.search(r'(?:Rating|Overall)[:：\s]*(\d+(?:\.\d+)?)', raw_text, re.IGNORECASE)
            if rating_match:
                metrics['avg_rating'] = float(rating_match.group(1))
            
            soundness_match = re.search(r'Soundness[:：\s]*(\d+)', raw_text, re.IGNORECASE)
            if soundness_match:
                metrics['avg_soundness'] = float(soundness_match.group(1))
            
            presentation_match = re.search(r'Presentation[:：\s]*(\d+)', raw_text, re.IGNORECASE)
            if presentation_match:
                metrics['avg_presentation'] = float(presentation_match.group(1))
            
            contribution_match = re.search(r'Contribution[:：\s]*(\d+)', raw_text, re.IGNORECASE)
            if contribution_match:
                metrics['avg_contribution'] = float(contribution_match.group(1))
            
            metrics['num_reviewers'] = 1  # Fast Mode通常是单审稿人
            return metrics
    
    # Standard Mode: 从reviews列表提取
    reviews = review_result.get('reviews', [])
    if isinstance(reviews, list) and reviews:
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
    
    return metrics


# ==================== 加载测试数据 ====================
print("\n[1/4] 加载测试数据...")
with open('Tutorial/generated_paper.json', 'r', encoding='utf-8') as f:
    all_papers = json.load(f)

test_paper = all_papers[0]
print(f"  ✓ 测试论文: {test_paper['title']}")
print(f"    LaTeX长度: {len(test_paper['latex'])} 字符")


# ==================== 初始化模型 ====================
print("\n[2/4] 初始化 DeepReviewer...")
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_FOR_REVIEWER

from ai_researcher import DeepReviewer
reviewer = DeepReviewer(
    custom_model_name="WestlakeNLP/DeepReviewer-7B",
    device="cuda",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.95
)
print(f"  ✓ DeepReviewer 已就绪 (GPU {GPU_FOR_REVIEWER})")


# ==================== 运行实验 ====================
print("\n[3/4] 运行评审...")

# Fast Mode（参照tutorial_3的调用方式）
print("  [1/2] Fast Mode 评审中...")
start_time = time.time()
fast_review = reviewer.evaluate(
    [test_paper['latex']],
    mode="Fast Mode"
)[0]
fast_time = time.time() - start_time
print(f"  ✓ Fast Mode 完成 ({fast_time:.2f}秒)")
print(f"    实际输出长度: {len(fast_review.get('raw_text', ''))} 字符")

# Standard Mode
print(f"  [2/2] Standard Mode 评审中 ({NUM_REVIEWERS}个审稿人)...")
start_time = time.time()
standard_review = reviewer.evaluate(
    [test_paper['latex']],
    mode="Standard Mode",
    reviewer_num=NUM_REVIEWERS
)[0]
standard_time = time.time() - start_time
print(f"  ✓ Standard Mode 完成 ({standard_time:.2f}秒)")
print(f"    实际输出长度: {len(standard_review.get('raw_text', ''))} 字符")


# ==================== 提取指标 ====================
print("\n提取指标...")

# 🔍 调试：先看看Fast Mode的实际结构
print("\n【调试】Fast Mode结构:")
print(f"  raw_text长度: {len(fast_review.get('raw_text', ''))}")
print(f"  reviews数量: {len(fast_review.get('reviews', []))}")
print(f"  前500字符:\n{fast_review.get('raw_text', '')[:500]}")

fast_metrics = extract_metrics(fast_review, mode_name="Fast Mode")
standard_metrics = extract_metrics(standard_review, mode_name="Standard Mode")

fast_metrics['review_time'] = fast_time
standard_metrics['review_time'] = standard_time

# 打印对比
print("\n" + "=" * 80)
print("指标对比")
print("=" * 80)

comparison = pd.DataFrame({
    'Fast Mode': fast_metrics,
    'Standard Mode': standard_metrics
})

print(comparison.T)

# 关键发现
print("\n关键发现:")
print("=" * 60)

time_speedup = standard_time / fast_time
print(f"1. 速度: Fast Mode 快 {time_speedup:.2f}倍")
print(f"   Fast: {fast_time:.1f}秒, Standard: {standard_time:.1f}秒")

detail_ratio = standard_metrics['review_length'] / fast_metrics['review_length'] if fast_metrics['review_length'] > 0 else 0
print(f"\n2. 详细程度: Standard Mode 详细 {detail_ratio:.2f}倍")
print(f"   Fast: {fast_metrics['review_length']}字符, Standard: {standard_metrics['review_length']}字符")

rating_diff = standard_metrics['avg_rating'] - fast_metrics['avg_rating']
print(f"\n3. 评分差异: {rating_diff:+.2f} (Standard - Fast)")
print(f"   Fast: {fast_metrics['avg_rating']:.2f}, Standard: {standard_metrics['avg_rating']:.2f}")

feedback_fast = fast_metrics['total_weaknesses'] + fast_metrics['total_strengths']
feedback_standard = standard_metrics['total_weaknesses'] + standard_metrics['total_strengths']
feedback_ratio = feedback_standard / feedback_fast if feedback_fast > 0 else 0
print(f"\n4. 反馈量: Standard Mode 多 {feedback_ratio:.2f}倍")
print(f"   Fast: {feedback_fast}条, Standard: {feedback_standard}条")

# 效率分析
fast_efficiency = fast_metrics['avg_rating'] / fast_time if fast_time > 0 else 0
standard_efficiency = standard_metrics['avg_rating'] / standard_time if standard_time > 0 else 0
print(f"\n5. 效率 (评分/秒):")
print(f"   Fast: {fast_efficiency:.4f}")
print(f"   Standard: {standard_efficiency:.4f}")


# ==================== 可视化 ====================
print("\n[4/4] 生成可视化...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('实验3: Fast Mode vs Standard Mode 对比', fontsize=16, fontweight='bold')

# 1. 时间对比
axes[0, 0].bar(['Fast Mode', 'Standard Mode'], 
               [fast_time, standard_time], 
               alpha=0.8, color=['#2ecc71', '#3498db'])
axes[0, 0].set_ylabel('Time (seconds)')
axes[0, 0].set_title('评审时间')
axes[0, 0].grid(axis='y', alpha=0.3)
for i, (mode, t) in enumerate([('Fast', fast_time), ('Standard', standard_time)]):
    axes[0, 0].text(i, t + 5, f"{t:.1f}s", ha='center', va='bottom')

# 2. 长度对比
axes[0, 1].bar(['Fast Mode', 'Standard Mode'], 
               [fast_metrics['review_length'], standard_metrics['review_length']], 
               alpha=0.8, color=['#2ecc71', '#3498db'])
axes[0, 1].set_ylabel('Characters')
axes[0, 1].set_title('评审长度')
axes[0, 1].grid(axis='y', alpha=0.3)

# 3. 审稿人数量
axes[0, 2].bar(['Fast Mode', 'Standard Mode'], 
               [fast_metrics['num_reviewers'], standard_metrics['num_reviewers']], 
               alpha=0.8, color=['#2ecc71', '#3498db'])
axes[0, 2].set_ylabel('Number')
axes[0, 2].set_title('审稿人数量')
axes[0, 2].grid(axis='y', alpha=0.3)

# 4. 评分对比
score_metrics = ['avg_rating', 'avg_soundness', 'avg_presentation', 'avg_contribution']
x_pos = np.arange(len(score_metrics))
width = 0.35

fast_scores = [fast_metrics[m] for m in score_metrics]
standard_scores = [standard_metrics[m] for m in score_metrics]

axes[1, 0].bar(x_pos - width/2, fast_scores, width, label='Fast', alpha=0.8, color='#2ecc71')
axes[1, 0].bar(x_pos + width/2, standard_scores, width, label='Standard', alpha=0.8, color='#3498db')
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('评分对比')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(['Overall', 'Soundness', 'Presentation', 'Contribution'], rotation=45, ha='right')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# 5. 反馈数量
feedback_metrics = ['total_weaknesses', 'total_strengths', 'total_questions']
x_pos2 = np.arange(len(feedback_metrics))

fast_feedback = [fast_metrics[m] for m in feedback_metrics]
standard_feedback = [standard_metrics[m] for m in feedback_metrics]

axes[1, 1].bar(x_pos2 - width/2, fast_feedback, width, label='Fast', alpha=0.8, color='#2ecc71')
axes[1, 1].bar(x_pos2 + width/2, standard_feedback, width, label='Standard', alpha=0.8, color='#3498db')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('反馈数量')
axes[1, 1].set_xticks(x_pos2)
axes[1, 1].set_xticklabels(['Weaknesses', 'Strengths', 'Questions'], rotation=45, ha='right')
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

# 6. 效率vs质量
axes[1, 2].scatter([fast_time], [fast_metrics['avg_rating']], 
                   s=200, alpha=0.7, color='#2ecc71', label='Fast', marker='o')
axes[1, 2].scatter([standard_time], [standard_metrics['avg_rating']], 
                   s=200, alpha=0.7, color='#3498db', label='Standard', marker='s')

axes[1, 2].annotate('Fast Mode', 
                    xy=(fast_time, fast_metrics['avg_rating']),
                    xytext=(10, 10), textcoords='offset points', fontsize=10, ha='left')
axes[1, 2].annotate('Standard Mode', 
                    xy=(standard_time, standard_metrics['avg_rating']),
                    xytext=(10, -15), textcoords='offset points', fontsize=10, ha='left')

axes[1, 2].set_xlabel('Review Time (seconds)')
axes[1, 2].set_ylabel('Overall Rating')
axes[1, 2].set_title('效率 vs 质量权衡')
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('experiment3_mode_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ 图表已保存: experiment3_mode_comparison.png")

# 保存数据
results = {
    'timestamp': datetime.now().isoformat(),
    'test_paper': test_paper['title'],
    'fast_mode': fast_metrics,
    'standard_mode': standard_metrics,
    'comparison': {
        'time_speedup': time_speedup,
        'detail_ratio': detail_ratio,
        'rating_diff': rating_diff,
        'feedback_ratio': feedback_ratio,
        'fast_efficiency': fast_efficiency,
        'standard_efficiency': standard_efficiency
    }
}

with open('experiment3_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("  ✓ 结果已保存: experiment3_results.json")

comparison.T.to_excel('experiment3_comparison.xlsx')
print("  ✓ Excel已保存: experiment3_comparison.xlsx")

# 使用建议
print("\n" + "=" * 80)
print("📋 使用建议:")
print("=" * 80)
print("  Fast Mode: 快速筛选、初步评估、大批量处理")
print("  Standard Mode: 正式评审、需要详细反馈、深度分析")
print("=" * 80)


# ==================== 清理显存 ====================
print("\n[清理] 释放显存...")
del reviewer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("  ✓ 显存已释放")

print("\n" + "=" * 80)
print("✅ 实验3完成！")
print("=" * 80)
