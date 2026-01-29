#!/usr/bin/env python3
"""
实验2: 迭代改进实验

验证CycleResearcher的核心机制：通过评审反馈迭代改进论文质量
- 生成初始论文
- 评审并获取反馈
- 基于反馈改进论文
- 多轮迭代，观察质量提升
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
GPU_FOR_RESEARCHER = "3"   # CycleResearcher使用的GPU
GPU_FOR_REVIEWER = "4"     # DeepReviewer使用的GPU
NUM_ITERATIONS = 3         # 迭代次数
RESEARCH_QUESTION = "How can we improve the efficiency of transformer models for long-context processing?"

print("=" * 80)
print("实验2: 迭代改进实验")
print("=" * 80)
print(f"研究问题: {RESEARCH_QUESTION}")
print(f"迭代次数: {NUM_ITERATIONS}")


# ==================== 辅助函数 ====================
def extract_metrics(review_result):
    """从CycleReviewer评审结果中提取量化指标"""
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
    
    # CycleReviewer 格式：顶层有avg_rating, soundness等字段
    import re
    
    # 提取评分
    try:
        metrics['avg_rating'] = float(review_result.get('avg_rating', 0))
    except (ValueError, TypeError):
        rating_str = str(review_result.get('rating', '0'))
        numbers = re.findall(r'\d+', rating_str)
        if numbers:
            metrics['avg_rating'] = float(numbers[0])
    
    for field, key in [('soundness', 'avg_soundness'), 
                       ('presentation', 'avg_presentation'),
                       ('contribution', 'avg_contribution')]:
        try:
            field_str = str(review_result.get(field, '0'))
            numbers = re.findall(r'\d+', field_str)
            if numbers:
                metrics[key] = float(numbers[0])
        except:
            pass
    
    # 计数weaknesses, strengths等（CycleReviewer是列表，每个元素是审稿人的文本）
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
    
    # 审稿人数量
    if 'reviews' in review_result and isinstance(review_result['reviews'], list):
        metrics['num_reviewers'] = len(review_result['reviews'])
    else:
        metrics['num_reviewers'] = 4  # CycleReviewer默认4个审稿人
    
    # 评审长度
    if 'content' in review_result:
        metrics['review_length'] = len(review_result['content'])
    
    return metrics


# ==================== 初始化模型 ====================
print("\n[1/3] 初始化模型...")

# CycleResearcher
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_FOR_RESEARCHER
from ai_researcher import CycleResearcher
print(f"  初始化 CycleResearcher (GPU {GPU_FOR_RESEARCHER})...")
researcher = CycleResearcher(
    model_size="12B",
    device="cuda",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.95,
    max_model_len=32000
)
print("  ✓ CycleResearcher 已就绪")

# CycleReviewer (用于评审)
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_FOR_REVIEWER
from ai_researcher import CycleReviewer as ReviewerModel
print(f"  初始化 CycleReviewer (用于评审) (GPU {GPU_FOR_REVIEWER})...")
reviewer = ReviewerModel(
    model_size="8B",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.95
)
print("  ✓ CycleReviewer (评审) 已就绪")


# ==================== 迭代实验 ====================
print(f"\n[2/3] 开始{NUM_ITERATIONS}轮迭代...")

iteration_results = []
current_paper = None
previous_review = None

for iteration in range(NUM_ITERATIONS):
    print(f"\n{'─' * 80}")
    print(f"Iteration {iteration}")
    print(f"{'─' * 80}")
    
    # 生成/改进论文
    if iteration == 0:
        print("  [1/2] 生成初始论文...")
        start_time = time.time()
        paper_result = researcher.generate_paper(
            topic=RESEARCH_QUESTION
        )
        gen_time = time.time() - start_time
        current_paper = paper_result[0]['latex']
        print(f"  ✓ 论文生成完成 ({gen_time:.1f}秒, {len(current_paper)}字符)")
    else:
        print("  [1/2] 基于反馈改进论文...")
        
        # 🔍 调试：查看上一轮评审结果结构
        if iteration == 1:
            print("\n  【调试】上一轮评审结果结构:")
            print(f"    类型: {type(previous_review)}")
            print(f"    键: {list(previous_review.keys()) if isinstance(previous_review, dict) else 'N/A'}")
            if 'weaknesses' in previous_review:
                print(f"    weaknesses类型: {type(previous_review['weaknesses'])}")
                print(f"    weaknesses内容: {previous_review['weaknesses'][:500] if isinstance(previous_review['weaknesses'], str) else previous_review['weaknesses']}")
            if 'content' in previous_review:
                print(f"    content长度: {len(previous_review['content'])}")
                print(f"    content前500字符:\n{previous_review['content'][:500]}")
        
        # 提取反馈（从CycleReviewer结果中）
        feedback_items = []
        if previous_review:
            import re
            # CycleReviewer的weaknesses字段是列表
            if 'weaknesses' in previous_review and isinstance(previous_review['weaknesses'], list):
                for weakness in previous_review['weaknesses']:
                    if isinstance(weakness, str) and weakness.strip():
                        feedback_items.append(weakness.strip())
            
            # 也可以从content中提取
            elif 'content' in previous_review and previous_review['content']:
                weakness_section = re.search(r'###?\s*Weaknesses?\s*\n+(.*?)(?=###|\Z)', 
                                           previous_review['content'], re.DOTALL | re.IGNORECASE)
                if weakness_section:
                    text = weakness_section.group(1).strip()
                    # 按行拆分
                    for line in text.split('\n'):
                        line = line.strip()
                        if line and len(line) > 20:
                            feedback_items.append(line.lstrip('-').lstrip('*').strip())
        
        # 构造明确的改进指令
        if feedback_items:
            improvement_instruction = (
                f"Please write an improved version of a research paper on: {RESEARCH_QUESTION}\n\n"
                f"The previous version received the following criticisms that MUST be addressed:\n"
            )
            for i, item in enumerate(feedback_items[:5], 1):  # 最多5个主要问题
                improvement_instruction += f"{i}. {item}\n"
            improvement_instruction += "\nPlease generate an improved paper that specifically addresses these issues."
        else:
            improvement_instruction = RESEARCH_QUESTION
        
        # 🔍 调试：查看提取的反馈
        if iteration == 1:
            print(f"\n  【调试】提取的反馈内容:")
            print(f"    反馈条目数: {len(feedback_items)}")
            print(f"    改进指令长度: {len(improvement_instruction)}")
            print(f"    改进指令:\n{improvement_instruction[:800]}")
        
        start_time = time.time()
        paper_result = researcher.generate_paper(
            topic=improvement_instruction
        )
        gen_time = time.time() - start_time
        current_paper = paper_result[0]['latex']
        print(f"  ✓ 论文改进完成 ({gen_time:.1f}秒, {len(current_paper)}字符)")
    
    # 评审论文
    print("  [2/2] 评审论文...")
    start_time = time.time()
    review_result = reviewer.evaluate([current_paper])[0]  # CycleReviewer不需要mode参数
    review_time = time.time() - start_time
    print(f"  ✓ 评审完成 ({review_time:.1f}秒)")
    
    # 提取指标
    metrics = extract_metrics(review_result)
    metrics['iteration'] = iteration
    metrics['generation_time'] = gen_time
    metrics['review_time'] = review_time
    
    iteration_results.append(metrics)
    previous_review = review_result
    
    # 打印关键指标
    print(f"\n  关键指标:")
    print(f"    Rating:       {metrics['avg_rating']:.2f}")
    print(f"    Soundness:    {metrics['avg_soundness']:.2f}")
    print(f"    Presentation: {metrics['avg_presentation']:.2f}")
    print(f"    Contribution: {metrics['avg_contribution']:.2f}")
    print(f"    Weaknesses:   {metrics['total_weaknesses']}")

print("\n✓ 迭代实验完成")


# ==================== 分析结果 ====================
print("\n改进分析:")
print("=" * 60)

exp2_df = pd.DataFrame(iteration_results)
exp2_df['composite_quality'] = (
    exp2_df['avg_rating'] * 0.4 + 
    exp2_df['avg_soundness'] * 0.2 + 
    exp2_df['avg_presentation'] * 0.2 + 
    exp2_df['avg_contribution'] * 0.2
)

if len(exp2_df) > 1:
    initial = exp2_df.iloc[0]
    final = exp2_df.iloc[-1]
    
    rating_change = final['avg_rating'] - initial['avg_rating']
    rating_pct = (rating_change / initial['avg_rating'] * 100) if initial['avg_rating'] > 0 else 0
    print(f"1. 总体评分: {initial['avg_rating']:.2f} → {final['avg_rating']:.2f} ({rating_change:+.2f}, {rating_pct:+.1f}%)")
    
    soundness_change = final['avg_soundness'] - initial['avg_soundness']
    print(f"2. Soundness: {initial['avg_soundness']:.2f} → {final['avg_soundness']:.2f} ({soundness_change:+.2f})")
    
    presentation_change = final['avg_presentation'] - initial['avg_presentation']
    print(f"3. Presentation: {initial['avg_presentation']:.2f} → {final['avg_presentation']:.2f} ({presentation_change:+.2f})")
    
    contribution_change = final['avg_contribution'] - initial['avg_contribution']
    print(f"4. Contribution: {initial['avg_contribution']:.2f} → {final['avg_contribution']:.2f} ({contribution_change:+.2f})")
    
    weakness_reduction = initial['total_weaknesses'] - final['total_weaknesses']
    print(f"5. 问题减少: {initial['total_weaknesses']:.0f} → {final['total_weaknesses']:.0f} ({weakness_reduction:+.0f})")
    
    quality_change = final['composite_quality'] - initial['composite_quality']
    quality_pct = (quality_change / initial['composite_quality'] * 100) if initial['composite_quality'] > 0 else 0
    print(f"6. 综合质量: {initial['composite_quality']:.2f} → {final['composite_quality']:.2f} ({quality_change:+.2f}, {quality_pct:+.1f}%)")
    
    print(f"\n结论: 迭代机制{'有效 ✓' if rating_change > 0 else '效果不明显 ✗'}")


# ==================== 可视化 ====================
print("\n[3/3] 生成可视化...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('实验2: 迭代改进分析', fontsize=16, fontweight='bold')

iterations = exp2_df['iteration'].values

# 1. 评分演变
axes[0, 0].plot(iterations, exp2_df['avg_rating'], 'o-', linewidth=2, markersize=8, label='Overall Rating')
axes[0, 0].plot(iterations, exp2_df['avg_soundness'], 's-', linewidth=2, markersize=8, label='Soundness')
axes[0, 0].plot(iterations, exp2_df['avg_presentation'], '^-', linewidth=2, markersize=8, label='Presentation')
axes[0, 0].plot(iterations, exp2_df['avg_contribution'], 'd-', linewidth=2, markersize=8, label='Contribution')
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('Score')
axes[0, 0].set_title('评分演变')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks(iterations)

# 2. 问题数量趋势
width = 0.35
axes[0, 1].bar(iterations - width/2, exp2_df['total_weaknesses'], width, label='Weaknesses', alpha=0.8)
axes[0, 1].bar(iterations + width/2, exp2_df['total_suggestions'], width, label='Suggestions', alpha=0.8)
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('问题和建议数量')
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)
axes[0, 1].set_xticks(iterations)

# 3. 改进百分比
if len(exp2_df) > 1:
    initial_rating = exp2_df.iloc[0]['avg_rating']
    rating_improvement = [(r - initial_rating) / initial_rating * 100 if initial_rating > 0 else 0 
                          for r in exp2_df['avg_rating']]
    
    axes[1, 0].bar(iterations, rating_improvement, alpha=0.8, color='green')
    axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Improvement (%)')
    axes[1, 0].set_title('相对初始版本的改进')
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].set_xticks(iterations)
    
    for i, v in enumerate(rating_improvement):
        axes[1, 0].text(iterations[i], v + 0.5, f'{v:.1f}%', ha='center', va='bottom')

# 4. 综合质量
axes[1, 1].plot(iterations, exp2_df['composite_quality'], 'o-', linewidth=3, 
                markersize=10, color='purple', label='Composite Quality')
axes[1, 1].fill_between(iterations, exp2_df['composite_quality'], alpha=0.3, color='purple')
axes[1, 1].set_xlabel('Iteration')
axes[1, 1].set_ylabel('Quality Score')
axes[1, 1].set_title('综合质量变化')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xticks(iterations)
axes[1, 1].legend()

for i, v in enumerate(exp2_df['composite_quality']):
    axes[1, 1].text(iterations[i], v + 0.05, f'{v:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('experiment2_iterative_improvement.png', dpi=300, bbox_inches='tight')
print("  ✓ 图表已保存: experiment2_iterative_improvement.png")

# 保存数据
results = {
    'timestamp': datetime.now().isoformat(),
    'research_question': RESEARCH_QUESTION,
    'num_iterations': NUM_ITERATIONS,
    'iterations': iteration_results,
    'summary': {
        'initial_rating': float(exp2_df.iloc[0]['avg_rating']) if len(exp2_df) > 0 else 0,
        'final_rating': float(exp2_df.iloc[-1]['avg_rating']) if len(exp2_df) > 0 else 0,
        'improvement': float(rating_change) if len(exp2_df) > 1 else 0
    }
}

with open('experiment2_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("  ✓ 结果已保存: experiment2_results.json")

exp2_df.to_csv('experiment_2_iterations.csv', index=False)
print("  ✓ Excel已保存: experiment2_iterations.xlsx")


# ==================== 清理显存 ====================
print("\n[清理] 释放显存...")
del researcher, reviewer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("  ✓ 显存已释放")

print("\n" + "=" * 80)
print("✅ 实验2完成！")
print("=" * 80)
