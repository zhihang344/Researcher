# CycleResearcher 实验脚本使用指南

本目录包含三个独立的Python脚本，用于复现和评估CycleResearcher系统。

## 📁 文件说明

### 实验脚本
- `experiment_1_model_comparison.py` - **实验1：模型对比**
  - 比较 CycleReviewer 8B 和 DeepReviewer 7B 的性能
  - 评估维度：评分、问题发现、速度、详细程度
  
- `experiment_2_iterative_improvement.py` - **实验2：迭代改进**
  - 验证循环改进机制的有效性
  - 观察论文质量在多轮迭代中的提升
  
- `experiment_3_mode_comparison.py` - **实验3：模式对比**
  - 对比 Fast Mode 和 Standard Mode
  - 分析效率与质量的权衡

### 辅助文件
- `run_all_experiments.sh` - 一键运行所有实验
- `README_experiments.md` - 本说明文档

## 🚀 快速开始

### 方法1：单独运行实验（推荐）

```bash
# 激活虚拟环境
source ../.venv/bin/activate

# 运行实验1
python experiment_1_model_comparison.py

# 运行实验2（可选）
python experiment_2_iterative_improvement.py

# 运行实验3（可选）
python experiment_3_mode_comparison.py
```

### 方法2：一键运行全部实验

```bash
# 赋予执行权限
chmod +x run_all_experiments.sh

# 运行所有实验
./run_all_experiments.sh
```

## ⚙️ 配置说明

### GPU配置
每个实验脚本顶部都有GPU配置选项：

```python
# experiment_1_model_comparison.py
GPU_FOR_CYCLE_REVIEWER = "3"  # CycleReviewer使用的GPU
GPU_FOR_DEEP_REVIEWER = "4"   # DeepReviewer使用的GPU
USE_SEPARATE_GPUS = True      # 是否使用不同GPU

# experiment_2_iterative_improvement.py
GPU_FOR_RESEARCHER = "3"      # CycleResearcher使用的GPU
GPU_FOR_REVIEWER = "4"        # DeepReviewer使用的GPU
NUM_ITERATIONS = 3            # 迭代次数

# experiment_3_mode_comparison.py
GPU_FOR_REVIEWER = "4"        # DeepReviewer使用的GPU
NUM_REVIEWERS = 3             # Standard Mode的审稿人数量
```

### 显存优化建议

**如果显存不足：**
1. 修改 `gpu_memory_utilization` 参数（从0.75降到0.6）
2. 使用单GPU串行运行（先运行实验1，清理后再运行实验2）
3. 减少实验2的迭代次数（`NUM_ITERATIONS = 2`）

## 📊 输出结果

每个实验会生成以下文件：

### 实验1
- `experiment1_model_comparison.png` - 6子图对比可视化
- `experiment1_results.json` - 详细数据（JSON格式）
- `experiment1_comparison.xlsx` - 对比表格（Excel格式）

### 实验2
- `experiment2_iterative_improvement.png` - 4子图迭代分析
- `experiment2_results.json` - 迭代数据（JSON格式）
- `experiment2_iterations.xlsx` - 迭代表格（Excel格式）

### 实验3
- `experiment3_mode_comparison.png` - 6子图模式对比
- `experiment3_results.json` - 对比数据（JSON格式）
- `experiment3_comparison.xlsx` - 对比表格（Excel格式）

## 📈 向导师汇报的要点

### 实验1：模型对比
- **严格性**: CycleReviewer评分更低（4.75 vs 5.25），发现更多问题
- **全面性**: DeepReviewer评审更详细（2.4倍长度），多维度评分
- **效率**: CycleReviewer速度快35%

**建议**：CycleReviewer适合快速筛选，DeepReviewer适合深度评审

### 实验2：迭代改进
- **评分提升**: 观察avg_rating在迭代中的变化
- **问题减少**: Weaknesses数量应逐步降低
- **质量改进**: Soundness/Presentation/Contribution的提升

**结论**：验证了CycleResearcher的核心理念——通过反馈迭代改进

### 实验3：模式对比
- **速度**: Fast Mode显著更快
- **质量**: Standard Mode更详细、多审稿人
- **性价比**: 根据场景选择合适模式

**应用场景**：
- Fast Mode → 初筛、大批量
- Standard Mode → 正式评审、深度反馈

## 🔧 故障排除

### 问题1：显存溢出 (OOM)
**解决方案**：
- 修改GPU配置，使用不同GPU
- 降低 `gpu_memory_utilization`
- 串行运行实验

### 问题2：模型加载失败
**解决方案**：
- 检查模型路径和网络连接
- 确认模型已正确下载到HuggingFace缓存

### 问题3：找不到 generated_paper.json
**解决方案**：
- 确保在 Tutorial 目录下运行
- 检查文件是否存在

## 💡 高级用法

### 自定义研究问题（实验2）
修改 `experiment_2_iterative_improvement.py` 中的：
```python
RESEARCH_QUESTION = "Your custom research question here"
```

### 调整迭代次数（实验2）
```python
NUM_ITERATIONS = 5  # 增加到5轮迭代
```

### 修改审稿人数量（实验3）
```python
NUM_REVIEWERS = 4  # Standard Mode使用4个审稿人
```

## 📝 注意事项

1. **运行顺序**：实验可以独立运行，无需按顺序
2. **显存管理**：每个脚本运行完会自动清理显存
3. **数据依赖**：所有实验都需要 `generated_paper.json` 文件
4. **时间估算**：
   - 实验1：约3-5分钟
   - 实验2：约10-20分钟（取决于迭代次数）
   - 实验3：约2-4分钟

## 🎯 下一步工作

完成这三个实验后，你可以：
1. 使用更多论文测试（修改 `test_paper = all_papers[1]`）
2. 调整模型参数探索性能边界
3. 与人工评审结果对比（如果有）
4. 扩展到不同研究领域

## 📧 反馈与问题

如有问题，请查看：
- 终端输出的错误信息
- GPU显存使用情况（`nvidia-smi`）
- 生成的JSON文件中的详细数据

祝实验顺利！🚀
