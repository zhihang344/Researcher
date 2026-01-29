#!/bin/bash
# 运行所有实验的脚本

echo "========================================="
echo "CycleResearcher 完整实验流程"
echo "========================================="
echo ""

# 检查Python环境
if [ ! -d "../.venv" ]; then
    echo "❌ 未找到虚拟环境，请先运行: python -m venv ../.venv"
    exit 1
fi

# 激活虚拟环境
source ../.venv/bin/activate
echo "✓ 虚拟环境已激活"
echo ""

# 实验1：模型对比
echo "========================================="
echo "实验1: 模型对比"
echo "========================================="
python experiment_1_model_comparison.py
if [ $? -ne 0 ]; then
    echo "❌ 实验1失败"
    exit 1
fi
echo ""
echo "✓ 实验1完成"
echo ""
echo "按Enter继续实验2，或Ctrl+C退出..."
read

# 实验2：迭代改进
echo "========================================="
echo "实验2: 迭代改进"
echo "========================================="
python experiment_2_iterative_improvement.py
if [ $? -ne 0 ]; then
    echo "❌ 实验2失败"
    exit 1
fi
echo ""
echo "✓ 实验2完成"
echo ""
echo "按Enter继续实验3，或Ctrl+C退出..."
read

# 实验3：模式对比
echo "========================================="
echo "实验3: 模式对比"
echo "========================================="
python experiment_3_mode_comparison.py
if [ $? -ne 0 ]; then
    echo "❌ 实验3失败"
    exit 1
fi
echo ""
echo "✓ 实验3完成"
echo ""

# 汇总结果
echo "========================================="
echo "所有实验完成！"
echo "========================================="
echo ""
echo "生成的文件："
echo "  - experiment1_model_comparison.png"
echo "  - experiment1_results.json"
echo "  - experiment1_comparison.xlsx"
echo ""
echo "  - experiment2_iterative_improvement.png"
echo "  - experiment2_results.json"
echo "  - experiment2_iterations.xlsx"
echo ""
echo "  - experiment3_mode_comparison.png"
echo "  - experiment3_results.json"
echo "  - experiment3_comparison.xlsx"
echo ""
echo "🎉 可以向导师汇报了！"
