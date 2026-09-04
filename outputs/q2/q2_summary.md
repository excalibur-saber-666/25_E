# 第二问总结

## 数据与无泄漏边界

使用第一问的 26 维 Diagnostic 特征。源域共有 56 个原始 MAT 文件，类别数为 {'OR': 28, 'B': 12, 'IR': 12, 'N': 4}。传统模型以每文件 mean+std 的 52 维特征输入；MLP 以窗口特征训练但按文件平衡采样、按文件平均概率评价。

LOLO 和辅助 Stratified Group K-Fold 的所有 Scaler、调参与训练均只在各外层训练文件内完成。结果仅说明源域跨载荷/跨文件的泛化能力，不估计 A–P 目标域准确率。

## 最优模型

按预注册的 LOLO 选择规则，最终模型为 **mlp**：Macro-F1=0.930±0.106，Balanced Accuracy=0.940±0.097，最低平均类别 Recall=0.833。

## 第三问接口

保存了最优诊断模型（如适用）以及独立 MLP Encoder、Classifier、Scaler、特征 schema 与标签顺序。MLP 全源域重训权重仅用于第三问初始化，不是新的独立测试结果。

详细表格见 `model_comparison.csv`、`lolo_results.csv`、`group_cv_results.csv` 和 `best_model_summary.json`。
