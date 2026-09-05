# 第三问：无监督跨域候选诊断结果

目标 A–P 没有真值标签；本目录不含、也不报告目标域准确率。所有类别均是模型候选。

## 方法选择

最终候选方法：`CORAL`。DANN source-retention F1 差值（DANN−source-only）为 `0.029`，负迁移标记为 `False`；DANN 平均 seed agreement 为 `0.850`，类别塌缩标记为 `True`。

## 域差异（文件级 embedding）

| method | MMD | PAD | domain BA |
|---|---:|---:|---:|
| source_only | 0.5334 | 1.8214 | 0.9554 |
| coral | 0.1119 | 1.7857 | 0.9464 |
| dann | 0.2707 | 1.8036 | 0.9509 |

MMD/PAD 的下降只说明域表征更难区分，不能证明目标类别正确。

## 可复核输出

- `target_predictions_final.csv`：A–P 的最终候选、三方法对照和内部稳定性；
- `source_retention_summary.csv`：fold-specific 源域 LOLO 保持验证；
- `dann_training_history.csv`、`domain_metrics.csv`、embedding CSV：供第四问解释；
- `figures/`：PCA、域指标、训练曲线、方法差异、稳定性和最终候选。
