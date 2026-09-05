# 第三问：CORAL 无监督跨域候选诊断结果

目标 A–P 没有真值标签；本目录不含、也不报告目标域准确率。所有类别均是模型候选。

## 正式方法

正式方法固定为 `CORAL`。其协方差采用窗口级、文件等权统计；线性头仅通过 sample_weight 进行一次类别/文件平衡。CORAL source-retention 负迁移标记为 `False`。DANN 仅为对照，其 mean-probability 与 majority-vote 标签均独立保存。

## 域差异（文件级 embedding）

| method | MMD | PAD | domain BA |
|---|---:|---:|---:|
| source_mlp | 0.5334 | 1.8214 | 0.9554 |
| source_linear | 0.5334 | 1.8214 | 0.9554 |
| coral | 0.1071 | 1.7500 | 0.9375 |
| dann | 0.2707 | 1.8036 | 0.9509 |

## CORAL 源域 LOLO 保持

| method | Macro-F1 | BA | Recall N | Recall B | Recall IR | Recall OR |
|---|---:|---:|---:|---:|---:|---:|
| coral | 0.8315 | 0.8304 | 0.7500 | 0.8333 | 0.9167 | 0.8214 |
| source_linear | 0.8773 | 0.8690 | 0.7500 | 0.9167 | 0.9167 | 0.8929 |
| source_mlp | 0.8773 | 0.8690 | 0.7500 | 0.9167 | 0.9167 | 0.8929 |

MMD/PAD 的下降只说明域表征更难区分，不能证明目标类别正确。epsilon 与 LOTO 检查只用于稳定性评估，未用于调参或选标签。

## 可复核输出

- `target_predictions_final.csv`：最终 CORAL 候选及一致性、稳定性、复核标记；
- `source_linear_predictions.csv`、`coral_source_retention_summary.csv`、`coral_epsilon_sensitivity.csv`、`coral_leave_one_target_out.csv`；
- `coral_label_changes.csv`：修正前后标签对照；
- `dann_*`：失败对照、训练历史和窗口级预测。
