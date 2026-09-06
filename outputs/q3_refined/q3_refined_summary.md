# 第三问封版前公平验证与解释链修正

最终唯一主方法：**Source-only Transfer-v2 S56**。surrogate UDA F1 exceeds both adaptation methods by predeclared 0.01。该结论只使用统一 surrogate UDA、无标签稳定性和类别塌缩规则；未读取 A–P 真值或 PDF/参考答案。

| Method | Surrogate UDA F1 | BA | Worst-load F1 | MMD | Seed stability | RPM stability | Collapse |
|---|---:|---:|---:|---:|---:|---:|---|
| source_only | 0.9696±0.0526 | 0.9702 | 0.8786 | 0.3013 | 0.762 | 0.938 | False |
| coral | 0.9422±0.0430 | 0.9524 | 0.8786 | 0.0967 | 0.850 | 0.958 | False |
| class_regularized_ot | 0.9422±0.0430 | 0.9524 | 0.8786 | 0.0991 | 0.812 | 0.979 | False |

## 解释边界

`target_class_geometry_final.csv` 在最终方法空间计算：CORAL 使用 CORAL 变换后的 source，OT 使用 OT affine 变换后的 source，source-only 不作对齐；target embedding 从不以标签参与变换。距离与最近类仅是模型几何证据，不是目标真值。

## 固定 Hz 风险

见 `fixed_hz_dependency_audit.md`。full Transfer-v2 用作预先固定的工程基线；固定 Hz 包络的设备依赖风险不能由本数据的无标签目标验证排除，因而 A–P 仍只能称候选诊断标签。

相对提交版本的 pre-fair 候选，有 2/16 个文件发生变化；逐文件表见 `target_label_changes_fair_benchmark.csv`，没有按外部答案回改。

A–P 中需复核：9/16。即使公平 benchmark 完成，也只有在稳定性复核解释充分后才能宣称第三问封版；当前不报告任何目标 accuracy/F1/recall。
