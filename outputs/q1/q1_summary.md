# 第一问总结

第一问已封版。正式源域由 48 kHz DE 故障文件和 48 kHz Normal 文件组成，正式特征为 Diagnostic26。

所有 LOLO 以原始 MAT 文件为独立样本；窗口只用于文件内特征提取，不构成独立测试样本。结果只说明源域跨载荷诊断价值，不代表无标签目标域的准确率。

第一问不包含目标文件分类或旧迁移路线；第二问只读取 `features_source_diagnostic.csv`、`source_metadata.csv` 和 `feature_names_diagnostic.json`。
