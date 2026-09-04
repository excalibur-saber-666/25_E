# 2025 Huawei Cup E — bearing-fault transfer analysis

本仓库包含华为杯 E 题材料、数据审计代码和第一问的可复现实验结果。

当前主流程是第一问，不对无标签的目标文件 A–P 声称诊断准确率：

```powershell
python .\src\q1_pipeline.py
```

运行后将生成 `outputs/q1/` 下的数据审计、源域选择、文件级 metadata、特征数据集和图表。

请从以下文档开始阅读：

- [当前运行结果、已知问题与待反馈事项](项目说明/当前运行结果与待反馈问题.md)
- [第一问重构任务说明](项目说明/25_E_第一问重构_Codex任务说明.md)
- [第一问封版验证结果](项目说明/第一问封版验证结果.md)
- [实施评审与科学边界](项目说明/评审结论与实施说明.md)
- [详细使用说明](项目说明/README.md)

第二问源域故障诊断可运行：

```powershell
python .\src\q2_pipeline.py
```

结果位于 `outputs/q2/`；其中的源域 LOLO/Group CV 指标不代表 A–P 目标域准确率。

关键约束：所有训练/验证必须按原始 `.mat` 文件分组；重叠窗口不是独立样本；没有目标真值和目标轴承几何时，不报告目标准确率或强行标注 BPFO/BPFI/BSF。
