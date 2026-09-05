# 2025 Huawei Cup E — bearing-fault transfer analysis

本仓库包含华为杯 E 题材料、数据审计代码和第一问的可复现实验结果。

第一问已封版，第二问源域诊断及其深入完善入口见下文。目标文件 A–P 无真值标签：

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

第二问深入完善（五种子、文件级统计、训练内校准、消融及 Transfer20 初始化）：

```powershell
python .\src\q2_pipeline.py --output-dir outputs\q2_recheck_baseline
python .\src\q2_robustness.py --stage all
python -m unittest discover -s tests -p "test_q2_*.py"
```

新结果保存在 `outputs/q2_refined/`，原 `outputs/q2/` 保留为基线。续接请先阅读
[第二问深入完善交接](项目说明/第二问深入完善交接.md) 和
[实际结果报告](outputs/q2_refined/q2_refined_summary.md)。
第三问只考虑通过门槛和接口检查的 `q2_transfer20_*` 初始化材料；旧版 26D encoder 不适配 20D Transfer 输入。

关键约束：所有训练/验证必须按原始 `.mat` 文件分组；重叠窗口不是独立样本；没有目标真值和目标轴承几何时，不报告目标准确率或强行标注 BPFO/BPFI/BSF。
