# 2025 华为杯 E 题：高速列车轴承智能故障诊断

本仓库仅保留已确认的主路线：

- 第一问：`src/q1_pipeline.py` 与 `outputs/q1/`，正式特征为 Diagnostic26；
- 第二问：`src/q2_pipeline.py`，唯一正式模型为 Diagnostic26 + MLP；
- 第三问：`src/q3_pipeline.py`，冻结 Q2 encoder 后直接调用 PythonOT/POT 的 `SinkhornLpl1Transport`；正式 POT 与稳定性重拟合均已数值收敛，A–P 仍仅是无真值候选而非定稿结论。

运行第一问：

```powershell
python .\src\q1_pipeline.py
```

第二问的干净重跑入口：

```powershell
python .\src\q2_pipeline.py
python -m unittest discover -s tests -p "test_q2_*.py"
```

第三问运行入口：

```powershell
python -m unittest discover -s tests -p "test_q3_*.py"
python .\src\q3_pipeline.py
```

第二问使用原始 `.mat` 文件作为独立统计单位：所有划分按文件分组，Scaler 只在训练折拟合，窗口仅用于训练，测试时按文件平均窗口概率。LOLO 和 Group CV 只说明源域泛化，不能替代无标签目标域的准确率。

第三问也以 MAT 文件为 OT 单位，使用源类别/文件等权质量和目标文件等权质量；不读取 A–P 标签或 PDF 答案。`outputs/q3/verification.json` 是当前数值状态的唯一判定依据。

详细状态见 [项目总体进程](项目总体进程.md) 和 [项目说明](项目说明/README.md)。
