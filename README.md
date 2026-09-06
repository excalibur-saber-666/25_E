# 2025 华为杯 E 题：高速列车轴承智能故障诊断

本仓库仅保留已确认的主路线：

- 第一问：`src/q1_pipeline.py` 与 `outputs/q1/`，正式特征为 Diagnostic26；
- 第二问：`src/q2_pipeline.py`，唯一正式模型为 Diagnostic26 + MLP（待在干净目录重新运行）；
- 第三问：未开始。后续将冻结第二问 MLP encoder，并从零实现 POT Class-Regularized OT。

运行第一问：

```powershell
python .\src\q1_pipeline.py
```

第二问的干净重跑入口：

```powershell
python .\src\q2_pipeline.py
python -m unittest discover -s tests -p "test_q2_*.py"
```

第二问使用原始 `.mat` 文件作为独立统计单位：所有划分按文件分组，Scaler 只在训练折拟合，窗口仅用于训练，测试时按文件平均窗口概率。LOLO 和 Group CV 只说明源域泛化，不能替代无标签目标域的准确率。

详细状态见 [项目总体进程](项目总体进程.md) 和 [项目说明](项目说明/README.md)。
