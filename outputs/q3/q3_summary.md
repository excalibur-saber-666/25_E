# 第三问：POT Class-Regularized OT

正式链路为 Diagnostic26 → 冻结 Q2 MLP encoder (32D) → PythonOT/POT `SinkhornLpl1Transport` → 线性头。正式参数为 `reg_e=0.1`、`reg_cl=1.0`，源/目标以 MAT 文件等权参与 OT。A–P 没有用于训练、调参或评分的标签；表中仅为候选诊断。

Surrogate LOLO POT：Macro-F1=0.907，BA=0.940。这只是在源域留载荷伪目标协议下的诊断，不是 A–P 准确率。MMD 从 0.1183 变为 0.0920，只描述分布接近程度。

正式 600 rpm 运行的 POT 未收敛警告数为 0；稳定性与 surrogate 重拟合累计为 0。若该值非零，相关候选仅应视为数值复核对象，不应进入解释阶段。
