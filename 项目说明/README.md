# 项目说明

本目录只保留第一问封版验证材料及必要的第三方声明。

- 第一问：`第一问封版验证结果.md`
- 第二问：正式代码为 `../src/q2_pipeline.py`；唯一模型是 Diagnostic26 + MLP，最终全源 refit seed 固定为 2025。
- 第三问：`../src/q3_pipeline.py` 直接使用 PythonOT/POT 的 `SinkhornLpl1Transport`。本轮候选输出位于 `../outputs/q3/`；正式和稳定性重拟合均以固定数值设置收敛，但候选并不等同于无标签目标域真值或准确率。

历史的第三问、迁移特征、多模型比较和旧候选标签均已删除，不能作为当前结论引用。
