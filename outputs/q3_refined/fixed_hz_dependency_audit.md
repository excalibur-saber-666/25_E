# 固定 Hz 包络依赖审计

消融只在源域文件级 LOLO Logistic 探针上完成，未读取 A–P 标签或候选结果。

```csv
ablation,feature_count,macro_f1,balanced_accuracy,evaluation
full,30,0.873015873015873,0.9017857142857143,source file-level LOLO logistic probe only
no_absolute_hz,12,0.7473544973544974,0.7797619047619048,source file-level LOLO logistic probe only
no_envelope,10,0.6382095410628019,0.6845238095238095,source file-level LOLO logistic probe only
no_order,9,0.6897368421052632,0.7142857142857143,source file-level LOLO logistic probe only
```

`no_absolute_hz` 删除三段固定 Hz 包络特征；`no_envelope` 还删除不依赖固定 Hz 的全频包络描述；`no_order` 删除所有阶次描述。四组 schema 不同。

固定 Hz 包络可能编码 CWRU 传感器/结构共振，因此其源域增益不能被解释为跨设备正确性。正式 Transfer-v2 暂保留 full schema，以预先固定的源域可辨识性作为工程基线；最终目标结论保持候选级，且 no_absolute_hz 是必须报告的设备依赖敏感性，而非按 A–P 结果选择。
