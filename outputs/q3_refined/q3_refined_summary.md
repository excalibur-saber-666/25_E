# 第三问迁移链路定向修正结果

最终唯一候选方法为 **Class-Regularized OT Transfer-v2 S56**。此选择仅由预先设定的源域 LOLO 保持门槛触发：CORAL 相对 Source-MLP 低于 -0.05 时运行 Class-Regularized OT，OT 只有达到同一门槛才取代 CORAL；没有使用 A–P 真值、PDF 或参考答案。旧 Transfer20 CORAL 保留在 `outputs/q3/` 作为基线。

Transfer-v2 使用每窗 8 个名义转数、256 samples/revolution（2048 点）的近似恒速角域重采样；因没有转速脉冲，这不是严格 order tracking。角域 Welch 分辨率固定为 0.25 order；包络特征采用 500–2000、2000–4000、4000–8000 Hz 三个固定频带。

| Method | Features | Source variant | Source retention F1 | MMD | Target class distribution | Encoder stability | RPM stability | Collapse |
|---|---|---|---:|---:|---|---:|---:|---|
| Source-only | Transfer20 | S56 | 0.8773 | 0.5334 | baseline only | NA | NA | NA |
| Old CORAL | Transfer20 | S56 | 0.8315 | 0.1071 | baseline only | NA | NA | no |
| CORAL | Transfer-v2 | S56 file-balanced | 0.9563 | 0.0857 | {'N': 0, 'B': 1, 'IR': 7, 'OR': 8} | NA | NA | False |
| CORAL | Transfer-v2 | S56 class-balanced | 0.9166 | 0.0943 | {'N': 2, 'B': 1, 'IR': 6, 'OR': 7} | 0.888 | NA | False |
| CORAL | Transfer-v2 | S16 | 0.9365 | 0.1748 | {'N': 7, 'B': 4, 'IR': 4, 'OR': 1} | NA | NA | False |
| Raw-feature CORAL | Transfer-v2 | S56 class-balanced | NA | NA | {'N': 4, 'B': 1, 'IR': 2, 'OR': 9} | NA | NA | False |
| Class-Reg OT | Transfer-v2 | S56 class-balanced | 0.9855 | 0.0796 | {'N': 1, 'B': 1, 'IR': 7, 'OR': 7} | 0.850 | 0.875 | False |

- CORAL 源域 LOLO Macro-F1：0.9166；Class-Reg OT：0.9855；Source-MLP：0.9702。这些都是源域保持验证，不是目标准确率。
- A–P 候选类别分布：{'N': 1, 'B': 1, 'IR': 7, 'OR': 7}；需复核文件数：11/16。
- 5 encoder、570/600/630 rpm、LOTO 和窗口子采样均已生成；它们只作为可靠性证据。

不得把 MMD/PAD 降低、置信度或候选标签写成目标域 accuracy/F1/recall。最终方法仍有 11/16 个文件需要复核，因此第三问只能交付候选标签，**尚不可声称完全封版或开始第四问**。
