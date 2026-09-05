# 第二问深入完善：实际结果与交接

本次只使用第一问已冻结的源域特征与 metadata。outputs/q2 保留为旧基线；本目录为完善版。

## 复现与必要修正

原基线 448 条 LOLO/Group CV 文件预测和概率逐项完全复现，56 文件、806 窗口、26 维，外层泄漏为零。
完善版保留文件级外层 LOLO、类/文件等权采样与窗口 softmax 均值。修正 sklearn 随机模型固定种子问题；
所有 MLP 候选共用同一组内层折和初始化种子。因每个 MAT 已聚合为一行，用 StratifiedKFold 直接分层文件组，
确保稀少 N 类进入每折；通常三折，校准子内层仅剩两个 N 文件时降为二折。传统模型内层也统一按 argmax(predict_proba) 计分，修复 SVM 决策函数与概率分类规则不一致。该修正会改变 seed=2025 结果，不能称其为原版同协议重跑。

## 多种子主结果

以下 ± 是五个随机种子之间的样本标准差；每个 seed 先取四个 LOLO 折的算术均值。

| 模型 | Macro-F1 | BA | F1 min–max |
|---|---|---|---|
| mlp | 0.923 ± 0.018 | 0.935 ± 0.016 | 0.896–0.942 |
| random_forest | 0.916 ± 0.016 | 0.918 ± 0.018 | 0.891–0.934 |
| logistic_regression | 0.872 ± 0.021 | 0.893 ± 0.019 | 0.859–0.908 |
| rbf_svm | 0.808 ± 0.030 | 0.827 ± 0.028 | 0.754–0.823 |

当前均值领先模型：mlp。MLP 分别胜过 LR / RF 的种子数为 {'logistic_regression': 5, 'random_forest': 3}（共五个）。
主模型推荐依据完整五种子均值；模型选择后的同批分数仍属于比较性估计，没有额外独立测试集证明获选模型的性能。
各模型每个 seed 最弱载荷计数：{'logistic_regression': {0: 5}, 'mlp': {0: 5}, 'random_forest': {0: 5}, 'rbf_svm': {0: 5}}。若载荷并列，此计数按首个最小值计；完整每折表保留。

## 错误文件与解释范围

重点文件的错误种子数如下（无错误也保留）：

| 模型 | 文件 | 错误 seeds | 常见错误类别 | 真类平均概率 |
|---|---|---|---|---|
| logistic_regression | IR014_0 | 5/5 | OR | 0.263 |
| logistic_regression | OR014@6_0 | 5/5 | B | 0.207 |
| mlp | IR014_0 | 5/5 | OR | 0.056 |
| mlp | OR014@6_0 | 5/5 | B | 0.196 |
| random_forest | IR014_0 | 5/5 | OR | 0.126 |
| random_forest | OR014@6_0 | 5/5 | B | 0.209 |
| rbf_svm | IR014_0 | 5/5 | OR | 0.242 |
| rbf_svm | OR014@6_0 | 5/5 | B | 0.358 |
| mlp | IR021_0 | 4/5 | B | 0.405 |
| logistic_regression | IR021_0 | 0/5 | 无错误 | 0.756 |
| random_forest | IR021_0 | 0/5 | 无错误 | 0.546 |
| rbf_svm | IR021_0 | 0/5 | 无错误 | 0.839 |

全部文件的反复错误、按载荷/尺寸/位置错误率、训练类中心距离、类条件特征漂移分别见 persistent_misclassifications.csv、error_factor_summary.csv、file_class_distances.csv、class_conditional_feature_shift.csv。
距离基于外层训练文件拟合的 52D 标准化空间；PCA 仅作事后解释。错误与尺寸/位置/载荷的关联不是独立因果证据，且类中心距离不是概率。

IR014_0 距离真实 IR 类中心 4.645、最近其他类中心 3.887（最近类别 OR）；OR014@6_0 最近类别为 B。这支持局部特征重叠与错误有关。
但 IR021_0 最近中心仍为 IR，中心距离无法解释其多数种子误判。需要保留非线性边界、样本稀少和训练随机性的可能性。
0 hp IR 组相对其他载荷 IR 组的最大标准化位移（分母为训练文件整体 SD）为：psd_peak_ratio=-1.465, order_4_8=1.356, psd_entropy=1.033。每折测试 IR 只有三个文件，无法据此确定物理因果。
OR014@6_3 的跨种子持续错误说明薄弱点不限于 0 hp。metadata_load_by_*.csv 确认四个载荷具有相同的类别、故障尺寸和外圈位置计数，不支持“0 hp 某种尺寸/位置样本数量少”这个解释。但同类内部位置/尺寸差异、转速变化及物理试验关联仍限制因果归因。

## Bootstrap 与模型差异

每个模型每个 seed 的 56 个 OOF 文件按真实类别重采样 2000 次；模型及 seed 使用相同文件抽样索引。
同时报告每 seed 区间与 seed 平均 pooled 指标区间。这里先合并四折计算 pooled F1，再平均 seeds；与前文先平均四折 F1 定义不同。
以下差值为 seed 平均 pooled 指标；95% 为 percentile 区间，未对多个比较作校正。

| 比较 | 指标 | 差值 | 95% CI |
|---|---|---|---|
| mlp - logistic_regression | macro_f1 | 0.059 | [-0.016, 0.139] |
| mlp - logistic_regression | balanced_accuracy | 0.042 | [-0.017, 0.110] |
| mlp - random_forest | macro_f1 | 0.009 | [-0.058, 0.077] |
| mlp - random_forest | balanced_accuracy | 0.017 | [-0.049, 0.085] |

区间包含零时不能称显著优于。所有区间条件于现有训练/预测和类比例，不覆盖重训不确定性或未见载荷总体；四折模型训练集重叠，文件也可能共享物理工况。N 仅四文件且若全判对，bootstrap Recall 可能退化为 [1,1]，不表示总体 Recall 必为一。

## 校准与第三问概率边界

校准当前均值领先模型及 MLP。每个外层训练集内部再次交叉拟合，校准文件既不参与该次模型训练，也不参与超参数和早停选择。
先算文件内窗口 softmax 均值，再以 log(文件概率)/T 校准；区别于“窗口 logits/T 后平均”。T 在 [0.05,20] 的固定对数网格上以文件 NLL 选择。正 T 不改变文件 argmax。
NLL 按文件平均，Brier 为四类平方误差之和的文件均值，ECE 用十个等宽置信区间。采用建议要求平均 NLL/ECE 均改善且至少四个 seed 两者同时改善；记录所有结果，不选取有利折。

| 模型 | 校准 | NLL | Brier | ECE |
|---|---|---|---|---|
| mlp | after | 0.288 | 0.143 | 0.088 |
| mlp | before | 0.244 | 0.131 | 0.062 |

```json
{
  "mlp": {
    "recommend_source_calibration": false,
    "mean_delta": {
      "nll": 0.043975336402631515,
      "ece": 0.02595393768509898,
      "brier": 0.011929359062426093
    },
    "n_seeds_both_nll_ece_improved": 0,
    "target_calibration_validated": false
  }
}
```
内层校准文件来自已见的训练载荷，而外层是未见载荷，概率可靠性并不必然外推。当前结果支持“不采用这次校准”，不能据此断言所有校准方法普遍无效。
校准参数的详细缓存可由脚本重建，封版仓库不再跟踪；默认导出的全源模型保持未经校准。源域 T 不可直接当作目标域可信度保证；Transfer20 概率也尚未校准。

## 消融与 20D 接口

预先固定 Diagnostic26、Transfer20、去绝对幅值、去阶次、去包络五组；LR/MLP 各跑五 seed × 四折，见 feature_group_ablation.csv。未依据测试折动态筛特征。

| 特征组 | 模型 | Macro-F1 | BA | 最低类 Recall | 0 hp F1 |
|---|---|---|---|---|---|
| Diagnostic26 | logistic_regression | 0.872 ± 0.021 | 0.893 | 0.850 | 0.798 |
| Diagnostic26 | mlp | 0.923 ± 0.018 | 0.935 | 0.850 | 0.796 |
| NoAmplitude | logistic_regression | 0.877 ± 0.007 | 0.898 | 0.833 | 0.798 |
| NoAmplitude | mlp | 0.890 ± 0.020 | 0.905 | 0.817 | 0.797 |
| NoEnvelope | logistic_regression | 0.873 ± 0.004 | 0.899 | 0.833 | 0.790 |
| NoEnvelope | mlp | 0.891 ± 0.025 | 0.906 | 0.833 | 0.821 |
| NoOrder | logistic_regression | 0.860 ± 0.024 | 0.877 | 0.833 | 0.717 |
| NoOrder | mlp | 0.897 ± 0.017 | 0.910 | 0.817 | 0.770 |
| Transfer20 | logistic_regression | 0.868 ± 0.009 | 0.893 | 0.833 | 0.798 |
| Transfer20 | mlp | 0.884 ± 0.051 | 0.897 | 0.817 | 0.806 |

消融分数差异属于预定义特征组的关联性比较；未进行等效性检验，分数接近不等于证明统计等效，也不自动改变第二问冻结的 26D 主任务。
Transfer20 MLP：LOLO Macro-F1=0.884 ± 0.051，BA=0.897；最低类别均值 Recall=0.817。
eligible_for_q3_initialization=True；target_data_used=false；target_accuracy=null。
门槛 0.85/0.70 仅为工程规则。20→32→4 加载与推理检查通过；20 维特征顺序严格对应第一问 transfer schema。固定 seed=2025 在全源域重新调参与重训，未选择最好的 seed。
Diagnostic26 和 Transfer20 均独立导出 encoder/classifier/scaler/schema/config；最佳传统模型亦保存完整 pipeline。全源域重训没有新的独立测试分数。

## 实际运行与复现

```powershell
python src/q2_pipeline.py --output-dir outputs/q2_recheck_baseline
python src/q2_robustness.py --stage diagnostic
python src/q2_robustness.py --stage calibration
python src/q2_robustness.py --stage ablation
python src/q2_robustness.py --stage transfer
python src/q2_robustness.py --stage analysis
python -m unittest discover -s tests -p "test_q2_*.py"
```
可用 --stage all 一次运行。详细 runs 缓存不作为封版结果跟踪；q2_refined_config.json 记录版本、特征输入哈希、随机种子、采样与校准定义。

## 修改范围与后续工作

新增 src/q2_robustness.py、src/q2_analysis.py、src/q2_transfer_pretrain.py 及 tests；第一问源数据、标签、metadata、原 q2_pipeline.py 和 outputs/q2 基线未修改。
总体交接文件为根目录 项目总体进程.md；本报告与 项目说明/第二问深入完善交接.md 是第二问细节入口。
第二问已完成源域证据补强；第三问已完成 Source-only、CORAL、DANN 对照，正式候选与交接见 outputs/q3/ 和 项目说明/第三问实施与第四问交接.md。
源域只有 56 个独立文件、N 仅四文件；需区分各载荷数量覆盖和故障尺寸/位置的物理混杂。CWRU 跨载荷泛化不等同高速列车目标泛化。
补充审计：第一问 Diagnostic26 来自全源域方差/相关性筛选及预定可疑特征排除；MI 用作审计而未用于该筛选条件。故本轮嵌套验证只覆盖固定 schema 下第二问的拟合过程，不能声称从特征筛选开始的端到端完全独立。若需这种声明，必须把第一问数据驱动筛选纳入外层训练折重做；本次按任务要求保持第一问不变。Transfer20 则是第一问预定义的固定名单。
