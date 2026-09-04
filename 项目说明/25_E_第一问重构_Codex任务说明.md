# 第一问重构与代码修改任务说明

## 任务目标

基于当前项目中的 `bearing_mvp.py`、`dann_feature_baseline.py` 及现有运行结果，对**第一问代码与实验流程进行重构**。

当前目标不是继续调 DANN/CORAL，也不是提前给目标域 A-P 分类，而是先把第一问真正要求的内容做扎实：

1. 系统筛选适合作为后续迁移学习源域的数据；
2. 完成滚动轴承故障机理分析；
3. 建立规范的数据预处理和文件级 metadata；
4. 提取具有物理意义、且尽量具有跨工况/跨域稳定性的特征；
5. 形成后续第二问和第三问能够直接使用的标准源域特征数据集；
6. 自动生成第一问论文所需的图表、统计表和中间结果。

---

# 项目背景与当前状态

项目仓库：

- https://github.com/excalibur-saber-666/25_E

当前数据目录：

```text
数据集/数据集/
├─ 源域数据集/
│  ├─ 12kHz_DE_data
│  ├─ 12kHz_FE_data
│  ├─ 48kHz_DE_data
│  └─ 48kHz_Normal_data
└─ 目标域数据集/
   ├─ A.mat
   ├─ ...
   └─ P.mat
```

当前已有代码：

- `bearing_mvp.py`
- `dann_feature_baseline.py`

当前已有输出包括：

- `data_manifest.csv`
- `source_file_cv_predictions.csv`
- `target_predictions_source_only.csv`
- `target_predictions_coral.csv`
- `target_prediction_comparison.csv`
- `target_candidates.csv`
- `three_method_comparison.csv`
- `training_history.csv`
- `summary.json`

当前 `bearing_mvp.py` 的核心方案为：

```text
16 个受控源文件
= N×4 + B007×4 + IR007×4 + OR007@6×4
        ↓
48 kHz → 32 kHz
        ↓
16384 点滑窗
        ↓
16 维人工特征
        ↓
RBF-SVM
        ↓
Source-only / CORAL
        ↓
目标域 A-P 候选分类
```

`dann_feature_baseline.py` 则在同一 16 文件受控源集上进一步训练 feature-level DANN。

---

# 已知问题

## 1. 16 文件源集过于简单，导致源域验证结果失真

当前源域文件级交叉验证结果为：

```text
Macro-F1 = 1.0
Balanced Accuracy = 1.0

N recall  = 1.0
OR recall = 1.0
IR recall = 1.0
B recall  = 1.0
```

该结果并不能证明模型具有良好的跨工况或跨域泛化能力。

原因是当前四类分别固定为：

- B：只使用 `B007`
- IR：只使用 `IR007`
- OR：只使用 `OR007@6`
- N：4 个 Normal 文件

即故障尺寸、OR 位置等关键变化因素被固定，任务本身过于容易。

### 修改要求

保留这 16 个文件，但重新定义用途：

```text
mechanism_subset
```

仅用于：

- 故障机理展示；
- 受控条件对比；
- 时域/频域/包络谱典型图。

不得继续把它作为正式源域训练集的唯一数据来源。

---

## 2. 正式源域应扩展到完整 48 kHz DE + Normal

正式第一问主源域改为：

```text
48kHz_DE_data 全部故障文件
+
48kHz_Normal_data 全部正常文件
```

要求保留：

- B 多种故障尺寸；
- IR 多种故障尺寸；
- OR 多故障尺寸；
- OR 不同故障位置；
- 0/1/2/3 hp 多载荷；
- Normal 全部 48 kHz 文件。

请自动扫描目录，不要硬编码最终文件数量。

同时生成实际统计结果，确认：

```text
N / B / IR / OR
各类别文件数
各故障尺寸数量
各载荷数量
OR 各位置数量
RPM 范围
```

---

## 3. 当前目标域结果明显不稳定，暂时不适合继续调迁移模型

现有输出中：

### Source-only

A-P 16 个目标文件：

```text
16/16 全部预测为 N
```

并且：

```text
16/16 全部被 OOD 检测标记为 out_of_distribution=True
```

当前 OOD 阈值约：

```text
25.24
```

目标域平均 Mahalanobis 距离范围约：

```text
49.7 ~ 22391
```

说明当前目标域整体明显位于源域特征分布之外。

因此不能把 Source-only 的 N 预测理解为“目标域多数正常”。

---

### CORAL

当前 CORAL 目标预测分布约为：

```text
N  = 7
B  = 4
IR = 3
OR = 2
```

但是当前代码存在评价逻辑问题：

`predict_target()` 在 CORAL 调用中没有传入 OOD detector 和 OOD threshold，因此 CORAL 输出中的：

```text
out_of_distribution=False
```

不能解释为“CORAL 已经通过 OOD 检验”，实际上只是没有执行该检测。

### 修改要求

第一问中先暂停 CORAL 目标分类。

如果后续继续保留 CORAL baseline，则必须重新构建：

- 对齐空间中的 OOD reference；
- 合理的 source/target covariance 诊断；
- CORAL 前后 MMD/PAD 对比。

不得因为 CORAL 置信度较高就宣称迁移有效。

---

## 4. 当前 DANN 存在明显类别塌缩风险

现有 DANN 对 A-P 的预测分布约为：

```text
N  = 6
B  = 5
OR = 5
IR = 0
```

即 IR 类完全消失。

同时训练末期大致出现：

```text
classification_loss ≈ 0.0366
domain_loss         ≈ 0.6886
domain_accuracy     ≈ 0.521
```

说明：

- 源域分类任务已经高度拟合；
- 域分类器接近随机；
- 但目标类别结构并未得到可信保持。

这属于典型的：

```text
全局域对齐看似成功
≠
类别条件分布正确对齐
```

### 修改要求

第一问阶段停止继续调 DANN。

`dann_feature_baseline.py` 暂时保留，不删除，但不作为第一问主流程。

第三问重新使用时再进行独立重构。

---

## 5. 当前特征集过于粗糙

当前正式分类特征只有约 16 维，主要包括：

- 归一化均值/偏度/峭度；
- 峰值因子；
- 脉冲因子；
- 裕度因子；
- 包络峭度；
- 6 个很宽的阶次频带能量。

存在两个问题。

### 问题 A：真实幅值信息被弱化

当前先做 median/MAD 窗口归一化，再从归一化信号计算大量特征。

真实 RMS 虽然计算了，但只被放入 diagnostics，没有进入正式特征向量。

需要改为：

```text
原始物理幅值特征
+
归一化后的形态特征
```

两套信息同时保留。

---

### 问题 B：阶次特征过粗

当前只有：

```text
0.5-2
2-4
4-8
8-16
16-32
32-64
```

六个宽阶次区间。

这会抹平：

- 周期冲击；
- 包络峰结构；
- 特征阶次附近能量；
- 调制侧带；
- 自相关周期性。

第一问需要补充更细致、但不过分依赖目标轴承几何参数的机理特征。

---

# 重点检查内容

在正式修改前，先自动检查整个数据集，输出检查报告。

至少确认：

1. 所有 `.mat` 文件路径；
2. 每个文件中真实变量名；
3. DE / FE / BA 信号变量情况；
4. RPM 是否存在；
5. 无 RPM 文件能否从文件名恢复；
6. 采样率来源；
7. 信号长度；
8. 是否存在异常或空文件；
9. 48 kHz DE + Normal 实际总文件数；
10. 12 kHz DE / FE 数据的实际结构；
11. 目标 A-P 每个文件长度和变量名。

将检查结果保存为：

```text
outputs/q1/data_audit.csv
outputs/q1/data_audit_summary.json
```

---

# 修改范围

## 一、重构 `bearing_mvp.py`

不要继续让 `bearing_mvp.py` 同时承担：

- 第一问特征分析；
- SVM；
- CORAL；
- 目标域分类。

建议将第一问拆为清晰模块。

可按以下结构重构：

```text
src/
或现有项目根目录下：

q1_data_audit.py
q1_source_selection.py
q1_preprocess.py
q1_features.py
q1_mechanism_analysis.py
q1_feature_selection.py
q1_pipeline.py
```

如果不希望增加过多文件，也可以在现有项目结构内合理合并，但职责必须清晰。

---

## 二、保留两个源域集合

### A. mechanism_subset

用途：

- 受控机理对比；
- 绘图；
- 解释 N/B/IR/OR 的典型特征。

建议仍采用：

```text
N ×4
B007 ×4
IR007 ×4
OR007@6 ×4
```

但仅作为展示集。

---

### B. full_source

正式源域：

```text
48kHz_DE_data
+
48kHz_Normal_data
```

用于：

- 正式特征提取；
- 特征质量分析；
- 第二问模型输入；
- 第三问源域输入。

禁止为了“类别平衡”删除大量原始故障文件。

类别不平衡由第二问：

- class weight；
- file-level sampler；
- balanced sampling；

处理。

---

# 源域选择需要增加定量依据

第一问题目要求“根据目标域迁移需求筛选源域数据”，因此不能只依赖：

> DE 更靠近故障轴承，所以选 48 kHz DE。

需要增加一个**候选源域与目标域的无标签分布差异比较**。

候选集：

```text
12kHz_DE
12kHz_FE
48kHz_DE
```

注意：该比较必须避免采样率和带宽混杂。

---

## 公平比较规则

为了比较三个候选源域与目标域的距离，单独建立：

```text
source_selection_view
```

该视图只用于“选源域”，不作为最终建模数据。

统一到：

```text
fs = 12 kHz
有效频带 <= 5~6 kHz
```

具体处理：

```text
48 kHz DE → 12 kHz
目标 32 kHz → 12 kHz
12 kHz DE 保持不变
12 kHz FE 保持不变
```

必须使用抗混叠重采样。

然后从所有域提取相同的、与标签无关的通用统计特征，例如：

- RMS；
- 峭度；
- 峰值因子；
- 频谱熵；
- 谱质心；
- 分频带能量；
- 包络谱熵；
- 自相关峰；
- 通用阶次/周期统计量。

计算：

```text
MMD
Proxy A-distance
```

推荐同时输出：

```text
source_domain_selection.csv
```

字段示例：

```text
candidate_domain
sample_rate_common
common_band
mmd
proxy_a_distance
n_files
notes
```

### 重要限制

MMD/PAD 只作为“分布接近程度”的辅助依据。

最终源域选择需要结合：

1. 测点物理位置；
2. 数据完整性；
3. 采样带宽；
4. 与目标域的无标签分布距离。

不要只根据一个 MMD 数值机械选源域。

---

# 正式源域预处理

主源域仍采用 48 kHz DE + Normal。

正式建模视图统一到：

```text
fs = 32 kHz
```

使用：

```python
scipy.signal.resample_poly
```

进行：

```text
48 kHz → 32 kHz
```

基础预处理：

1. 去均值；
2. 去线性趋势；
3. 异常值检查；
4. 滑窗；
5. 对需要形态归一化的分支做 median/MAD 或 z-score；
6. 同时保留未做幅值归一化的原始物理统计量。

不要加入未经验证的复杂滤波或降噪算法。

---

# 分窗策略

第一版保留：

```text
WINDOW = 16384
HOP    = 8192
fs     = 32000
```

即：

```text
0.512 s
50% overlap
```

但增加窗口敏感性分析：

```text
8192
16384
32768
```

第一问不要求为了模型精度大量调参，只需比较：

- 特征稳定性；
- 类内方差；
- 类间距离；
- 文件级特征一致性。

最终给出一个有依据的窗口选择。

---

# 数据泄漏硬约束

必须始终保存：

```text
file_id
```

同一原始 `.mat` 文件产生的所有窗口在后续任何监督训练/验证中必须属于同一组。

禁止：

```text
全部窗口随机 shuffle
→ 再 split train/test
```

所有 scaler、PCA、特征选择器等在第二问训练时必须只在训练 fold 拟合。

第一问生成完整特征表时可保存原始特征，但不要提前使用全体标签对数据做不可逆监督筛选后再声称独立验证。

---

# 第一问特征体系重构

第一问目标不是堆几百个特征，而是建立一套：

```text
物理幅值
+
波形形态
+
频域
+
包络域
+
周期性
+
转速归一化
+
自动统计候选特征
```

的结构化特征集。

---

## A. 原始物理幅值特征

必须直接从未做幅值标准化的 detrended signal 提取：

- RMS
- 标准差
- 方差
- 峰值
- 峰峰值
- 绝对平均值
- 能量

这些特征不能因为 median/MAD normalization 被丢失。

---

## B. 归一化形态特征

从归一化信号提取：

- skewness
- kurtosis
- crest factor
- impulse factor
- margin factor
- shape factor
- zero crossing rate

---

## C. 频域特征

使用 Welch PSD，至少加入：

- spectral centroid
- spectral RMS frequency
- spectral bandwidth
- spectral entropy
- dominant frequency
- dominant peak ratio
- multiple band energies
- normalized band-energy ratios

频带划分应有明确依据，不要随意堆过多区间。

---

## D. 包络域特征

使用 Hilbert envelope。

至少包括：

- envelope RMS
- envelope kurtosis
- envelope crest factor
- envelope spectral entropy
- envelope dominant peak
- envelope peak concentration
- envelope band-energy ratios

---

## E. 周期性特征

至少加入：

- autocorrelation first significant peak
- autocorrelation peak magnitude
- dominant period
- periodicity strength
- impact interval variability

用于描述周期冲击，而不是只靠宽阶次带能量。

---

## F. 阶次/转速归一化特征

源域使用每个文件真实 RPM：

\[
f_r = RPM/60
\]

建立：

\[
Order=f/f_r
\]

提取：

- order spectral entropy
- order dominant peak
- several order-band energies
- envelope-order peak concentration
- rotational-order modulation features

注意：

目标域题面只给约 600 rpm，没有精确逐时转速。

因此在第一问和后续论文中只能称：

```text
基于名义转速/文件转速的阶次归一化特征
```

不得写成：

```text
精确同步角域重采样
```

除非后续确实建立了可靠目标转速估计方法并验证。

---

# 故障机理分析

使用 `mechanism_subset`。

每一类选择代表样本：

```text
N
B007
IR007
OR007@6
```

尽量控制相同或相近负载。

自动绘制：

1. 时域波形；
2. Welch PSD；
3. Hilbert 包络谱；
4. 自相关曲线；
5. 包络阶次谱。

源域已知轴承参数和 RPM 时，计算并标注：

- BPFO
- BPFI
- BSF
- FTF
- 转频及必要倍频。

### BSF 注意事项

检查题面 BSF 公式与常见文献定义可能存在的 1/2 因子差异。

不要静默修改题面公式。

代码层面允许同时计算并展示：

```text
BSF_standard
2 × BSF_standard
```

并在输出说明中明确：

> 不同资料对滚动体旋转频率/冲击频率的定义可能存在倍频差异，实际解释结合包络谱观测。

---

# tsfresh 接入

增加 `tsfresh` 作为自动候选特征补充。

参考：

- https://github.com/blue-yonder/tsfresh

但不要把全部自动特征直接用于第二问。

建议：

```text
人工特征
+
tsfresh candidate features
        ↓
低方差删除
        ↓
高相关删除
        ↓
稳定性/互信息/重要性筛选
```

目标最终保留：

```text
约 30~50 个核心特征
```

不是硬性维数，若数据支持 40~70 维也可以，但必须说明筛选原则。

---

# 特征筛选原则

第一问优先做**可解释、不过度监督化**的筛选。

建议输出：

1. 缺失率；
2. 方差；
3. 类内变异系数；
4. Pearson/Spearman 相关性；
5. 与标签互信息（仅作为辅助）；
6. 文件级稳定性；
7. 重要性排序。

高相关冗余初始阈值可用：

```text
|r| > 0.95
```

保留更具物理意义或更稳定的一个。

---

# 禁止修改内容

1. 不得人为获取或使用目标域 A-P 的真实标签；
2. 不得利用网上泄露/论文披露的 A-P 标签进行调参或验证；
3. 不得在第一问声称目标域准确率；
4. 不得为了结果好看强制目标域四类数量均衡；
5. 不得为了平衡源域而删除大量原始故障文件；
6. 不得随机窗口级拆分造成文件泄漏；
7. 不得把漂亮的 t-SNE/UMAP 当作分类正确性的证明；
8. 不得继续以 16 文件受控子集的 100% 准确率作为主结果；
9. 不得因为 CORAL/DANN 置信度较高而直接写“迁移成功”；
10. 暂时不要引入 LMMD、CDAN、图神经网络等额外复杂迁移模型。

---

# 硬性技术要求

## 1. 输出目录统一

第一问统一输出到：

```text
outputs/q1/
```

建议结构：

```text
outputs/q1/
├─ audit/
├─ metadata/
├─ source_selection/
├─ mechanism/
├─ features/
├─ figures/
└─ reports/
```

---

## 2. 所有图必须可复现

每张图保存：

```text
PNG
```

建议同时保存：

```text
PDF 或 SVG
```

如环境方便。

所有图必须：

- 白底；
- 标题、坐标轴完整；
- 图例清楚；
- 单位明确；
- 文件名可辨识；
- 不依赖交互式界面。

---

## 3. 代码必须支持命令行运行

至少提供：

```bash
python q1_pipeline.py --data-root "数据集/数据集"
```

允许通过参数设置：

- output-dir
- window
- hop
- target-fs
- random-seed

---

## 4. 不破坏原始数据

只读取：

```text
数据集/数据集/
```

不得修改、重命名、覆盖任何原始 `.mat` 文件。

---

# 自主执行权限

Codex 可以自主：

1. 检查当前仓库结构；
2. 阅读并重构 `bearing_mvp.py`；
3. 新建第一问模块；
4. 调整函数组织；
5. 安装/声明必要 Python 依赖；
6. 运行非破坏性脚本；
7. 自动检查 `.mat` 文件；
8. 生成 CSV / JSON / 图表；
9. 修复运行错误；
10. 对窗口长度、特征集做小规模合理敏感性实验。

但不要自行扩展到第二问/第三问完整模型开发。

如发现本任务说明与真实数据结构不一致，以真实数据为准，并在最终报告中明确说明修改原因。

---

# 验证标准

第一问重构完成后，至少满足以下条件。

## A. 数据审计

能够输出：

```text
data_audit.csv
data_audit_summary.json
```

并准确统计所有源域和目标域文件。

---

## B. 源域筛选

生成：

```text
source_domain_selection.csv
```

至少比较：

```text
12kHz_DE
12kHz_FE
48kHz_DE
```

与目标域在公平公共频带/采样率条件下的：

```text
MMD
Proxy A-distance
```

并给出源域选择说明。

---

## C. 主源域 metadata

生成：

```text
source_metadata.csv
```

字段至少包括：

```text
file_id
file_path
label
load
rpm
fault_size
fault_position
sampling_rate
signal_length
```

---

## D. 特征输出

生成：

```text
features_source_raw.csv
features_source_selected.csv
feature_quality.csv
feature_names.json
```

其中每条窗口必须带：

```text
file_id
window_id
label
```

---

## E. 机理结果

至少生成：

- 四类时域典型图；
- 四类 PSD；
- 四类 Hilbert 包络谱；
- 理论故障频率标注图；
- 自相关/周期性图；
- 包络阶次谱图。

---

## F. 特征质量分析

至少生成：

- 特征相关性热图；
- 特征稳定性统计；
- PCA 或 UMAP 源域可视化；
- 特征筛选前后维数说明。

PCA/UMAP 只用于展示，不作为最终准确率证据。

---

## G. 不再以目标分类作为第一问成功标准

第一问完成标准是：

```text
数据筛选合理
+
机理解释成立
+
特征数据完整
+
代码可复现
```

而不是：

```text
A-P 分类置信度高
```

---

# 最终交付要求

请完成代码修改后提交以下内容。

## 1. 修改后的代码

至少包含：

```text
q1_pipeline.py
```

以及必要模块。

如保留：

```text
bearing_mvp.py
dann_feature_baseline.py
```

请明确说明：

- 哪些函数继续复用；
- 哪些功能已废弃；
- DANN 为什么暂不属于第一问主流程。

---

## 2. 运行说明

新增或更新 README，给出：

```bash
pip install ...
python q1_pipeline.py ...
```

完整运行步骤。

---

## 3. 第一问输出文件

至少包括：

```text
outputs/q1/data_audit.csv
outputs/q1/data_audit_summary.json
outputs/q1/source_domain_selection.csv
outputs/q1/source_metadata.csv
outputs/q1/features_source_raw.csv
outputs/q1/features_source_selected.csv
outputs/q1/feature_quality.csv
outputs/q1/feature_names.json
outputs/q1/figures/*
```

---

## 4. 第一问总结报告

生成：

```text
outputs/q1/q1_summary.md
```

必须说明：

1. 实际扫描到的数据数量；
2. 最终选择了哪些源域数据；
3. 为什么选择；
4. 预处理流程；
5. 分窗方案；
6. 提取了哪些特征；
7. 哪些特征被删除；
8. 机理图显示了什么；
9. 目前还存在哪些不确定性；
10. 哪些内容留到第二问/第三问。

---

# 当前阶段结论

当前 16 文件 + 16 维特征的方案可以保留为一个**受控 MVP 和故障机理演示基线**，但不能作为第一问最终方案。

当前 Source-only、CORAL、DANN 在目标域上的明显分歧已经表明：

```text
源域内部容易区分
≠
特征具有可靠跨域迁移能力
```

因此当前优先级是：

```text
完整源域
→ 公平源域筛选
→ 机理分析
→ 多域特征构建
→ 特征质量验证
→ 标准源域特征数据集
```

第一问重构完成并验证后，再进入第二问分类与第三问迁移学习。
